"""SAITS training pipeline driven from JSON-RPC parameters."""

from __future__ import annotations

import re
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import registry
from .data import (
    load_toa5_xlsx,
    parse_sensors,
    resample_to_grid,
    make_windows_segmented,
    load_env_file,
    align_env_to_grid,
    build_time_features,
)


def run_training(params: dict[str, Any], emit: Callable[[str, dict], None]) -> dict[str, Any]:
    """Train a SAITS model on the given files. Returns a model_id once the
    model is loaded into the in-process registry.

    Params (all optional except `files`):
      files             list[str]   xlsx paths to load+concat
      step              str         "5min" by default
      window            int         288 (24h at 5min)
      stride            int         72 (75% overlap)
      train_end         str         "YYYY-MM-DD" (default 2022-01-01)
      val_start         str         "YYYY-MM-DD" (default 2023-01-01)
      val_end           str         "YYYY-MM-DD" (default 2023-06-15)
      selected_columns  list[str]   subset of sensor columns to keep (default = all)
      training_windows  list[{start,end}]  explicit periods to feed to training
                                          (overrides train_end if provided)
      env_file          str         path to env xlsx (VPD/PAR/ETo). Optional.
      env_columns       list[str]   which env columns to include as features.
                                    Required if env_file is given.
      time_features     list[str]   any of doy/doy_sin/doy_cos/hod_sin/hod_cos
      model_config      dict        SAITS hyperparams (see defaults below)
      device            str         "auto" | "cuda" | "cpu"
    """
    # ----- defaults & param parsing -----
    files = params["files"]
    step = params.get("step", "5min")
    window = int(params.get("window", 288))
    stride = int(params.get("stride", 72))
    train_end = pd.Timestamp(params.get("train_end", "2022-01-01"))
    val_start = pd.Timestamp(params.get("val_start", "2023-01-01"))
    val_end   = pd.Timestamp(params.get("val_end",   "2023-06-15"))
    cfg = {
        "n_layers": 2,
        "d_model": 128,
        "n_heads": 4,
        "d_k": 32,
        "d_v": 32,
        "d_ffn": 256,
        "dropout": 0.1,
        "attn_dropout": 0.1,
        "ORT_weight": 1.0,
        "MIT_weight": 1.0,
        "batch_size": 32,
        "epochs": 30,
        "patience": 5,
        "learning_rate": 1e-3,
        "max_nan_frac": 0.5,
    }
    cfg.update(params.get("model_config") or {})

    # pypots SAITS hard-requires patience < epochs (else it raises at
    # construction). The UI lets the user set a low epoch count but never
    # sends patience, so a small epochs (<=5) would crash training with
    # "patience must be smaller than epochs". Clamp defensively.
    cfg["epochs"] = max(1, int(cfg["epochs"]))
    cfg["patience"] = max(0, min(int(cfg["patience"]), cfg["epochs"] - 1))

    device_pref = params.get("device", "auto")
    import torch  # heavy: imported here so health() stays fast
    if device_pref == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_pref

    emit("train_started", {"device": device, "config": cfg})

    # ----- load + resample + select features -----
    t0 = time.time()
    generic = bool(params.get("generic_table"))

    if generic:
        # Train on a CALCULATED dataset exported by the Rust side (Parquet/CSV)
        # — columns are arbitrary (e.g. T600_*), not the TC_/SF_ convention, so
        # features are taken from `feature_columns` (or all numeric columns).
        from .data import load_generic_table
        emit("debug", {"step": "loading_generic", "file": str(files[0])})
        df_raw = load_generic_table(files[0])

        # Merge extra files (other stations) if provided.
        #
        # Joined on the TIMESTAMP index (axis=1), i.e. their sensors become
        # ADDITIONAL COLUMNS observed at the same instants. This used to concat
        # on axis=0 (stacking rows) and then drop duplicate timestamps keeping
        # the first — since the other stations cover the same period as the main
        # one, every one of their rows was a duplicate and got dropped. The
        # feature was therefore a no-op: the extra sensors never reached the
        # model.
        extra_files = params.get("extra_files") or []
        extra_columns = set(params.get("extra_columns") or [])
        extra_added: list[str] = []
        for ef in extra_files:
            emit("debug", {"step": "loading_extra", "file": str(ef)})
            df_extra = None
            try:
                df_extra = load_generic_table(ef)
            except Exception as exc:
                try:
                    df_extra = load_and_concat([ef])
                except Exception:
                    emit("warning", {"step": "extra_file_failed", "file": str(ef), "error": str(exc)})
                    continue
            if extra_columns:
                keep = [c for c in df_extra.columns if c in extra_columns]
                if not keep:
                    emit("warning", {"step": "extra_no_matching_cols", "file": str(ef)})
                    continue
                df_extra = df_extra[keep]
            # Same name on both sides would silently collide on join; suffix the
            # incoming one so both stay addressable.
            clash = [c for c in df_extra.columns if c in df_raw.columns]
            if clash:
                df_extra = df_extra.rename(columns={c: f"{c}__x{len(extra_added)}" for c in clash})
            df_extra = df_extra[~df_extra.index.duplicated(keep="first")]
            df_raw = df_raw.join(df_extra, how="outer")
            extra_added.extend(list(df_extra.columns))
        if extra_files:
            df_raw = df_raw.sort_index()
            emit("debug", {"step": "extra_merged", "shape": list(df_raw.shape),
                           "n_extra_files": len(extra_files), "n_extra_cols": len(extra_added),
                           "extra_cols": extra_added[:40]})

        requested = params.get("feature_columns") or params.get("selected_columns")
        if requested:
            # The extra stations' columns are features too. They are not part of
            # `selected_columns` (which lists the sensors of the MAIN dataset),
            # so without this union they would be joined into the frame and then
            # dropped again when the feature list is built.
            requested = list(requested) + [c for c in extra_added if c not in requested]
            feat_cols = [c for c in requested if c in df_raw.columns]
        else:
            feat_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c]) and not str(c).startswith("Inv")]
        if not feat_cols:
            raise ValueError("Aucune colonne numérique à entraîner dans le dataset.")
        emit("debug", {"step": "generic_loaded", "shape": list(df_raw.shape), "n_features": len(feat_cols), "elapsed": round(time.time()-t0, 1)})

        # Match the resample grid to the dataset's NATIVE cadence. A calculated
        # dataset (e.g. T600 at ~30 min) resampled onto the default 5-min grid
        # would be ~80% NaN → every training window exceeds max_nan_frac and we
        # build ZERO windows. So infer the dominant step from the index, and
        # size the window to ~24 h of that cadence (unless the user fixed them).
        if "step" not in params:
            diffs = df_raw.index.to_series().diff().dropna()
            diffs = diffs[diffs > pd.Timedelta(0)]
            med = diffs.median() if len(diffs) else pd.NaT
            if pd.notna(med) and med > pd.Timedelta(0):
                secs = int(round(med.total_seconds()))
                if secs % 3600 == 0:
                    step = f"{secs // 3600}h"
                elif secs % 60 == 0:
                    step = f"{secs // 60}min"
                else:
                    step = f"{secs}s"
                if "window" not in params:
                    steps_per_day = max(1, int(round(pd.Timedelta("1D") / med)))
                    window = int(min(512, max(24, steps_per_day)))
                    stride = max(1, window // 4)
        emit("debug", {"step": "generic_cadence", "freq": step, "window": window, "stride": stride})
    else:
        emit("debug", {"step": "loading_files", "files": [str(f) for f in files]})

        # Per-file load with explicit progress, so we can pinpoint which file
        # (or which step inside the loader) is the bottleneck if anything hangs.
        dfs = []
        for path_in in files:
            from pathlib import Path
            p = Path(path_in)
            cache_path = p.with_suffix(".pkl.gz")
            cache_hit = (
                cache_path.exists()
                and cache_path.stat().st_mtime >= p.stat().st_mtime
            ) if p.exists() else False
            emit("debug", {
                "step": "loading_one",
                "file": str(p),
                "size_mb": round(p.stat().st_size / 1e6, 1) if p.exists() else None,
                "cache_path": str(cache_path),
                "cache_exists": cache_path.exists(),
                "cache_hit": cache_hit,
            })
            t_one = time.time()
            df_one = load_toa5_xlsx(p)
            emit("debug", {
                "step": "loaded_one",
                "file": str(p),
                "shape": list(df_one.shape),
                "elapsed": round(time.time() - t_one, 1),
            })
            dfs.append(df_one)

        # Union of columns across files (different trees / stations rarely share
        # TC_/SF_ columns; intersection would drop everything). Then collapse
        # duplicate timestamps with first-non-null. See `load_and_concat`.
        if len(dfs) == 1:
            df_raw = dfs[0]
        else:
            df_raw = pd.concat(dfs, axis=0)
            df_raw = df_raw.groupby(df_raw.index).first().sort_index()
        union_cols = list(df_raw.columns)
        emit("debug", {"step": "union_cols", "n_cols": len(union_cols)})
        emit("debug", {"step": "files_loaded", "shape": list(df_raw.shape), "elapsed": round(time.time()-t0, 1)})

        sensors = parse_sensors(list(df_raw.columns))
        feat_cols = [s.name for s in sensors]
        if not feat_cols:
            raise ValueError("No TC_/SF_ sensor columns recognised in input files.")

        # Optional user-side subset (wizard's "Capteurs détectés" step). Preserve
        # the order of `feat_cols` so it matches across train/val/save/load.
        selected = params.get("selected_columns")
        if selected:
            selected_set = set(selected)
            unknown = selected_set - set(feat_cols)
            if unknown:
                raise ValueError(f"Unknown selected columns: {sorted(unknown)[:5]}")
            feat_cols = [c for c in feat_cols if c in selected_set]
            if not feat_cols:
                raise ValueError("selected_columns filtered out all sensors.")
        emit("debug", {"step": "sensors_parsed", "n_features": len(feat_cols)})

    df = resample_to_grid(df_raw[feat_cols], step=step)
    emit("debug", {"step": "resampled", "shape": list(df.shape)})

    # Optional env features (VPD / PAR / ETo from a separate file).
    env_file = params.get("env_file")
    env_columns = params.get("env_columns") or []
    if env_file and env_columns:
        env_raw = load_env_file(env_file)
        env_aligned = align_env_to_grid(env_raw, df.index, columns=list(env_columns), step=step)
        # Suffix to avoid name collisions with sensors (unlikely but safe).
        env_aligned = env_aligned.add_prefix("env_")
        df = pd.concat([df, env_aligned], axis=1)
        feat_cols = feat_cols + list(env_aligned.columns)
        emit("debug", {"step": "env_merged", "env_cols": list(env_aligned.columns)})

    # Engineered time features (DOY / hour-of-day cyclic encodings) — SMART
    # default. The TA-group study (saits_lab/exp5) showed cyclic time features
    # are a TRADE-OFF: they rescue a *system-wide* gap (all sensors down) but
    # DEGRADE the common single-sensor fill (~2x worse RMSE) because the model
    # leans on the climatological cycle instead of the near-identical sibling
    # sensors. Real meteo predictors (VPD/PAR) rescue system-wide gaps about as
    # well (within noise) WITHOUT that single-sensor regression — the better
    # balanced choice, though time-only can edge it on system-gap & spike
    # detection. So the default is "meteo first":
    #   - env/meteo provided        -> no auto time features (meteo anchors)
    #   - tiny group (<=2 sensors)  -> add cyclic (few/no siblings, needs SOME
    #                                  anchor or a gap collapses to the mean)
    #   - otherwise                 -> none (rely on siblings; opt in if you
    #                                  expect system-wide outages)
    # An explicit `time_features` list from the caller (including []) wins.
    time_features = params.get("time_features")
    if time_features is None:
        n_sensor_feat = sum(1 for c in feat_cols if not c.startswith("env_"))
        has_env = bool(env_file and env_columns)
        if has_env or n_sensor_feat > 2:
            time_features = []
        else:
            time_features = ["hod_sin", "hod_cos", "doy_sin", "doy_cos"]
    if time_features:
        tf = build_time_features(df.index, list(time_features))
        tf = tf.add_prefix("time_")
        df = pd.concat([df, tf], axis=1)
        feat_cols = feat_cols + list(tf.columns)
        emit("debug", {"step": "time_features_added", "names": list(tf.columns)})

    # Build df_train. If the wizard gave explicit training_windows, we
    # concatenate the slices verbatim — make_windows_segmented will still
    # respect the natural gaps inside each one.
    training_windows = params.get("training_windows") or []
    if training_windows:
        slices = []
        for w in training_windows:
            s = pd.Timestamp(w["start"])
            e = pd.Timestamp(w["end"])
            slc = df.loc[s:e]
            if len(slc) > 0:
                slices.append(slc)
        if not slices:
            raise ValueError("training_windows matched no data.")
        df_train = pd.concat(slices)
        df_train = df_train[~df_train.index.duplicated(keep="first")].sort_index()
        emit("debug", {"step": "windows_applied", "n_windows": len(slices), "n_rows": int(len(df_train))})
    else:
        df_train = df.loc[:train_end]

    # Optional EXCLUSION windows (blacklist) — drop every row whose timestamp
    # falls inside any of the user-marked "bad" zones. Applied AFTER the
    # whitelist so the two modes compose cleanly: good zones define the
    # candidate pool, bad zones carve out the noisy slices inside it.
    exclude_windows = params.get("exclude_windows") or []
    if exclude_windows:
        keep = pd.Series(True, index=df_train.index)
        for w in exclude_windows:
            s = pd.Timestamp(w["start"])
            e = pd.Timestamp(w["end"])
            keep &= ~((df_train.index >= s) & (df_train.index <= e))
        n_before = int(len(df_train))
        df_train = df_train[keep]
        emit("debug", {
            "step": "excluded_applied",
            "n_exclude": len(exclude_windows),
            "n_dropped": n_before - int(len(df_train)),
            "n_rows_after": int(len(df_train)),
        })

    df_val   = df.loc[val_start:val_end]

    # Generic (calculated-dataset) training rarely matches the default 2023
    # validation window, which would leave df_val empty. Fall back to a tail
    # fractional split so early-stopping still has a validation set. This also
    # applies when the user picked explicit training_windows (zones) without a
    # matching validation window — otherwise df_val would be empty and the fit
    # would have nothing to early-stop on.
    if generic and len(df_val) == 0 and len(df_train) > 20:
        n_val = max(1, int(len(df_train) * 0.15))
        df_val = df_train.iloc[-n_val:]
        df_train = df_train.iloc[:-n_val]
        emit("debug", {"step": "generic_val_split", "n_val_rows": int(len(df_val))})

    mu  = df_train.mean()
    sig = df_train.std().replace(0, 1.0)
    emit("debug", {"step": "stats_computed", "n_train_rows": int(len(df_train)), "n_val_rows": int(len(df_val))})

    W_train = make_windows_segmented(
        df_train, feat_cols, mu, sig, window=window, stride=stride,
        max_nan_frac=cfg["max_nan_frac"],
    )
    emit("debug", {"step": "train_windows_built", "shape": list(W_train.shape)})
    W_val = make_windows_segmented(
        df_val, feat_cols, mu, sig, window=window, stride=stride,
        max_nan_frac=cfg["max_nan_frac"],
    )
    emit("debug", {"step": "val_windows_built", "shape": list(W_val.shape)})
    if len(W_train) == 0:
        raise ValueError("No training windows could be built — check files / dates / NaN ratio.")

    emit("data_ready", {
        "n_features": len(feat_cols),
        "n_train_windows": int(len(W_train)),
        "n_val_windows": int(len(W_val)),
        "load_seconds": round(time.time() - t0, 1),
    })

    # ----- build & train SAITS -----
    from pypots.imputation import SAITS
    from pypots.optim import Adam
    from pygrinder import mcar

    np.random.seed(0)
    W_val_corrupt = mcar(W_val.copy(), p=0.1) if len(W_val) > 0 else None

    saits = SAITS(
        n_steps=window,
        n_features=len(feat_cols),
        n_layers=cfg["n_layers"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_k=cfg["d_k"],
        d_v=cfg["d_v"],
        d_ffn=cfg["d_ffn"],
        dropout=cfg["dropout"],
        attn_dropout=cfg["attn_dropout"],
        ORT_weight=cfg["ORT_weight"],
        MIT_weight=cfg["MIT_weight"],
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        patience=cfg["patience"],
        optimizer=Adam(lr=cfg["learning_rate"]),
        num_workers=0,
        device=device,
        saving_path=None,
        model_saving_strategy=None,
    )

    train_set = {"X": W_train}
    val_set = {"X": W_val_corrupt, "X_ori": W_val} if W_val_corrupt is not None else None

    # Hook into pypots' (standard) logger so we can stream "epoch" events
    # to the parent (Rust app) for a live progress bar. pypots logs look:
    #   Epoch 001 - training loss (MAE): 0.95, validation MSE: 0.76
    import logging as _logging
    epoch_re = re.compile(
        r"Epoch (\d+)\s*-\s*training loss \(MAE\):\s*([\d.]+)"
        r"(?:.*validation MSE:\s*([\d.]+))?"
    )
    total_epochs = cfg["epochs"]

    class _EpochHandler(_logging.Handler):
        def emit(self, record):
            try:
                text = record.getMessage()
                m = epoch_re.search(text)
                if m:
                    emit("epoch", {
                        "epoch": int(m.group(1)),
                        "total_epochs": total_epochs,
                        "train_loss": float(m.group(2)),
                        "val_mse": float(m.group(3)) if m.group(3) else None,
                    })
            except Exception:  # noqa: BLE001
                pass

    handler = _EpochHandler()
    handler.setLevel(_logging.INFO)
    pypots_logger = _logging.getLogger("pypots")
    pypots_logger.addHandler(handler)
    # Some loggers default to WARNING and would swallow Epoch INFO lines.
    if pypots_logger.level == 0 or pypots_logger.level > _logging.INFO:
        pypots_logger.setLevel(_logging.INFO)

    t1 = time.time()
    try:
        saits.fit(train_set=train_set, val_set=val_set)
    finally:
        pypots_logger.removeHandler(handler)
    train_seconds = round(time.time() - t1, 1)
    emit("train_finished", {"train_seconds": train_seconds})

    # ----- validation score -----
    # The pypots epoch logger format is version-dependent, so instead of
    # relying on parsed log lines we compute a concrete validation score here:
    # re-impute the artificially-masked validation windows and measure the
    # error on exactly those held-out positions. Values are in z-scored units
    # (windows were standardized), so they're comparable across sensors.
    metrics = None
    try:
        if W_val_corrupt is not None and len(W_val) > 0:
            imp = np.asarray(saits.impute({"X": W_val_corrupt}))
            mask = np.isnan(W_val_corrupt) & ~np.isnan(W_val)
            n_masked = int(mask.sum())
            if n_masked > 0:
                diff = imp[mask] - W_val[mask]
                mse = float(np.mean(diff ** 2))
                mae = float(np.mean(np.abs(diff)))
                metrics = {
                    "val_mae": mae,
                    "val_mse": mse,
                    "val_rmse": float(mse ** 0.5),
                    "n_masked": n_masked,
                    "units": "z-score",
                }
                emit("debug", {"step": "val_scored", "val_rmse": round(metrics["val_rmse"], 4), "val_mae": round(mae, 4)})
    except Exception as e:  # noqa: BLE001
        emit("debug", {"step": "val_metrics_error", "error": str(e)})

    # ----- contiguous-block validation (the HONEST metric) -----
    # MCAR scatters single missing points, which massively over-states real gap
    # filling skill — the app fills *contiguous blocks*. So we also blank the
    # central half of every fully-observed val window across the SENSOR columns,
    # re-impute, and score exactly those positions. This is the number that
    # actually predicts field performance; surfaced next to the MCAR one.
    try:
        if len(W_val) > 0:
            sensor_idx = [i for i, c in enumerate(feat_cols)
                          if not c.startswith("env_") and not c.startswith("time_")]
            full = W_val[~np.isnan(W_val).any(axis=(1, 2))]
            if len(full) > 0 and sensor_idx:
                a, b = window // 4, 3 * window // 4
                Wb = full.copy()
                for si in sensor_idx:
                    Wb[:, a:b, si] = np.nan
                impb = np.asarray(saits.impute({"X": Wb}))
                d = (impb[:, a:b, :] - full[:, a:b, :])[:, :, sensor_idx]
                block_rmse = float(np.sqrt(np.mean(d ** 2)))
                if metrics is None:
                    metrics = {"units": "z-score"}
                metrics["val_block_rmse"] = block_rmse
                metrics["n_block_windows"] = int(len(full))
                emit("debug", {"step": "val_block_scored",
                               "val_block_rmse": round(block_rmse, 4)})
    except Exception as e:  # noqa: BLE001
        emit("debug", {"step": "val_block_metrics_error", "error": str(e)})

    # ----- per-column reconstruction error (DETECTION calibration) -----
    # SAITS-based detection blanks the WHOLE target column and reconstructs it
    # from the other sensors, then flags points whose error is much larger than
    # the model's TYPICAL error. So measure that typical per-column error HERE,
    # the same way detection reconstructs (blank the entire column over a held-
    # out good window), robustly (median |residual|, in z units). Stored in meta
    # and used as the detection threshold scale — far better than thresholding
    # the residual *distribution*, which collapses on near-constant channels and
    # flags normal points (the 229/321 false positives).
    recon_error_z: dict[str, float] = {}
    try:
        if len(W_val) > 0:
            full = W_val[~np.isnan(W_val).any(axis=(1, 2))]
            if len(full) > 0:
                for ci, col in enumerate(feat_cols):
                    if col.startswith("env_") or col.startswith("time_"):
                        continue
                    Wb = full.copy()
                    Wb[:, :, ci] = np.nan  # blank the entire column (detection mode)
                    impc = np.asarray(saits.impute({"X": Wb}))
                    resid = impc[:, :, ci] - full[:, :, ci]
                    recon_error_z[col] = float(np.median(np.abs(resid)) * 1.4826)
                emit("debug", {"step": "recon_error_scored", "n_cols": len(recon_error_z)})
    except Exception as e:  # noqa: BLE001
        emit("debug", {"step": "recon_error_metrics_error", "error": str(e)})

    # ----- register & return -----
    model_id = registry.new_id()
    entry = registry.ModelEntry(
        model_id=model_id,
        saits=saits,
        feat_cols=feat_cols,
        mu=mu.tolist(),
        sig=sig.tolist(),
        window=window,
        sampling_step=step,
        config=cfg,
        meta={
            "files": [str(f) for f in files],
            "train_period": [str(df_train.index.min()), str(df_train.index.max())],
            "val_period":   [str(df_val.index.min()),   str(df_val.index.max())] if len(df_val) else None,
            "n_train_windows": int(len(W_train)),
            "n_val_windows": int(len(W_val)),
            "train_seconds": train_seconds,
            "device": device,
            "selected_columns": list(feat_cols),
            "training_windows": training_windows or None,
            # Auxiliary feature spec — predict() needs these to rebuild the
            # exact same feature matrix at inference time.
            "env_file": str(env_file) if env_file else None,
            "env_columns": list(env_columns) if env_columns else [],
            "time_features": list(time_features) if time_features else [],
            "metrics": metrics,
            # Per-sensor typical reconstruction error (z units), used to
            # calibrate SAITS-based outlier detection. See run_predict.
            "recon_error_z": recon_error_z,
        },
    )
    registry.register(entry)

    return {
        "model_id": model_id,
        "n_features": len(feat_cols),
        "feat_cols": feat_cols,
        "n_train_windows": int(len(W_train)),
        "n_val_windows": int(len(W_val)),
        "train_seconds": train_seconds,
        "device": device,
        "metrics": metrics,
        "epochs_configured": cfg["epochs"],
    }
