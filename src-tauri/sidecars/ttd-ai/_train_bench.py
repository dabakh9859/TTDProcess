"""Time each phase of run_training to find the bottleneck the user is hitting."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# pandas/openpyxl before torch on Windows
from ai import data as data_mod  # noqa
import torch  # noqa


FILES = [
    r"c:\Users\Aziz\Documents\apps\Data\FS_Niakhar_1_20201106-20211230.xlsx",
    r"c:\Users\Aziz\Documents\apps\Data\Niakhar 1 continu 2023.xlsx",
]


def main() -> int:
    print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}", flush=True)

    t0 = time.time()
    df = data_mod.load_and_concat(FILES)
    print(f"[{time.time()-t0:.1f}s] load_and_concat: shape={df.shape}", flush=True)

    sensors = data_mod.parse_sensors(list(df.columns))
    feat_cols = [s.name for s in sensors]
    print(f"feat_cols: {len(feat_cols)}", flush=True)

    t1 = time.time()
    df5 = data_mod.resample_to_grid(df[feat_cols], "5min")
    print(f"[{time.time()-t1:.1f}s] resample: shape={df5.shape}", flush=True)

    import pandas as pd
    df_train = df5.loc[:pd.Timestamp("2022-01-01")]
    df_val = df5.loc[pd.Timestamp("2023-01-01"):pd.Timestamp("2023-06-15")]

    mu = df_train.mean()
    sig = df_train.std().replace(0, 1.0)

    t2 = time.time()
    W_train = data_mod.make_windows_segmented(
        df_train, feat_cols, mu, sig, window=288, stride=72, max_nan_frac=0.5,
    )
    W_val = data_mod.make_windows_segmented(
        df_val, feat_cols, mu, sig, window=288, stride=72, max_nan_frac=0.5,
    )
    print(f"[{time.time()-t2:.1f}s] make_windows: train={W_train.shape}, val={W_val.shape}", flush=True)

    if len(W_train) == 0:
        print("NO TRAIN WINDOWS — abort")
        return 1

    from pypots.imputation import SAITS
    from pypots.optim import Adam
    from pygrinder import mcar
    import numpy as np

    np.random.seed(0)
    W_val_corrupt = mcar(W_val.copy(), p=0.1) if len(W_val) > 0 else None

    saits = SAITS(
        n_steps=288, n_features=len(feat_cols),
        n_layers=2, d_model=128, n_heads=4, d_k=32, d_v=32, d_ffn=256,
        dropout=0.1, attn_dropout=0.1,
        ORT_weight=1.0, MIT_weight=1.0,
        batch_size=32,
        epochs=30,       # match the sidecar default
        patience=5,
        optimizer=Adam(lr=1e-3),
        num_workers=0,
        device="cuda",
        saving_path=None,
        model_saving_strategy=None,
    )

    t3 = time.time()
    train_set = {"X": W_train}
    val_set = {"X": W_val_corrupt, "X_ori": W_val} if W_val_corrupt is not None else None
    saits.fit(train_set=train_set, val_set=val_set)
    print(f"[{time.time()-t3:.1f}s] SAITS fit (3 epochs)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
