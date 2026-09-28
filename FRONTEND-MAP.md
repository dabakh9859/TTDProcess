# Cartographie du frontend — plan de réécriture

Reconstruite par analyse du bundle `dist/assets/index-0nLOWM5B.js` (build du 04/08/2026).
Objectif : savoir quoi réécrire, dans quel ordre, et avec quel contrat backend —
avant d'écrire la première ligne.

Stack : React + Tailwind CSS v4.2.2 + ECharts + zustand + lucide + xlsx, bundlé par Vite/esbuild.

## Arbre des composants racine

```
main.tsx
└── <StrictMode>
    └── ?view=detached ? DetachedChartWindow : App
                                                └── AppLayout
                                                    ├── ThemeProvider   (injecte les tokens CSS)
                                                    ├── Sidebar         (5 sections, 15 onglets)
                                                    └── <div>
                                                        ├── TopBar
                                                        ├── 15 × <main>  ← toutes montées, masquées en CSS
                                                        └── StatusBar
```

Deux points d'architecture à conserver :

1. **Les 15 vues sont montées en permanence.** Le changement d'onglet ne fait que basculer
   `display: block | none`. C'est ce qui permet à chaque vue de garder son état local sans
   store global — ne remplace pas ça par un routeur sans mesurer l'impact.
2. **Fenêtres détachées.** Un graphe peut être ouvert dans une fenêtre séparée :
   `new WebviewWindow('detached-<id>', { url: 'index.html?view=detached&key=…', width: 920, height: 620 })`.
   Les données transitent par `localStorage` sous la clé `ttd-detached-<id>`. La capability
   Tauri `detached-*` est déjà configurée pour ça.

## État global

`recovered/store.ts` — le store zustand reconstruit à l'identique (clé `ttd-app-store`,
`version: 1`). 16 champs, dont `dataVersion` qui sert de signal de rafraîchissement :
toute mutation backend l'incrémente, les vues l'utilisent comme dépendance d'effet.

Seuls 5 champs sont persistés : `currentTab`, `theme`, `language`, `tableTargetDataset`,
`sidebarCollapsed`.

## Les 15 vues

`i18n` = clés de traduction du namespace de la vue. `cmds` = commandes backend appelées.

| Section | Onglet | i18n | cmds | Graphes | Poids |
|---|---|---:|---:|---:|---|
| DONNÉES | `importation` | 49 | 5 | 0 | 23 Ko |
| DONNÉES | `envdata` | 62 | 7 | 0 | 22 Ko |
| DONNÉES | `tableau` | 48 | 2 | 1 | ⚠️ non mesurable |
| ANALYSE | `calculs` | 194 | 13 | 2 | 59 Ko |
| ANALYSE | `calculsAdvanced` | 23 | 2 | 0 | 10 Ko |
| ANALYSE | `agregation` | 68 | 6 | 0 | 27 Ko |
| NETTOYAGE | `aiTraining` | 0 ⚠️ | 13 | 7 | 46 Ko |
| NETTOYAGE | `detection` | 371 | 22 | 9 | 92 Ko |
| NETTOYAGE | `gapfilling` | 161 | 26 | 3 | 90 Ko |
| RÉSULTATS | `scenarios` | 22 | 4 | 3 | 33 Ko |
| RÉSULTATS | `visualisation` | 135 | 9 | 1 | 60 Ko |
| GESTION | `export` | 52 | 5 | 0 | 11 Ko |
| GESTION | `journal` | 16 | 0 | 0 | 6 Ko |
| GESTION | `explications` | 315 | 0 | 0 | 24 Ko |
| GESTION | `parametres` | 27 | 1 | 0 | 7 Ko |

⚠️ **`tableau`** : la région du bundle qui suit ce composant contient aussi ECharts,
le poids n'est donc pas exploitable. Les autres chiffres (i18n, cmds) restent justes.

⚠️ **`aiTraining`** : seule vue sans clés i18n — ses textes français sont écrits en dur
dans le JSX. À internationaliser au passage, ou à laisser tel quel, mais sache que c'est
une exception voulue ou un oubli d'origine.

## Contrat backend par vue

```
importation      get_app_status, get_column_stats, get_sheet_names, load_data, preview_file
envdata          clear_env_data, get_env_info, get_sheet_names, get_table_page,
                 load_env_data_multi, load_env_data_sliced, preview_file
tableau          get_table_page, update_cell
calculs          run_pipeline, recompute_from_stage, set_sap_flow_params, set_ttdplus_params,
                 run_ttdplus_pipeline, list_diurnal_regression_diagnostics,
                 get_diurnal_regression_diagnostic, load_env_data, get_env_info,
                 clear_env_data, get_cleaning_info, get_sheet_names, get_table_page
calculsAdvanced  compute_advanced_chain, list_advanced_sensors
agregation       aggregate_data, list_aggregation_sources, list_aggregations,
                 load_aggregation, delete_aggregation, rename_aggregation
aiTraining       ai_health, ai_train, ai_predict, ai_model_save, ai_model_load,
                 ai_list_models, ai_inspect_files, cleaning_list_columns,
                 cleaning_ml_status, load_env_data, get_env_info, get_sheet_names,
                 get_table_page
detection        cleaning_detect_only, cleaning_run_classical, cleaning_train_ml,
                 cleaning_detect_ml, cleaning_apply_ml, cleaning_detect_saits,
                 cleaning_apply_cells, cleaning_mark_manual_nan, cleaning_reset_ml,
                 cleaning_reset_to_raw, cleaning_lock_permanent, cleaning_unlock_permanent,
                 cleaning_list_columns, cleaning_ml_status, get_cleaning_info,
                 ai_list_models, ai_model_load, load_env_data, get_env_info,
                 clear_env_data, get_sheet_names, get_table_page
gapfilling       gap_filling_run_classical, cleaning_complete_saits,
                 cleaning_commit_completion, cleaning_discard_completion,
                 cleaning_time_gaps, cleaning_reindex_time, cleaning_train_ml,
                 cleaning_apply_ml, cleaning_reset_ml, cleaning_reset_to_raw,
                 cleaning_list_columns, cleaning_ml_status, get_cleaning_info,
                 get_cleaning_pre_rows, ai_train, ai_list_models, list_scenarios,
                 list_scenario_datasets, viz_load_file, viz_unload_file, load_env_data,
                 get_env_info, clear_env_data, preview_file, get_sheet_names, get_table_page
scenarios        save_scenario, load_scenario, list_scenarios, delete_scenario
visualisation    viz_list_sources, viz_load_file, viz_unload_file, viz_overlay_load,
                 viz_overlay_remove, viz_overlay_list, list_scenarios, get_sheet_names,
                 get_table_page
export           export_data, export_data_multi, export_aggregation, get_datasets_info,
                 list_aggregations
journal          — (lit `logs` dans le store, pas d'appel backend)
explications     — (documentation statique)
parametres       set_sap_flow_params
```

## Commandes ajoutées après la récupération

Le backend expose aujourd'hui **93** commandes (88 au moment de la récupération).
Les 5 nouvelles, ajoutées en septembre 2026 :

| Commande | Fichier Rust | Appelée par `dist/` | Vue |
|---|---|---|---|
| `delete_model` | `commands/ml.rs` | oui | suppression d'un modèle (`kind === "ml"`) |
| `ai_model_delete` | `commands/ai.rs` | oui | suppression d'un modèle SAITS |
| `export_data_multi_files` | `commands/export.rs` | oui | `export` — un fichier par jeu de données dans un dossier |
| `cleaning_origin_flags` | `commands/cleaning_v2.rs` | non | — |
| `export_origin_flags` | `commands/cleaning_v2.rs` | non | — |

⚠️ Les trois premières sont appelées par le bundle `dist/assets/index-0nLOWM5B.js`,
qui a été **patché à la main** le 22/09/2026 (il n'est plus identique à
`recovered/dist-original/`). Ces appels n'existent dans aucune source : ils sont à
réécrire dans les vues React correspondantes. Pour les retrouver dans le bundle :
chercher le nom de la commande.

## Commandes backend jamais appelées

17 des 88 commandes d'origine n'apparaissent nulle part dans le bundle :

```
train_model  predict  list_models  save_model  load_model        ← ancien moteur ML natif
detect_outliers  fill_gaps  validate_detection                   ← cleaning v1
get_logs  clear_logs                                             ← journal passé côté store
clear_session  preview_env_conditions  env_column_quick_stats
ai_list_env_columns  cleaning_apply_saits  cleaning_apply_saits_clean
cleaning_restore_cells
```

Ce sont soit des restes de versions précédentes, soit des commandes prévues et jamais
branchées. Ne les recâble pas par réflexe : vérifie d'abord dans le Rust si elles font
encore ce que leur nom suggère.

## Ressources prêtes à réutiliser

Dans `recovered/` — extraites du bundle, directement réinjectables :

| Fichier | Contenu |
|---|---|
| `i18n-fr.json` / `i18n-en.json` | **1 582 clés bilingues**, JSON valide. Toute l'UI est traduite ; la fonction d'origine est `t(key, params)` avec substitution `{param}` |
| `theme.css` | **412 lignes**, palettes dark + light complètes, **avec les commentaires d'origine** (les template literals échappent à la minification) |
| `store.ts` | Le store zustand reconstruit |

Le thème s'applique via `document.documentElement.setAttribute('data-theme', theme)` et
un `<style>` injecté. Les tokens sont sémantiques : `--bg-1..5`, `--text-1..5`,
`--border-1..3`. Le JS lit ces variables au runtime pour styler les tooltips ECharts —
garde les mêmes noms.

## Ordre de réécriture suggéré

1. **Socle** — projet Vite + Tailwind v4, `store.ts`, `theme.css`, module i18n, wrapper `invoke`.
2. **Layout** — Sidebar (5 sections), TopBar, StatusBar, le conteneur à 15 `<main>`.
3. **`importation` → `tableau` → `calculs`** — le chemin critique : sans lui rien d'autre
   n'a de données à afficher. 20 commandes à elles trois.
4. **`export`, `journal`, `parametres`** — petits, peu de commandes, bons pour valider le socle.
5. **`detection` et `gapfilling`** — les deux plus grosses vues (48 commandes, 532 clés i18n,
   12 graphes à elles deux). À garder pour la fin.
6. **`explications`** — 315 clés i18n mais zéro logique : de la documentation statique,
   récupérable presque mécaniquement depuis les fichiers i18n.

## Limites de cette analyse

Le découpage par vue repose sur l'ordre des définitions dans le bundle. C'est fiable pour
les commandes et les clés i18n (recherche par chaîne exacte), mais les poids en Ko sont
approximatifs : un composant partagé entre deux vues est compté dans une seule région, et
la région de `tableau` absorbe ECharts.

Les noms de composants (`tY`, `Gme`, `t0e`…) sont les identifiants minifiés — ils ne
servent qu'à retrouver le code dans le bundle, pas à nommer tes futurs fichiers.
