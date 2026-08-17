# TTDProcess v2.5.0 — notes de récupération

Récupération du 16/08/2026. Le projet avait été réduit à `src-tauri/` seul (config,
frontend et fichiers racine perdus). Ce document décrit ce qui a été restauré, comment,
et ce qui reste incertain.

## Comment la récupération a été possible

Tauri embarque le frontend compilé, compressé en Brotli, dans `src-tauri/target/`.
Ces artefacts avaient survécu. Comme ni Node, ni Python, ni Rust n'étaient installés
sur la machine, la décompression Brotli a été faite en servant les fichiers à Chrome
headless avec un en-tête `Content-Encoding: br`.

Les fichiers `target/**/deps/*.d` (dep-info de Cargo) et `target/**/.fingerprint/**`
ont fourni la liste exacte des dépendances, leurs versions et leurs features.

## État des fichiers

### Intacts (jamais perdus)

- `src-tauri/src/` — 35 fichiers Rust, 19 134 lignes : l'intégralité du backend
  (88 commandes Tauri exposées)
- `src-tauri/sidecars/ttd-ai/` — 11 fichiers Python, 1 656 lignes : le service
  d'imputation SAITS

### Restaurés à l'identique depuis `target/`

- `dist/` — le frontend compilé : `index.html`, `assets/index-0nLOWM5B.js` (2,1 Mo),
  `assets/index-B2V9uPAU.css` (Tailwind v4.2.2), `favicon.svg`, `icons.svg`.
  Build du 04/08/2026 — le plus récent disponible.
- `src-tauri/sidecars/python-embed/` — le dossier était tronqué (24 entrées dans
  `site-packages` au lieu de 143 ; `python.exe` et les DLL manquaient). Les
  19 857 fichiers manquants ont été recopiés depuis `target/debug/sidecars/`.
  Vérifié : Python 3.11.9, numpy 2.4.5, torch 2.12.0+cu126 s'importent correctement.
- `src-tauri/capabilities/default.json` — copie exacte du `capabilities.json` généré.
- `src-tauri/icons/` — l'icône 32×32 RGBA embarquée a été extraite ; c'est l'icône
  Tauri par défaut (le projet n'avait pas de logo custom). Les tailles supérieures
  sont des agrandissements. Pour les régénérer proprement :
  `npx @tauri-apps/cli icon chemin/vers/logo.png`

### Reconstruits (déduits des artefacts — à relire)

- `src-tauri/Cargo.toml` — versions et features extraites des fingerprints Cargo,
  donc fidèles à ce qui était réellement compilé.
- `src-tauri/build.rs` — le manifeste Windows est repris mot pour mot du `resource.rc`
  compilé. Note : il demande `requireAdministrator` (UAC à chaque lancement).
  C'était bien le cas dans le build d'origine ; à changer en `asInvoker` si ce
  n'était pas voulu.
- `src-tauri/tauri.conf.json` — voir les incertitudes ci-dessous.
- `src-tauri/RECOVERED-crate-versions.txt` — les 392 crates avec leur version exacte
  telles que compilées le 04/08/2026.

### Perdu, non récupérable

- Les **sources** du frontend (`src/*.tsx`, `package.json`, `vite.config.ts`,
  `tsconfig.json`). Le bundle dans `dist/` est minifié et sans sourcemap.
  Stack identifiée : React + Tailwind CSS v4 + ECharts + d3 + zustand + xlsx + lucide,
  bundlé par Vite/esbuild.
- `Cargo.lock`.

## Incertitudes dans `tauri.conf.json`

Ces valeurs ne sont pas déductibles des artefacts — elles étaient compilées dans le
binaire, qui a disparu. Elles sont fonctionnelles mais ne sont pas forcément celles
d'origine :

| Champ | Valeur mise | Remarque |
|---|---|---|
| `identifier` | `com.ttdprocess.app` | Sans impact sur tes données : le stockage utilise `%APPDATA%\ttdprocess\`, en dur dans le code Rust |
| `app.windows[0]` | 1440×900, min 1024×640 | Taille arbitraire, à ajuster |
| `bundle.targets` | `["msi"]` | Déduit des commentaires dans `ai/sidecar.rs` |
| `[profile.release]` | absent | Défauts Cargo (`opt-level=3`). Le profil d'origine est inconnu |

En revanche les fenêtres détachées sont confirmées : label `detached-*`, 920×620,
URL `index.html?view=detached&key=…` (lu dans le bundle).

## Prérequis pour compiler

Rien n'est installé sur cette machine. Dans l'ordre :

1. **Visual Studio Build Tools** — charge « Desktop development with C++ »
   (MSVC + Windows SDK). Obligatoire, Rust ne peut pas linker sans.
2. **Rust** — https://rustup.rs (toolchain `stable-x86_64-pc-windows-msvc`)
3. **WebView2** — déjà présent sur Windows 11.
4. **Node.js** — seulement si tu veux retravailler le frontend.

## Compiler

`tauri.conf.json` pointe sur `../dist` sans `beforeBuildCommand`, donc le build
n'a besoin ni de Node ni de npm :

```powershell
cd src-tauri
cargo build --release        # binaire seul
cargo install tauri-cli --version "^2"
cargo tauri build            # + installeur MSI
```

Si Cargo résout une version de dépendance incompatible (il n'y a plus de
`Cargo.lock`), épingle-la avec la version d'origine listée dans
`RECOVERED-crate-versions.txt` :

```powershell
cargo update -p <crate> --precise <version>
```

## Le socle frontend

Un projet Vite + React + TypeScript est en place à la racine : layout complet
(sidebar, topbar, barre de statut), store, i18n, thème, wrapper `invoke` typé,
et 15 vues encore à l'état de stub. Voir [FRONTEND-MAP.md](FRONTEND-MAP.md)
pour le plan de réécriture.

```powershell
npm install
npm run dev        # http://localhost:1420
```

Ce socle n'a **jamais été compilé** — Node n'est pas installé sur cette machine.
La cohérence a été vérifiée statiquement (imports, clés i18n, onglets, noms de
commandes : 0 problème), mais attends-toi à quelques ajustements de types au
premier `npm run build`. Deux versions à surveiller dans `package.json`, faute
d'avoir pu les lire dans le bundle : `lucide-react` (les noms d'icônes
`ChartLine`, `CircleCheck`, `TriangleAlert`, `CircleX` sont ceux des versions
récentes) et `@tailwindcss/vite`.

### ⚠️ `npm run build` écrase `dist/`

`dist/` contient aujourd'hui le frontend **récupéré et fonctionnel**. Un build
Vite le remplacerait par le socle, c'est-à-dire une app réduite à 15 pages
vides. Une copie de sauvegarde est dans `recovered/dist-original/`.

C'est aussi pourquoi `tauri.conf.json` ne déclare **pas** de
`beforeBuildCommand` : tant que la réécriture n'est pas finie, `cargo tauri build`
doit empaqueter le `dist/` récupéré, pas en reconstruire un. Quand tes vues
seront prêtes, rebranche Vite :

```json
"build": {
  "frontendDist": "../dist",
  "devUrl": "http://localhost:1420",
  "beforeDevCommand": "npm run dev",
  "beforeBuildCommand": "npm run build"
}
```

### Écart de version

Le frontend affiche `v2.6.0` (sidebar et topbar) alors que le Rust est en
`2.5.0`. C'était déjà le cas dans le build d'origine — le frontend avait pris
de l'avance. À aligner quand tu voudras.

## Sauvegarde

Le projet n'est pas sous contrôle de version. Un `git init` + un premier commit
éviterait une deuxième perte — et `src-tauri/target/` (4,6 Go) devrait être exclu
via `.gitignore`, sauf que ce sont précisément ces artefacts qui ont sauvé le projet
cette fois-ci.
