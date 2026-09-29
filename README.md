# TTDProcess

Application de bureau pour le traitement des données de flux de sève mesurées par
la méthode TTD (*Thermal Time Domain*). Elle couvre la chaîne complète : import
des fichiers de centrale d'acquisition, calcul du flux, nettoyage manuel des
séries, comblement des données manquantes par apprentissage profond, et export.

Développée pour le suivi de *Faidherbia albida* sur les stations de Niakhar
(Sénégal).

## Fonctionnalités

- **Import** — fichiers Excel, CSV et DAT ; chargement direct à n'importe quelle
  étape du pipeline (T600, flux de sève, etc.) sans repasser par les données brutes
- **Calculs TTD** — chaîne Tslope → Baseline → ΔT → T600 → T0 → K → flux de sève,
  plus la variante TTD+ et la régression diurne
- **Calculs avancés** — agrégation par groupe de capteurs : Jh → Jhp → Qh → Qd
- **Nettoyage** — sélection manuelle sur graphique en mode bande ou boîte,
  détection assistée, verrouillage des séries validées
- **Comblement** — modèle SAITS (PyTorch / PyPOTS) entraîné sur les capteurs de
  la station, enrichi des variables environnementales et de repères temporels
- **Visualisation** — séries multiples, superposition de scénarios, fenêtres
  détachables
- **Export** — Excel et CSV, par jeu de données ou par agrégation

Interface bilingue français / anglais, thèmes clair et sombre.

## Style de l’interface

Le thème sobre ardoise / bleu est défini dans `public/professional.css`, partagé
par le frontend historique et les sources React. Pour appliquer une modification
du style à l’application complète, exécuter `npm run ui:apply`, puis relancer
`./linux.sh` (ou recompiler Tauri). Cette commande conserve le JavaScript métier
de `dist/`. `./linux.sh originale` réapplique également le thème après restauration.

Attention : `npm run build` reconstruit l’interface React encore incomplète et
remplace le frontend historique. Ne pas l’utiliser pour une simple modification
du thème de l’application complète.

## Architecture

| Couche | Technologie |
|---|---|
| Interface | React 19, Tailwind CSS 4, ECharts, zustand |
| Application | Tauri 2 |
| Calcul | Rust, Polars, linfa |
| Apprentissage | Python 3.11, PyTorch, PyPOTS (processus séparé, JSON-RPC) |

Le moteur d'apprentissage tourne dans un processus Python distinct, piloté par le
cœur Rust via JSON-RPC sur l'entrée et la sortie standard. Ce découplage permet
d'utiliser PyTorch sans l'embarquer dans le binaire.

## Compilation

### Prérequis

- **Rust** (toolchain `stable-x86_64-pc-windows-msvc`)
- **Visual Studio Build Tools** avec la charge de travail C++ et le SDK Windows
- **Node.js 20+**
- **WebView2** (présent d'origine sur Windows 11)

Sous Windows, tout s'installe avec winget :

```powershell
winget install OpenJS.NodeJS.LTS
winget install Rustlang.Rustup
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

### Développer (hot reload)

```powershell
npm install
npm run app:dev       # nouvelle interface React (src/) + Rust
npm run app:legacy    # interface complète historique (dist/) + Rust
```

- **`app:dev`** lance Vite sur `http://localhost:1420` et ouvre l'application
  dessus : toute modification dans `src/` est appliquée à chaud (HMR), sans
  redémarrage. C'est le mode pour la réécriture des vues.
- **`app:legacy`** ignore Vite et sert `dist/` tel quel (le bundle fonctionnel,
  voir « État du projet »). Pas de HMR côté interface — c'est un fichier
  minifié — mais c'est le mode pour tester le backend sur l'application complète.
- Dans les deux cas, une modification d'un fichier `src-tauri/src/**/*.rs`
  recompile et relance l'application automatiquement.

La première compilation Rust prend plusieurs minutes (Polars) ; les suivantes
sont incrémentales.

### Construire

```bash
npm install
npm run build          # frontend
cd src-tauri
cargo build --release  # application
```

Pour produire l'installeur :

```bash
cargo install tauri-cli --version "^2"
cargo tauri build
```

### Runtime Python

Le runtime Python embarqué (~4,6 Go avec PyTorch) n'est pas versionné. Il se
place dans `src-tauri/sidecars/python-embed/` et doit fournir `python.exe` ainsi
que les paquets `torch`, `pypots`, `numpy` et `pandas`.

En développement, deux variables d'environnement permettent de pointer vers un
interpréteur existant :

```
TTD_AI_PYTHON    chemin de l'exécutable Python
TTD_AI_SERVICE   chemin de sidecars/ttd-ai/service.py
```

## Organisation du dépôt

```
src/                      interface React
src-tauri/src/            cœur Rust — calculs, commandes, état
  ├── core/               chaîne de calcul TTD et TTD+
  ├── commands/           commandes exposées à l'interface
  ├── ml/                 modèles natifs (régression, SVM)
  └── ai/                 pont vers le processus Python
src-tauri/sidecars/ttd-ai/  service d'apprentissage SAITS
dist/                     frontend compilé
```

## État du projet

Le cœur Rust et le service Python sont complets et fonctionnels : 93 commandes
exposées, chaîne de calcul entière, entraînement et comblement opérationnels.

Le frontend est **en cours de réécriture**. Le socle est en place — mise en page,
navigation, thème, internationalisation, état global — ainsi que la vue
d'importation. Les autres vues sont des ébauches qui affichent le contrat backend
qu'il leur reste à implémenter. Le dossier `dist/` contient la version compilée
antérieure, pleinement fonctionnelle : c'est elle que l'application embarque tant
que la réécriture n'est pas achevée.

## Licence

Aucune licence n'est encore attribuée. Tous droits réservés.
