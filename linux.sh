#!/usr/bin/env bash
#
# Lancement de TTDProcess sous Linux.
#
# Deux variables sont nécessaires ici et ne le sont pas sous Windows :
#
#   WEBKIT_DISABLE_DMABUF_RENDERER  WebKitGTK ne sait pas passer par DMA-BUF
#                                   sur pilote NVIDIA en session Wayland ;
#                                   sans ça l'app meurt au démarrage sur
#                                   « Error 71 dispatching to Wayland display ».
#
#   TTD_AI_PYTHON                   Le python-embed du dépôt ne contient que
#                                   des binaires win_amd64. On pointe sur le
#                                   venv Linux (voir .gitignore pour le
#                                   recréer). Le code lit déjà cette variable
#                                   en priorité — cf. src-tauri/src/ai/sidecar.rs.
#
# Usage :
#   ./linux.sh            lance l'app telle que dist/ est aujourd'hui
#   ./linux.sh release    idem, mais binaire optimisé — beaucoup plus rapide
#                         sur les gros exports et les calculs Polars
#   ./linux.sh originale  remet le bundle v2.6.0 complet dans dist/, puis lance
#                         ⚠ écrase les patchs faits à la main dans dist/ (22/09)
#   ./linux.sh socle      reconstruit dist/ depuis src/ (14 vues sur 15 sont
#                         encore des ViewStub), puis lance
#
# Pour travailler dans un shell interactif : source ./linux.sh env
#
set -e
# BASH_SOURCE et non $0 : le script doit aussi marcher quand il est sourcé,
# où $0 vaut le shell appelant et non le script.
cd "$(dirname "${BASH_SOURCE[0]}")"

# rustup n'ajoute cargo au PATH que dans les shells de connexion.
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

export WEBKIT_DISABLE_DMABUF_RENDERER=1
export TTD_AI_PYTHON="$PWD/.venv-ttd-ai/bin/python"

MODE=""
case "${1:-}" in
  env)
    echo "Variables exportées. TTD_AI_PYTHON=$TTD_AI_PYTHON"
    return 0 2>/dev/null || exit 0
    ;;
  originale)
    rm -rf dist && mkdir dist && cp -r recovered/dist-original/. dist/
    node scripts/apply-interface.mjs
    echo "dist/ ← bundle d'origine v2.6.0 (2,1 Mo)"
    ;;
  socle)
    npx vite build
    echo "dist/ ← socle réécrit"
    ;;
  release)
    MODE="--release"
    ;;
esac

cd src-tauri
exec cargo run $MODE
