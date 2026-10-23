#!/usr/bin/env bash
# Panneau de suivi en direct de tout ce qui tourne (PC, Kaggle, GitHub).
#   bash tools/panneau.sh            (rafraichi toutes les 15 s ; Ctrl-C pour quitter)
# Kaggle et GitHub sont interroges en arriere-plan toutes les 3 min (cache).
cd /c/Onechess
K="$HOME/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0/LocalCache/local-packages/Python313/Scripts/kaggle.exe"
G="/c/Program Files/GitHub CLI/gh.exe"
C=data/panneau_cache; mkdir -p "$C/klog"
B=$'\e[1m'; V=$'\e[32m'; J=$'\e[33m'; R=$'\e[31m'; Z=$'\e[0m'; Gr=$'\e[90m'

nuage() {   # Kaggle + GitHub -> fichiers du cache
  {
    for d in _kaggle/onechess-*/; do
      k=$(basename "$d"); [ -f "$d/kernel-metadata.json" ] || continue
      if [ -f "$C/klog/$k.fini" ]; then cat "$C/klog/$k.fini"; continue; fi
      st=$("$K" kernels status "ovnice/$k" 2>&1 | grep -oE 'KernelWorkerStatus.[A-Z_]*' | sed 's/KernelWorkerStatus.//')
      last=""
      if [ "$st" != RUNNING ] && [ "$st" != QUEUED ]; then
        last=$("$K" kernels logs "ovnice/$k" 2>/dev/null | grep -oE '"data":"[^"]*(parties|epoque|verdict|RESULTAT|ECHEC|SIGNATURE|rror|export)[^"]*' | tail -1 | sed 's/"data":"//;s/\\n$//')
      fi
      ligne="$k|$st|$last"
      case "$st" in COMPLETE|ERROR|CANCEL*) echo "$ligne" > "$C/klog/$k.fini";; esac
      echo "$ligne"
    done
  } > "$C/kaggle.tmp" 2>/dev/null && mv "$C/kaggle.tmp" "$C/kaggle.txt"
  { for r in onechess onechess-prive; do
      "$G" run list -R ovnice-dev/$r --limit 6 --json databaseId,displayTitle,status,conclusion,createdAt,updatedAt \
         -q '.[] | "\(.createdAt)|\(.databaseId)|\(.displayTitle)|\(.status)|\(.conclusion)|\(.createdAt)|\(.updatedAt)"'
    done | sort -r | cut -d'|' -f2- | head -8; } > "$C/gh.tmp" 2>/dev/null && mv "$C/gh.tmp" "$C/gh.txt"
  date +%s > "$C/nuage.date"
}

barre() {  # barre n total largeur
  local n=$1 t=$2 l=${3:-30} p
  p=$(( t>0 ? n*l/t : 0 )); [ $p -gt $l ] && p=$l; printf "["; printf "%*s" "$p" "" | tr " " "#"; printf "%*s" $((l-p)) "" | tr " " "."; printf "]"
}

entrainements() {
  echo "${B}== ENTRAINEMENTS (GPU du PC) ==${Z}"
  local vu=0
  for f in $(ls -t data/train_*.log 2>/dev/null | head -3); do
    local age=$(( $(date +%s) - $(stat -c %Y "$f") ))
    [ $age -gt 21600 ] && continue
    vu=1
    local der=$(grep -E 'epoque' "$f" | tail -1)
    local fin=$(grep -E 'export|RESULTAT' "$f" | tail -1)
    local n=$(echo "$der" | sed -E 's/.*epoque +([0-9]+)\/([0-9]+).*/\1/')
    local t=$(echo "$der" | sed -E 's/.*epoque +([0-9]+)\/([0-9]+).*/\2/')
    local err=$(echo "$der" | grep -oE 'erreur +[0-9.]+ cp' | sed 's/  */ /g')
    local meil=$(grep -oE 'erreur +[0-9.]+' "$f" | awk '{print $2}' | sort -n | head -1)
    # duree d'une epoque = mediane des 5 derniers ecarts (ignore les pauses de veille)
    local dt=$(grep -oE '\(([0-9]+) s\)' "$f" | tr -dc '0-9\n' | awk 'NR>1{print $1-p}{p=$1}' | tail -5 | sort -n | awk '{a[NR]=$1}END{print (NR? a[int((NR+1)/2)] : 0)}')
    printf "  %-24s " "$(basename "$f" .log)"
    if [ -n "$fin" ]; then echo "${V}TERMINE${Z}  $fin"; continue; fi
    [ -z "$n" ] && { echo "demarrage... $(tail -1 "$f")"; continue; }
    printf "epoque %2s/%s %s  %s  (meilleure %s cp)" "$n" "$t" "$(barre "$n" "$t" 25)" "$err" "$meil"
    if [ "${dt:-0}" -gt 0 ]; then
      local reste=$(( (t-n)*dt - age )); [ $reste -lt 0 ] && reste=0
      printf "\n  %-24s %ss/epoque, fin vers ${B}%s${Z}" "" "$dt" "$(date -d "@$(( $(date +%s) + reste ))" +%H:%M)"
      [ $age -gt $(( dt*3 )) ] && printf "  ${R}<- aucune epoque depuis %d min (arrete ? veille ?)${Z}" $((age/60))
    fi
    echo
  done
  [ $vu = 0 ] && echo "  ${Gr}aucun entrainement actif${Z}"
}

donnees() {
  echo "${B}== DONNEES (PC) ==${Z}"
  local pos=$(find data/nnue_elite -name '*.bin' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {printf "%d", s/32}')
  printf "  lot etiquete : ${B}%s${Z} positions   %s cible 250 M\n" "$(echo "$pos" | sed -E ':a;s/([0-9])([0-9]{3})($| )/\1 \2\3/;ta')" "$(barre "$((pos/1000000))" 250 25)"
  local a=$(( $(date +%s) - $(stat -c %Y data/elite_campaign.log) ))
  echo "  campagne   : $(grep -vE '^###' data/elite_campaign.log | grep -E 'fait|ARRET|etiquet|positions' | tail -1 | cut -c1-110)"
  echo "  conversion : $(tail -1 data/elite_conversion.log | cut -c1-110)"
  echo "  ${Gr}(journaux ecrits a la fin de chaque fichier ; derniere ecriture il y a $((a/60)) min)${Z}"
  local cv=$(tasklist 2>/dev/null | grep -ciE 'python|pgn-extract|stockfish')
  echo "  processus python/stockfish actifs : $cv"
}

matchs_locaux() {
  local l=""
  for f in $(ls -t data/*.log 2>/dev/null | head -15); do
    case "$f" in *train_*|*elite_*|*build_*) continue;; esac
    [ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt 1800 ] && continue
    local x=$(grep -E 'Elo|parties|LLR|verdict' "$f" | tail -1 | cut -c1-110)
    [ -n "$x" ] && l+="  $(basename "$f" .log) : $x"$'\n'
  done
  [ -n "$l" ] && { echo "${B}== MATCHS / SPRT SUR LE PC ==${Z}"; printf "%s" "$l"; }
}

kaggle() {
  echo "${B}== KAGGLE ==${Z}"
  [ -f "$C/kaggle.txt" ] || { echo "  ${Gr}premiere interrogation en cours...${Z}"; return; }
  # en cours d'abord, puis les 5 derniers termines
  { grep -E '\|(RUNNING|QUEUED)\|' "$C/kaggle.txt"; grep -vE '\|(RUNNING|QUEUED)\|' "$C/kaggle.txt" | tail -5; } |
  while IFS='|' read -r k st last; do
    case "$st" in RUNNING) c=$J; last="(journal lisible a la fin)";; QUEUED) c=$J;; COMPLETE) c=$V;; *) c=$R;; esac
    printf "  %-28s ${c}%-10s${Z} %s\n" "$k" "$st" "$(echo "$last" | cut -c1-90)"
  done
}

github() {
  echo "${B}== GITHUB ACTIONS ==${Z}"
  [ -f "$C/gh.txt" ] || { echo "  ${Gr}premiere interrogation en cours...${Z}"; return; }
  while IFS='|' read -r id titre st concl cree maj; do
    local nom=$(grep -E "^$id " ci/noms_runs.txt 2>/dev/null | cut -d' ' -f2-); [ -z "$nom" ] && nom=$titre
    local deb=$(date -d "$cree" +%s) dur
    if [ "$st" = completed ]; then
      dur=$(( $(date -d "$maj" +%s) - deb ))
      local v=$(grep -hE "verdict|^Elo" ci_resultats/$id/*/*.log 2>/dev/null | tail -1 | cut -c1-70)
      printf "  %-12s %-16s ${V}%-9s${Z} %3d min  %s\n" "$id" "$(echo "$nom" | cut -c1-16)" "$concl" $((dur/60)) "$v"
    else
      dur=$(( $(date +%s) - deb ))
      printf "  %-12s %-16s ${J}%-9s${Z} %s %d/360 min\n" "$id" "$(echo "$nom" | cut -c1-16)" "$st" "$(barre $((dur/60)) 360 20)" $((dur/60))
    fi
  done < <(head -8 "$C/gh.txt")
  # SPRT en lot : bilan cumule de chaque test (ci/sprt_lot.sh)
  for f in $(ls -t ci_resultats/lot/*.log 2>/dev/null | head -6); do
    [ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt 86400 ] && continue
    local e=$(grep -E "Elo|FIN|vague .* lancee" "$f" | tail -1 | sed 's/^ *//' | cut -c1-95)
    printf "  ${B}lot %-14s${Z} %s\n" "$(basename "$f" .log)" "$e"
  done
}

trap 'tput cnorm 2>/dev/null; echo; exit 0' INT TERM
while true; do
  der=$(cat "$C/nuage.date" 2>/dev/null || echo 0)
  if [ $(( $(date +%s) - der )) -gt 180 ] && { [ ! -f "$C/verrou" ] || [ $(( $(date +%s) - $(stat -c %Y "$C/verrou") )) -gt 600 ]; }; then
    date +%s > "$C/nuage.date"; ( touch "$C/verrou"; nuage; rm -f "$C/verrou" ) >/dev/null 2>&1 &
  fi
  sortie=$( {
    echo "${B}ONECHESS — panneau en direct${Z}   $(date '+%a %d %H:%M:%S')   ${Gr}(PC : 15 s ; Kaggle/GitHub : 3 min, dernier il y a $(( ($(date +%s) - der)/60 )) min)${Z}"
    echo
    entrainements; echo
    matchs_locaux
    donnees; echo
    kaggle; echo
    github; echo
    echo "${Gr}Ctrl-C pour quitter.  Lot de reference installe : bench 12 = $(cat ci/signature.txt 2>/dev/null)${Z}"
  } 2>/dev/null )
  clear; printf "%s\n" "$sortie"
  sleep 15
done
