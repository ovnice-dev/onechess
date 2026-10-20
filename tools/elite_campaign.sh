#!/usr/bin/env bash
# Campagne Elite : conversion PGN -> CSV puis etiquetage Stockfish, mois par mois.
#
# Deux chaines tournent en meme temps pour que le processeur ne reste jamais a
# moitie vide : la conversion (python-chess, lent, un coeur par mois) avance en
# fond pendant que l'etiquetage (Stockfish, tous les autres coeurs) traite les
# mois deja convertis. Un mois converti a son CSV ; un mois etiquete a son
# marqueur .fait. Une relance reprend exactement la ou elle s'etait arretee.
#
# Chaque mois est etiquete dans son propre repertoire, data/nnue_elite/<mois>/.
# Un repertoire de mois SANS marqueur .fait est un mois interrompu : il est
# efface puis refait. Sans cela, une reprise dupliquerait les positions deja
# etiquetees de ce mois : label.py ne peut pas dire ou il s'est arrete, ses
# travailleurs lisent par lots en parallele. On perd au plus le mois en cours.
#
# Cible : TARGET=50000000 arrete l'etiquetage des que le lot atteint 50 M de
# positions. On procede par etapes : 50 M, entrainement, verification que la
# courbe monte encore, puis seulement la suite.
#
# Entrainement sur ce lot : --data "data/nnue_elite/*/*.bin"
#
# ARRET PROPRE : `touch data/STOP_LABEL`. label.py cesse de distribuer des
# lignes, termine celles deja distribuees (une minute environ), note dans
# <mois>/REPRISE la ligne exacte ou il s'est arrete, et la campagne s'arrete en
# emportant la conversion. La relance reprend a cette ligne : rien n'est perdu
# ni double. Tuer les processus a la main fait perdre tout le mois en cours.
#
#   TARGET=50000000 WORKERS=9 bash tools/elite_campaign.sh
set -u
cd /c/Onechess

ZIPS=data/lichess_elite
CSV=data/elite_csv
OUT=data/nnue_elite
WORKERS=${WORKERS:-9}
CONV_WORKERS=${CONV_WORKERS:-2}
NODES=${NODES:-1000}
HASH=${HASH:-16}
TARGET=${TARGET:-}
LOG=data/elite_campaign.log
CONV_LOG=data/elite_conversion.log

mkdir -p "$CSV" "$OUT"
STOP=data/STOP_LABEL
rm -f "$STOP"   # une demande d'arret restee d'une fois precedente
total() { find "$OUT" -name '*.bin' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print int(s/32)}'; }

echo "### campagne Elite lancee $(date '+%Y-%m-%d %H:%M')   cible ${TARGET:-aucune}   deja : $(total) positions" | tee -a "$LOG"

# 1. Conversion en fond, tous les mois ; pgn_to_csv saute ceux deja faits.
python -u tools/pgn_to_csv.py --pgn "$ZIPS/lichess_elite_*.zip" --out "$CSV" \
    --workers "$CONV_WORKERS" >> "$CONV_LOG" 2>&1 &
CONV_PID=$!

arret_propre() {
    echo "$(date +%H:%M) ARRET PROPRE $1 -- reprise exacte au prochain lancement ; lot : $(total) positions" | tee -a "$LOG"
    kill "$CONV_PID" 2>/dev/null
    rm -f "$STOP"
    exit 0
}

# 2. Etiquetage des mois convertis, du plus recent au plus ancien.
while true; do
    next=""
    for c in $(ls "$CSV"/lichess_elite_*.csv 2>/dev/null | sort -r); do
        [ -f "${c%.csv}.fait" ] || { next="$c"; break; }
    done

    if [ -z "$next" ]; then
        # Plus rien a etiqueter : la conversion est-elle finie ?
        if grep -q "^termine" "$CONV_LOG" 2>/dev/null; then break; fi
        [ -f "$STOP" ] && arret_propre "en attente de conversion"
        sleep 60
        continue
    fi

    base=$(basename "$next" .csv)
    dir="$OUT/$base"
    skip=0
    if [ -f "$dir/REPRISE" ]; then
        # Arret propre la derniere fois : reprise a la ligne exacte
        skip=$(cat "$dir/REPRISE")
        echo "$(date +%H:%M) $base : reprise a la ligne $skip" | tee -a "$LOG"
    elif [ -d "$dir" ]; then
        echo "$(date +%H:%M) $base interrompu sans arret propre : $(find "$dir" -name '*.bin' | wc -l) fichiers effaces, mois refait" | tee -a "$LOG"
        rm -rf "$dir"
    fi
    mkdir -p "$dir"

    echo "$(date +%H:%M) etiquetage $base ($(( $(wc -l < "$next") - 1 )) positions candidates)" | tee -a "$LOG"
    python -u tools/label.py --csv "$next" --out "$dir" --workers "$WORKERS" \
        --nodes "$NODES" --hash "$HASH" --quiet-only 1 --skip "$skip" --stop-file "$STOP" >> "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 3 ]; then
        arret_propre "pendant $base"
    elif [ $rc -ne 0 ]; then
        echo "$(date +%H:%M) ECHEC etiquetage $base (code $rc)" | tee -a "$LOG"
        exit 1
    fi
    rm -f "$dir/REPRISE"
    touch "${next%.csv}.fait"

    n=$(total)
    echo "$(date +%H:%M) $base fait -- lot Elite : $n positions" | tee -a "$LOG"
    if [ -n "$TARGET" ] && [ "$n" -ge "$TARGET" ]; then
        echo "$(date +%H:%M) cible $TARGET atteinte" | tee -a "$LOG"
        break
    fi
done

echo "### campagne Elite terminee $(date '+%Y-%m-%d %H:%M') -- $(total) positions" | tee -a "$LOG"
