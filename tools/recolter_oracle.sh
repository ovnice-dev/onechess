#!/usr/bin/env bash
# Rapatrie l'autojeu de la machine Oracle (Always Free, 2 coeurs ARM, service
# systemd "autojeu", graines 300000+) dans data/autojeu/ (prefixe or_), puis
# efface les tranches recuperees sur la machine. Seules les tranches finies
# (~/tranches/g*) sont prises ; la tranche en cours reste sur place.
#   bash tools/recolter_oracle.sh
set -e
H=ubuntu@130.110.241.180
K="$HOME/.ssh/oracle_onechess"
D=/c/Onechess/data/autojeu
T=$(mktemp -d)
ssh -i "$K" $H 'systemctl is-active autojeu; tail -1 dernier.log; ls tranches' | tee "$T/etat"
for g in $(grep -E '^g[0-9]+$' "$T/etat"); do
  mkdir -p "$T/$g"
  scp -i "$K" -q "$H:tranches/$g/*.bin" "$T/$g/"
  for f in "$T/$g"/*.bin; do cp "$f" "$D/or_${g}_$(basename "$f")"; done
  ssh -i "$K" $H "rm -rf tranches/$g"
  echo "$g : $(cat "$T/$g"/*.bin | wc -c | awk '{print $1/32}') positions"
done
rm -rf "$T"
