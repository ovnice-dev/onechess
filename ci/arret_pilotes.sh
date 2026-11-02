#!/usr/bin/env bash
# Arrete le pilote de la file ET ses enfants sprt_lot (sinon les enfants
# survivent et un nouveau pilote lance des vagues en double). Les vagues deja
# lancees sur GitHub continuent ; un nouveau pilote les reprend.
for p in $(ps -ef | grep -E "ci/(file_lot|sprt_lot)\.sh" | grep -v grep | awk '{print $2}'); do kill "$p"; done
sleep 1
ps -ef | grep -E "ci/(file_lot|sprt_lot)\.sh" | grep -v grep || echo "aucun pilote"
