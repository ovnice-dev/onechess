#pragma once

#include <string>

#include "types.h"

namespace one {

class Position;

// ---------------------------------------------------------------------------
// Tables de finale Syzygy, via Fathom (MIT, Ronald de Man / basil00 / Jon Dart,
// engine/fathom/). Ce fichier est la seule frontiere entre le moteur et ce code
// tiers : rien d'autre n'inclut tbprobe.h.
//
// Ce que les tables apportent : au-dela d'un certain nombre de pieces, elles
// donnent le resultat EXACT -- gagne, nul, perdu -- sans recherche. La
// recherche n'a plus a explorer ces finales, elle lit la reponse. Le gain
// vient moins des positions ou l'on y arrive que de toutes celles ou la
// recherche voit une finale au loin et sait immediatement quoi en penser.
//
// Deux tables distinctes :
//   WDL (gagne/nul/perdu) : sondee PENDANT la recherche, c'est elle qui fait
//                           la force.
//   DTZ (distance au zero) : sondee A LA RACINE seulement, pour choisir un
//                           coup qui gagne VRAIMENT, c'est-a-dire en
//                           respectant la regle des 50 coups.
// ---------------------------------------------------------------------------

namespace Syzygy {

// Chemin vide, "<empty>" ou dossier absent : les sondes sont simplement
// desactivees. Aucune ressource externe ne doit etre obligatoire -- un moteur
// qui exige ses tables est refuse par les testeurs de classement.
void init(const std::string& path);
void free();

// 0 quand aucune table n'est chargee.
int  max_pieces();
const std::string& path();

// Sonde WDL. Rend false si la position n'est pas dans les tables ou si la
// sonde echoue. `v` recoit un score deja exprime en distance depuis la racine.
bool probe_wdl(const Position& pos, int ply, bool use50MoveRule, Value* v);

// Sonde DTZ a la racine. Rend le meilleur coup et le resultat, ou MOVE_NONE.
Move probe_root(const Position& pos, bool use50MoveRule, int* wdl);

}  // namespace Syzygy
}  // namespace one
