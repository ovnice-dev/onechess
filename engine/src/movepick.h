/*
  Onechess, a UCI chess engine derived from Stockfish.
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)
  Copyright (C) 2026 ovnice (modifications)

  Onechess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Onechess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Ordre dans lequel la recherche essaie les coups (reecrit pour Onechess le
// 15-11-2026). Les coups sont produits par tranches, au fur et a mesure : une
// coupure sur le coup de la table ou une bonne prise evite de generer le reste.

#pragma once

#include "history.h"
#include "movegen.h"
#include "position.h"
#include "tune.h"
#include "types.h"

namespace one {

// Cases attaquees par le camp qui n'a pas le trait (0 tant que ThreatHist est
// eteint : l'historique principal garde alors son index simple).
inline Bitboard threat_map(const Position& pos) {
    if (!Tune::ThreatHist)
        return 0;
    const Color eux = ~pos.side_to_move();
    return pos.attacks_by<PAWN>(eux) | pos.attacks_by<KNIGHT>(eux) | pos.attacks_by<BISHOP>(eux)
         | pos.attacks_by<ROOK>(eux) | pos.attacks_by<QUEEN>(eux) | pos.attacks_by<KING>(eux);
}

// Cases attaquees par le camp qui n'a pas le trait, toujours calculees
// (NmpTension, RfpTension), meme quand ThreatHist est eteint.
inline Bitboard threat_map_complete(const Position& pos) {
    const Color eux = ~pos.side_to_move();
    return pos.attacks_by<PAWN>(eux) | pos.attacks_by<KNIGHT>(eux) | pos.attacks_by<BISHOP>(eux)
         | pos.attacks_by<ROOK>(eux) | pos.attacks_by<QUEEN>(eux) | pos.attacks_by<KING>(eux);
}

// Place d'un coup dans l'historique principal : depart et arrivee, et pour
// chacune des deux cases, menacee ou non (idee de l'historique « 4D » de
// plusieurs moteurs libres).
inline int hist_index(Move m, Bitboard menaces) {
    const int depart  = (menaces & m.from_sq()) ? 2 : 0;
    const int arrivee = (menaces & m.to_sq()) ? 1 : 0;
    return m.from_to() + 4096 * (depart + arrivee);
}

class MovePicker {
   public:
    // Recherche principale et quiescence (depth <= 0)
    MovePicker(const Position&              pos,
               Move                         coupTable,
               Depth                        depth,
               const ButterflyHistory*      principal,
               const CapturePieceToHistory* prises,
               const PieceToHistory**       suites,
               const PawnHistory*           pions,
               const Move*                  meurtriers,
               Move                         reponse,
               const LowPlyHistory*         racine,
               int                          ply);

    // ProbCut : seulement les prises dont l'echange atteint « seuil »
    MovePicker(const Position& pos, Move coupTable, int seuil, const CapturePieceToHistory* prises);

    MovePicker(const MovePicker&)            = delete;
    MovePicker& operator=(const MovePicker&) = delete;

    // Coup suivant, ou Move::none() quand tout a ete rendu. Avec
    // sansTranquilles, les coups tranquilles ne sont plus proposes.
    Move next_move(bool sansTranquilles = false);

   private:
    enum Etape : int;

    template<GenType T>
    void noter();  // remplit la note de tri de la tranche [cur, fin)
    template<typename Test>
    Move premier(Test accepte);  // premier coup restant (hors coup de table) qui passe le test
    bool est_refutation(Move m) const { return m == refut[0] || m == refut[1] || m == refut[2]; }

    const Position&              pos;
    const ButterflyHistory*      hPrincipal = nullptr;
    const CapturePieceToHistory* hPrises    = nullptr;
    const PieceToHistory**       hSuites    = nullptr;
    const PawnHistory*           hPions     = nullptr;
    const LowPlyHistory*         hBas       = nullptr;   // (26-11) historique des premiers plis (option LowPly)
    int                          plyBas     = 0;

    Move  coupTable;
    Move  refut[3];         // deux coups meurtriers, puis la reponse au coup precedent
    int   refutSuivante = 0;
    int   etape;
    int   seuil = 0;        // ProbCut
    Depth prof  = 0;

    ExtMove* cur;           // prochain coup a examiner
    ExtMove* fin;           // fin de la tranche en cours
    ExtMove* finMauvaises;  // les mauvaises prises s'empilent en tete du tampon
    ExtMove* debTranq = nullptr;
    ExtMove* finTranq = nullptr;
    ExtMove  tampon[MAX_MOVES];
};

}  // namespace one
