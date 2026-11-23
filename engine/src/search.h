/*
  Onechess, a UCI chess engine.
  Copyright (C) 2026 ovnice

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

// Recherche : contraintes de l'interface, coups de la racine (reecrit pour
// Onechess le 15-11-2026).

#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include "history.h"
#include "misc.h"
#include "position.h"
#include "types.h"

namespace one {
namespace Search {

// Ce qu'un noeud sait de ses ancetres : une case par demi-coup depuis la
// racine, plus quelques cases avant elle pour lire (ss - 6) sans test.
struct Stack {
    Move*                     pv;
    PieceToHistory*           suite;        // historique de suite du coup joue a ce niveau
    PieceToCorrectionHistory* suiteCorr;    // et sa table de correction
    int                       ply;
    Move                      coup;         // coup en cours d'essai (ou coup nul)
    Move                      exclu;        // coup exclu (verification singuliere)
    Move                      meurtriers[2];
    Value                     evalStatique; // corrigee, VALUE_NONE en echec
    int                       noteStat;     // historique du coup, lu par le fils
    int                       nbCoups;
    bool                      enEchec;
    bool                      ttPv;
    bool                      ttTrouve;
    int                       coupures;     // coupures beta chez les fils
    int                       reduction;    // reduction LMR subie, lue par le fils
    Bitboard                  menaces;      // pour l'index de l'historique principal
};

// Un coup de la racine, sa variante principale et ses scores
struct RootMove {
    explicit RootMove(Move m) : pv{m} {}
    bool operator==(const Move& m) const { return pv.front() == m; }
    // Meilleur score d'abord ; a egalite, celui de l'iteration precedente
    bool operator<(const RootMove& autre) const {
        if (score != autre.score)
            return score > autre.score;
        return previousScore > autre.previousScore;
    }

    Value             score         = -VALUE_INFINITE;
    Value             previousScore = -VALUE_INFINITE;
    Value             averageScore  = -VALUE_INFINITE;
    std::uint64_t     effort        = 0;  // noeuds passes sous ce coup
    int               selDepth      = 0;
    bool              borneBasse = false, borneHaute = false;  // score d'une fenetre depassee (affichage)
    std::vector<Move> pv;
};
using RootMoves = std::vector<RootMove>;

// Contraintes de « go »
struct LimitsType {
    bool use_time_management() const { return time[WHITE] != 0 || time[BLACK] != 0; }

    std::vector<Move> searchmoves;
    TimePoint         time[COLOR_NB] = {0, 0};
    TimePoint         inc[COLOR_NB]  = {0, 0};
    TimePoint         movetime = 0, startTime = 0;
    int               movestogo = 0, depth = 0, mate = 0, perft = 0, infinite = 0;
    std::uint64_t     nodes      = 0;
    bool              silent     = false;  // le banc n'annonce rien
    bool              ponderMode = false;  // « go ponder »
};

extern LimitsType Limits;
// Score des tables a afficher pour la racine (VALUE_NONE si hors des tables)
extern Value RootTbScore;
// Pendant « go ponder », pas d'arret de soi-meme : une decision d'arret attend
// le « ponderhit ». Active : une recherche lancee par « go » n'a pas rendu son coup.
extern std::atomic<bool> Ponder, StopOnPonderhit, Active;
// Reglages des tables de finales, recopies des options a chaque « go » (22-11 : lire les options
// a chaque noeud coutait 31 % de vitesse des que des tables etaient chargees). limite = 0 : jamais de sonde.
struct ReglagesTb {
    int  limite = 0, profondeur = 1;
    bool regle50 = true;
};
extern ReglagesTb Tb;

void init();  // tables qui dependent des constantes reglables

}  // namespace Search
}  // namespace one
