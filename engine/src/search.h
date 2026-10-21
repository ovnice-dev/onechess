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

// Une case de la pile de recherche : tout ce qu'un noeud doit savoir de ses
// ancetres. Deux cases sont reservees avant la racine pour que les acces a
// (ss-1) et (ss-2) soient toujours valides.
struct Stack {
    Move*           pv;
    PieceToHistory* continuationHistory;
    PieceToCorrectionHistory* continuationCorrectionHistory;
    int             ply;
    Move            currentMove;
    Move            excludedMove;
    Move            killers[2];
    Value           staticEval;
    int             statScore;
    int             moveCount;
    bool            inCheck;
    bool            ttPv;
    bool            ttHit;
    int             cutoffCnt;
    int             reduction;   // reduction subie, lue par le fils
    Bitboard        threats;     // menaces du noeud, pour l'index de l'historique principal
};

// Un coup a la racine, avec sa variante principale et son score.
struct RootMove {
    explicit RootMove(Move m) : pv(1, m) {}

    bool operator==(const Move& m) const { return pv[0] == m; }

    // Tri decroissant : le meilleur score en tete, l'iteration precedente
    // departageant les egalites.
    bool operator<(const RootMove& m) const {
        return m.score != score ? m.score < score : m.previousScore < previousScore;
    }

    Value             score         = -VALUE_INFINITE;
    Value             previousScore = -VALUE_INFINITE;
    Value             averageScore  = -VALUE_INFINITE;
    std::uint64_t     effort        = 0;  // noeuds depenses sous ce coup
    int               selDepth      = 0;
    std::vector<Move> pv;
};

using RootMoves = std::vector<RootMove>;

// Contraintes envoyees par l'interface UCI.
struct LimitsType {
    LimitsType() {
        time[WHITE] = time[BLACK] = inc[WHITE] = inc[BLACK] = movetime = TimePoint(0);
        movestogo = depth = mate = perft = infinite = 0;
        nodes                                       = 0;
        startTime                                   = 0;
    }

    bool use_time_management() const { return time[WHITE] || time[BLACK]; }

    std::vector<Move> searchmoves;
    TimePoint         time[COLOR_NB], inc[COLOR_NB], movetime, startTime;
    int               movestogo, depth, mate, perft, infinite;
    std::uint64_t     nodes;
    bool              silent = false;   // le banc n'annonce pas de coup
};

extern LimitsType Limits;

void init();

}  // namespace Search
}  // namespace one
