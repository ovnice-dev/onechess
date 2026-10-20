#pragma once

#include "history.h"
#include "movegen.h"
#include "position.h"
#include "tune.h"
#include "types.h"

namespace one {

// Cases attaquees par l'adversaire du camp au trait. Nul tant que ThreatHist
// vaut 0 : l'historique principal reste alors indexe comme avant.
inline Bitboard threat_map(const Position& pos) {
    if (!Tune::ThreatHist)
        return 0;
    const Color them = ~pos.side_to_move();
    return pos.attacks_by<PAWN>(them) | pos.attacks_by<KNIGHT>(them) | pos.attacks_by<BISHOP>(them)
         | pos.attacks_by<ROOK>(them) | pos.attacks_by<QUEEN>(them) | pos.attacks_by<KING>(them);
}

// Index de l'historique principal : le coup, et l'etat de menace de ses deux
// cases. Repris de l'historique « 4D » d'Ethereal, Coda, Triumviratus.
inline int hist_index(Move m, Bitboard threats) {
    return m.from_to() + 4096 * (2 * bool(threats & m.from_sq()) + bool(threats & m.to_sq()));
}

// Le tri des coups est le poste le plus rentable d'une recherche alpha-beta :
// un bon ordre fait chuter le facteur de branchement effectif bien plus que
// n'importe quel elagage. La generation est etagee, pour ne jamais produire
// les coups tranquilles quand une capture suffit a couper.
//
// Ordre : coup de la TT, bonnes captures, refutations (killers et
// counter-move), coups tranquilles, mauvaises captures.
class MovePicker {
   public:
    MovePicker(const MovePicker&)            = delete;
    MovePicker& operator=(const MovePicker&) = delete;

    // Recherche principale et quiescence
    MovePicker(const Position&              p,
               Move                         ttm,
               Depth                        d,
               const ButterflyHistory*      mh,
               const CapturePieceToHistory* cph,
               const PieceToHistory**       ch,
               const PawnHistory*           ph,
               const Move*                  killers,
               Move                         cm,
               const LowPlyHistory*         lph,
               int                          pl);

    // ProbCut : uniquement les captures dont le SEE atteint le seuil
    MovePicker(const Position& p, Move ttm, int th, const CapturePieceToHistory* cph);

    Move next_move(bool skipQuiets = false);

   private:
    template<GenType>
    void score();

    template<typename Pred>
    Move select(Pred filter);

    ExtMove* begin() { return cur; }
    ExtMove* end() { return endMoves; }

    const Position&              pos;
    const ButterflyHistory*      mainHistory         = nullptr;
    const CapturePieceToHistory* captureHistory      = nullptr;
    const PieceToHistory**       continuationHistory = nullptr;
    const PawnHistory*           pawnHistory         = nullptr;
    const LowPlyHistory*         lowPlyHistory       = nullptr;

    Move     ttMove;
    Move     refutations[3];  // killer 1, killer 2, counter-move
    ExtMove *cur, *endMoves, *endBadCaptures;
    ExtMove *beginQuiets = nullptr, *endQuiets = nullptr;  // BadQuiets
    int      refutationIndex = 0;
    int      stage;
    int      threshold = 0;
    Depth    depth     = 0;
    int      ply       = 0;
    ExtMove  moves[MAX_MOVES];
};

}  // namespace one
