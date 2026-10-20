#include "movepick.h"
#include "prof.h"
#include "tune.h"

#include <algorithm>
#include <limits>
#include <cassert>
#include <utility>

namespace one {

namespace {

enum Stages {
    // recherche principale
    MAIN_TT,
    CAPTURE_INIT,
    GOOD_CAPTURE,
    REFUTATION,
    QUIET_INIT,
    QUIET,
    BAD_CAPTURE,

    // parades d'echec
    EVASION_TT,
    EVASION_INIT,
    EVASION,

    // ProbCut
    PROBCUT_TT,
    PROBCUT_INIT,
    PROBCUT,

    // quiescence
    QSEARCH_TT,
    QCAPTURE_INIT,
    QCAPTURE
};

// Remonte en tete tous les coups dont le score atteint limit, tries entre eux.
// Un tri complet couterait plus que ce qu'il rapporte : la plupart des coups
// ne seront jamais atteints.
void partial_insertion_sort(ExtMove* begin, ExtMove* end, int limit) {
    for (ExtMove *sortedEnd = begin, *p = begin + 1; p < end; ++p)
        if (p->value >= limit)
        {
            ExtMove tmp = *p, *q;
            *p          = *++sortedEnd;
            for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
                *q = *(q - 1);
            *q = tmp;
        }
}

}  // namespace

MovePicker::MovePicker(const Position&              p,
                       Move                         ttm,
                       Depth                        d,
                       const ButterflyHistory*      mh,
                       const CapturePieceToHistory* cph,
                       const PieceToHistory**       ch,
                       const PawnHistory*           ph,
                       const Move*                  killers,
                       Move                         cm,
                       const LowPlyHistory*         lph,
                       int                          pl) :
    pos(p),
    mainHistory(mh),
    captureHistory(cph),
    continuationHistory(ch),
    pawnHistory(ph),
    lowPlyHistory(lph),
    ttMove(ttm),
    depth(d),
    ply(pl) {

    // Un counter-move identique a un killer serait essaye deux fois
    refutations[0] = killers[0];
    refutations[1] = killers[1];
    refutations[2] = (cm != killers[0] && cm != killers[1]) ? cm : Move::none();

    // Simplification a tester (Stockfish les a retires) : sans killers, sans
    // contre-coup. Les coups restent tries par leurs historiques.
    if (Tune::NoKillers)
        refutations[0] = refutations[1] = Move::none();
    if (Tune::NoCounter)
        refutations[2] = Move::none();

    if (pos.checkers())
        stage = EVASION_TT + !(ttm && pos.pseudo_legal(ttm));
    else
        stage = (d > 0 ? MAIN_TT : QSEARCH_TT) + !(ttm && pos.pseudo_legal(ttm));
}

MovePicker::MovePicker(const Position& p, Move ttm, int th, const CapturePieceToHistory* cph) :
    pos(p),
    captureHistory(cph),
    ttMove(ttm),
    threshold(th) {

    assert(!pos.checkers());

    refutations[0] = refutations[1] = refutations[2] = Move::none();

    stage = PROBCUT_TT
          + !(ttm && pos.capture_stage(ttm) && pos.pseudo_legal(ttm) && pos.see_ge(ttm, threshold));
}

// Attribue un score de tri a chaque coup de la tranche courante.
template<GenType Type>
void MovePicker::score() {
    static_assert(Type == CAPTURES || Type == QUIETS || Type == EVASIONS, "type non trie");

    // Cases attaquees par une piece adverse de MOINDRE valeur que celle qui
    // s'y trouve ou s'y rend : un pion menace un cavalier, une piece mineure
    // menace une tour, une tour menace une dame. Repris de Stockfish. Calcule
    // une fois par lot de coups tranquilles, et seulement si le terme est actif.
    [[maybe_unused]] Bitboard threatByLesser[PIECE_TYPE_NB];
    if constexpr (Type == QUIETS)
    {
        for (auto& b : threatByLesser)
            b = 0;
        if (Tune::ThreatOrder)
        {
            const Color them       = ~pos.side_to_move();
            threatByLesser[KNIGHT] = threatByLesser[BISHOP] = pos.attacks_by<PAWN>(them);
            threatByLesser[ROOK]   = pos.attacks_by<KNIGHT>(them) | pos.attacks_by<BISHOP>(them)
                                   | threatByLesser[KNIGHT];
            threatByLesser[QUEEN]  = pos.attacks_by<ROOK>(them) | threatByLesser[ROOK];
        }
    }

    const Bitboard threats = Type == CAPTURES ? Bitboard(0) : threat_map(pos);

    for (auto& m : *this)
    {
        if constexpr (Type == CAPTURES)
        {
            // MVV domine, l'historique de captures ne fait que departager.
            // C'est volontaire : quand l'historique prend le dessus sur la
            // valeur de la piece prise, l'arbre gonfle sans rien gagner.
            const Piece     moved    = pos.moved_piece(m);
            const PieceType captured = type_of(pos.piece_on(m.to_sq()));

            m.value = 7 * PieceValue[pos.piece_on(m.to_sq())]
                    + (*captureHistory)[moved][m.to_sq()][captured] / 16;

            // to_sq() est vide pour la prise en passant et la promotion en
            // dame par poussee : sans ces deux lignes elles seraient triees
            // comme des coups sans gain.
            if (m.type_of() == EN_PASSANT)
                m.value += 7 * PawnValue;
            else if (m.type_of() == PROMOTION)
                m.value += 7 * (PieceValue[m.promotion_type()] - PawnValue);
        }

        else if constexpr (Type == QUIETS)
        {
            const Piece  pc = pos.moved_piece(m);
            const Square to = m.to_sq();

            m.value = 2 * (*mainHistory)[pos.side_to_move()][hist_index(m, threats)];
            m.value += 2 * (*continuationHistory[0])[pc][to];
            m.value += (*continuationHistory[1])[pc][to];
            m.value += (*continuationHistory[2])[pc][to];
            m.value += (*continuationHistory[3])[pc][to];
            m.value += (*pawnHistory)[pawn_structure_index(pos.pawn_key())][pc][to];

            // Pres de la racine, ce qui a coupe dans CE noeud aux iterations
            // precedentes pese plus qu'une moyenne sur tout l'arbre ; le poids
            // decroit avec la profondeur. Inerte tant que LowPlyHist vaut 0.
            if (Tune::LowPlyHist && ply < LOW_PLY_HISTORY_SIZE)
                m.value += Tune::LowPlyHist * (*lowPlyHistory)[ply][m.from_to()] / (1 + 2 * ply);

            // Un coup qui donne echec merite d'etre vu tot -- sauf s'il perd
            // la piece a l'echange : un echec qui laisse prendre la piece est
            // rarement le coup qui coupe. Stockfish exige see_ge(m, -75) a son
            // echelle. Inerte tant que CheckSeeMargin vaut 0 : le bonus va
            // alors a tous les echecs, comme avant.
            if ((pos.check_squares(type_of(pc)) & to)
                && (!Tune::CheckSeeMargin || pos.see_ge(m, -Tune::CheckSeeMargin)))
                m.value += 16384;

            // Mettre une piece a l'abri d'une menace par moindre qu'elle est
            // souvent le coup qui coupe ; l'y exposer, presque jamais. Le
            // diagnostic du 11-09 a montre que c'est la que le tri pechait :
            // 83,3 % de coupures au premier coup sans coup de table, contre
            // 93,1 % avec. Inerte tant que ThreatOrder vaut 0.
            if (Tune::ThreatOrder)
            {
                const PieceType pt = type_of(pc);
                m.value += PieceValue[pc] * Tune::ThreatOrder
                         * (int(bool(threatByLesser[pt] & m.from_sq()))
                            - int(bool(threatByLesser[pt] & to)));
            }
        }

        else  // EVASIONS : capturer l'attaquant d'abord, puis l'historique
        {
            if (pos.capture(m))
                m.value = PieceValue[pos.piece_on(m.to_sq())] - type_of(pos.moved_piece(m))
                        + (1 << 28);
            else
                m.value = (*mainHistory)[pos.side_to_move()][hist_index(m, threats)]
                        + (*continuationHistory[0])[pos.moved_piece(m)][m.to_sq()];
        }
    }
}

// Rend le premier coup restant qui passe le filtre, et avance le curseur.
template<typename Pred>
Move MovePicker::select(Pred filter) {
    for (; cur < endMoves; ++cur)
        if (*cur != ttMove && filter())
            return *cur++;

    return Move::none();
}

// Coup suivant dans l'ordre de tri. On ne genere une tranche que lorsqu'on en
// a reellement besoin.
Move MovePicker::next_move(bool skipQuiets) {
    PROF_SCOPE(MOVEPICK);
top:
    switch (stage)
    {

    case MAIN_TT :
    case EVASION_TT :
    case QSEARCH_TT :
    case PROBCUT_TT :
        ++stage;
        return ttMove;

    case CAPTURE_INIT :
    case PROBCUT_INIT :
    case QCAPTURE_INIT :
        cur = endBadCaptures = moves;
        endMoves             = generate<CAPTURES>(pos, cur);

        score<CAPTURES>();
        partial_insertion_sort(cur, endMoves, std::numeric_limits<int>::min());
        ++stage;
        goto top;

    case GOOD_CAPTURE :
        // Une capture perdante au sens SEE est mise de cote, pas jetee :
        // elle repassera en fin de liste.
        if (select([&]() {
                return pos.see_ge(*cur) ? true : (*endBadCaptures++ = *cur, false);
            }))
            return *(cur - 1);

        ++stage;
        [[fallthrough]];

    case REFUTATION :
        for (; refutationIndex < 3; ++refutationIndex)
        {
            Move m = refutations[refutationIndex];
            if (m && m != ttMove && !pos.capture_stage(m) && pos.pseudo_legal(m))
            {
                ++refutationIndex;
                return m;
            }
        }
        ++stage;
        [[fallthrough]];

    case QUIET_INIT :
        if (!skipQuiets)
        {
            cur      = endBadCaptures;
            endMoves = generate<QUIETS>(pos, cur);

            score<QUIETS>();
            partial_insertion_sort(cur, endMoves, -3000 * depth);
        }

        ++stage;
        [[fallthrough]];

    case QUIET :
        if (!skipQuiets
            && select([&]() {
                   return *cur != refutations[0] && *cur != refutations[1]
                       && *cur != refutations[2];
               }))
            return *(cur - 1);

        // Les mauvaises captures ont ete accumulees en tete du tampon
        cur      = moves;
        endMoves = endBadCaptures;

        ++stage;
        [[fallthrough]];

    case BAD_CAPTURE :
        return select([]() { return true; });

    case EVASION_INIT :
        cur      = moves;
        endMoves = generate<EVASIONS>(pos, cur);

        score<EVASIONS>();
        partial_insertion_sort(cur, endMoves, std::numeric_limits<int>::min());
        ++stage;
        [[fallthrough]];

    case EVASION :
    case QCAPTURE :
        return select([]() { return true; });

    case PROBCUT :
        return select([&]() { return pos.see_ge(*cur, threshold); });
    }

    assert(false);
    return Move::none();
}

}  // namespace one
