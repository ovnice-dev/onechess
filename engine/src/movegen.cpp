#include "movegen.h"

#include <cassert>
#include <initializer_list>

#include "bitboard.h"
#include "position.h"

namespace one {

namespace {

// Repartition des promotions entre les deux etages de generation, calee sur
// Position::capture_stage() :
//   CAPTURES : promotions par prise (4 pieces) + promotion en dame par poussee
//   QUIETS   : sous-promotions par poussee uniquement
// Chaque coup est ainsi genere exactement une fois par CAPTURES + QUIETS.
template<Color Us, GenType Type>
ExtMove* generate_pawn_moves(const Position& pos, ExtMove* moveList, Bitboard target) {
    constexpr Color     Them     = ~Us;
    constexpr Bitboard  TRank7BB = (Us == WHITE ? Rank7BB : Rank2BB);
    constexpr Bitboard  TRank3BB = (Us == WHITE ? Rank3BB : Rank6BB);
    constexpr Direction Up       = pawn_push(Us);
    constexpr Direction UpRight  = (Us == WHITE ? NORTH_EAST : SOUTH_WEST);
    constexpr Direction UpLeft   = (Us == WHITE ? NORTH_WEST : SOUTH_EAST);

    const Bitboard emptySquares = ~pos.pieces();
    const Bitboard enemies      = Type == EVASIONS ? pos.checkers() : pos.pieces(Them);

    const Bitboard pawnsOn7    = pos.pieces(Us, PAWN) & TRank7BB;
    const Bitboard pawnsNotOn7 = pos.pieces(Us, PAWN) & ~TRank7BB;

    // ------------------------------------------------------------- poussees
    if constexpr (Type != CAPTURES)
    {
        Bitboard b1 = shift<Up>(pawnsNotOn7) & emptySquares;
        Bitboard b2 = shift<Up>(b1 & TRank3BB) & emptySquares;

        if constexpr (Type == EVASIONS)
        {
            // En echec, seules les cases qui bloquent comptent
            b1 &= target;
            b2 &= target;
        }

        while (b1)
        {
            Square to   = pop_lsb(b1);
            *moveList++ = Move(to - Up, to);
        }

        while (b2)
        {
            Square to   = pop_lsb(b2);
            *moveList++ = Move(to - Up - Up, to);
        }
    }

    // ----------------------------------------------------------- promotions
    if (pawnsOn7)
    {
        auto add_promotions = [&](Square to, Square from, bool isCapture) {
            if constexpr (Type != QUIETS)
                *moveList++ = Move::make<PROMOTION>(from, to, QUEEN);

            if (Type == EVASIONS || Type == NON_EVASIONS || (Type == CAPTURES && isCapture)
                || (Type == QUIETS && !isCapture))
            {
                *moveList++ = Move::make<PROMOTION>(from, to, ROOK);
                *moveList++ = Move::make<PROMOTION>(from, to, BISHOP);
                *moveList++ = Move::make<PROMOTION>(from, to, KNIGHT);
            }
        };

        if constexpr (Type != QUIETS)
        {
            Bitboard b1 = shift<UpRight>(pawnsOn7) & enemies;
            Bitboard b2 = shift<UpLeft>(pawnsOn7) & enemies;

            while (b1)
            {
                Square to = pop_lsb(b1);
                add_promotions(to, to - UpRight, true);
            }

            while (b2)
            {
                Square to = pop_lsb(b2);
                add_promotions(to, to - UpLeft, true);
            }
        }

        Bitboard b3 = shift<Up>(pawnsOn7) & emptySquares;

        if constexpr (Type == EVASIONS)
            b3 &= target;

        while (b3)
        {
            Square to = pop_lsb(b3);
            add_promotions(to, to - Up, false);
        }
    }

    // -------------------------------------------------------------- prises
    if constexpr (Type != QUIETS)
    {
        Bitboard b1 = shift<UpRight>(pawnsNotOn7) & enemies;
        Bitboard b2 = shift<UpLeft>(pawnsNotOn7) & enemies;

        while (b1)
        {
            Square to   = pop_lsb(b1);
            *moveList++ = Move(to - UpRight, to);
        }

        while (b2)
        {
            Square to   = pop_lsb(b2);
            *moveList++ = Move(to - UpLeft, to);
        }

        // La prise en passant est generee sans condition supplementaire, meme
        // en parade d'echec : Position::legal() est prouve suffisant ici, et
        // un filtre approximatif ferait perdre des coups legaux.
        if (pos.state()->epSquare != SQ_NONE)
        {
            const Square epSq = pos.state()->epSquare;
            assert(rank_of(epSq) == relative_rank(Us, RANK_6));

            Bitboard b3 = pawnsNotOn7 & pawn_attacks_bb(Them, epSq);

            // epSquare n'est pose que si une prise existe reellement
            assert(b3);

            while (b3)
                *moveList++ = Move::make<EN_PASSANT>(pop_lsb(b3), epSq);
        }
    }

    return moveList;
}

template<Color Us, PieceType Pt>
ExtMove* generate_moves(const Position& pos, ExtMove* moveList, Bitboard target) {
    static_assert(Pt != KING && Pt != PAWN, "roi et pions ont leur propre generateur");

    Bitboard bb = pos.pieces(Us, Pt);

    while (bb)
    {
        Square   from = pop_lsb(bb);
        Bitboard b    = attacks_bb<Pt>(from, pos.pieces()) & target;

        while (b)
            *moveList++ = Move(from, pop_lsb(b));
    }

    return moveList;
}

template<Color Us, GenType Type>
ExtMove* generate_all(const Position& pos, ExtMove* moveList) {
    const Square ksq    = pos.square<KING>(Us);
    Bitboard     target = 0;

    // Sur double echec, seul le roi peut bouger : on saute tout le reste.
    if (Type != EVASIONS || !more_than_one(pos.checkers()))
    {
        target = Type == EVASIONS     ? between_bb(ksq, lsb(pos.checkers()))
               : Type == NON_EVASIONS ? ~pos.pieces(Us)
               : Type == CAPTURES     ? pos.pieces(~Us)
                                      : ~pos.pieces();  // QUIETS

        moveList = generate_pawn_moves<Us, Type>(pos, moveList, target);
        moveList = generate_moves<Us, KNIGHT>(pos, moveList, target);
        moveList = generate_moves<Us, BISHOP>(pos, moveList, target);
        moveList = generate_moves<Us, ROOK>(pos, moveList, target);
        moveList = generate_moves<Us, QUEEN>(pos, moveList, target);
    }

    Bitboard b = attacks_bb<KING>(ksq) & (Type == EVASIONS ? ~pos.pieces(Us) : target);

    while (b)
        *moveList++ = Move(ksq, pop_lsb(b));

    if constexpr (Type == QUIETS || Type == NON_EVASIONS)
        if (pos.can_castle(Us & ANY_CASTLING))
            for (CastlingRights cr : {Us & KING_SIDE, Us & QUEEN_SIDE})
                if (!pos.castling_impeded(cr) && pos.can_castle(cr))
                    *moveList++ = Move::make<CASTLING>(ksq, pos.castling_rook_square(cr));

    return moveList;
}

}  // namespace

// Genere les coups pseudo-legaux du type demande. Les coups qui laisseraient
// le roi en prise sont encore dedans : c'est generate<LEGAL> ou le tri des
// coups qui les ecarte, avec pos.legal().
template<GenType Type>
ExtMove* generate(const Position& pos, ExtMove* moveList) {
    static_assert(Type != LEGAL, "LEGAL a sa propre specialisation");
    assert((Type == EVASIONS) == bool(pos.checkers()));

    Color us = pos.side_to_move();

    return us == WHITE ? generate_all<WHITE, Type>(pos, moveList)
                       : generate_all<BLACK, Type>(pos, moveList);
}

template ExtMove* generate<CAPTURES>(const Position&, ExtMove*);
template ExtMove* generate<QUIETS>(const Position&, ExtMove*);
template ExtMove* generate<EVASIONS>(const Position&, ExtMove*);
template ExtMove* generate<NON_EVASIONS>(const Position&, ExtMove*);

// Filtre les coups pseudo-legaux. Seuls trois cas peuvent etre illegaux :
// une piece clouee, un coup de roi, une prise en passant.
template<>
ExtMove* generate<LEGAL>(const Position& pos, ExtMove* moveList) {
    Color    us     = pos.side_to_move();
    Bitboard pinned = pos.blockers_for_king(us) & pos.pieces(us);
    Square   ksq    = pos.square<KING>(us);
    ExtMove* cur    = moveList;

    moveList =
      pos.checkers() ? generate<EVASIONS>(pos, moveList) : generate<NON_EVASIONS>(pos, moveList);

    while (cur != moveList)
        if (((pinned & cur->from_sq()) || cur->from_sq() == ksq || cur->type_of() == EN_PASSANT)
            && !pos.legal(*cur))
            *cur = *(--moveList);
        else
            ++cur;

    return moveList;
}

}  // namespace one
