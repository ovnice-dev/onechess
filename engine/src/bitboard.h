#pragma once

#include <string>
#include "types.h"

#if defined(USE_PEXT)
    #include <immintrin.h>
#endif

namespace one {

namespace Bitboards {
void init();
std::string pretty(Bitboard b);
}

constexpr Bitboard FileABB = 0x0101010101010101ULL;
constexpr Bitboard FileBBB = FileABB << 1;
constexpr Bitboard FileCBB = FileABB << 2;
constexpr Bitboard FileDBB = FileABB << 3;
constexpr Bitboard FileEBB = FileABB << 4;
constexpr Bitboard FileFBB = FileABB << 5;
constexpr Bitboard FileGBB = FileABB << 6;
constexpr Bitboard FileHBB = FileABB << 7;

constexpr Bitboard Rank1BB = 0xFF;
constexpr Bitboard Rank2BB = Rank1BB << (8 * 1);
constexpr Bitboard Rank3BB = Rank1BB << (8 * 2);
constexpr Bitboard Rank4BB = Rank1BB << (8 * 3);
constexpr Bitboard Rank5BB = Rank1BB << (8 * 4);
constexpr Bitboard Rank6BB = Rank1BB << (8 * 5);
constexpr Bitboard Rank7BB = Rank1BB << (8 * 6);
constexpr Bitboard Rank8BB = Rank1BB << (8 * 7);

extern std::uint8_t SquareDistance[SQUARE_NB][SQUARE_NB];

extern Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
extern Bitboard LineBB[SQUARE_NB][SQUARE_NB];
extern Bitboard PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];
extern Bitboard PawnAttacksBB[COLOR_NB][SQUARE_NB];

// Une entree par case : masque d'occupation pertinente + pointeur dans la
// table partagee. Avec PEXT l'index est exact ; sinon on retombe sur les
// magic bitboards classiques.
struct Magic {
    Bitboard  mask;
    Bitboard* attacks;
#if !defined(USE_PEXT)
    Bitboard magic;
    unsigned shift;
#endif

    unsigned index(Bitboard occupied) const {
#if defined(USE_PEXT)
        return unsigned(_pext_u64(occupied, mask));
#else
        return unsigned(((occupied & mask) * magic) >> shift);
#endif
    }

    Bitboard attacks_bb(Bitboard occupied) const { return attacks[index(occupied)]; }
};

extern Magic RookMagics[SQUARE_NB];
extern Magic BishopMagics[SQUARE_NB];

constexpr Bitboard square_bb(Square s) {
    assert(is_ok(s));
    return (1ULL << s);
}

// ------------------------------------- operateurs Bitboard <-> Square
inline Bitboard  operator&(Bitboard b, Square s) { return b & square_bb(s); }
inline Bitboard  operator|(Bitboard b, Square s) { return b | square_bb(s); }
inline Bitboard  operator^(Bitboard b, Square s) { return b ^ square_bb(s); }
inline Bitboard& operator|=(Bitboard& b, Square s) { return b |= square_bb(s); }
inline Bitboard& operator^=(Bitboard& b, Square s) { return b ^= square_bb(s); }

inline Bitboard operator&(Square s, Bitboard b) { return b & s; }
inline Bitboard operator|(Square s, Bitboard b) { return b | s; }
inline Bitboard operator^(Square s, Bitboard b) { return b ^ s; }

inline Bitboard operator|(Square s1, Square s2) { return square_bb(s1) | s2; }

constexpr bool more_than_one(Bitboard b) { return b & (b - 1); }

constexpr Bitboard file_bb(File f) { return FileABB << f; }
constexpr Bitboard file_bb(Square s) { return file_bb(file_of(s)); }
constexpr Bitboard rank_bb(Rank r) { return Rank1BB << (8 * r); }
constexpr Bitboard rank_bb(Square s) { return rank_bb(rank_of(s)); }

// Decale tout le bitboard d'une direction, en tuant ce qui deborde lateralement.
template<Direction D>
constexpr Bitboard shift(Bitboard b) {
    return D == NORTH         ? b << 8
         : D == SOUTH         ? b >> 8
         : D == NORTH + NORTH ? b << 16
         : D == SOUTH + SOUTH ? b >> 16
         : D == EAST          ? (b & ~FileHBB) << 1
         : D == WEST          ? (b & ~FileABB) >> 1
         : D == NORTH_EAST    ? (b & ~FileHBB) << 9
         : D == NORTH_WEST    ? (b & ~FileABB) << 7
         : D == SOUTH_EAST    ? (b & ~FileHBB) >> 7
         : D == SOUTH_WEST    ? (b & ~FileABB) >> 9
                              : 0;
}

template<Color C>
constexpr Bitboard pawn_attacks_bb(Bitboard b) {
    return C == WHITE ? shift<NORTH_WEST>(b) | shift<NORTH_EAST>(b)
                      : shift<SOUTH_WEST>(b) | shift<SOUTH_EAST>(b);
}

inline Bitboard pawn_attacks_bb(Color c, Square s) {
    assert(is_ok(s));
    return PawnAttacksBB[c][s];
}

// Les deux cases sont-elles sur une meme ligne (rangee, colonne, diagonale) ?
inline bool aligned(Square s1, Square s2, Square s3) { return LineBB[s1][s2] & s3; }

// Ligne complete passant par s1 et s2 ; vide si elles ne sont pas alignees.
inline Bitboard line_bb(Square s1, Square s2) {
    assert(is_ok(s1) && is_ok(s2));
    return LineBB[s1][s2];
}

// Cases strictement entre s1 et s2, PLUS s2 lui-meme. Si les deux cases ne
// sont pas alignees, rend square_bb(s2). Cette convention rend le masque
// d'echec directement utilisable : "bloquer ou capturer".
inline Bitboard between_bb(Square s1, Square s2) {
    assert(is_ok(s1) && is_ok(s2));
    return BetweenBB[s1][s2];
}

template<typename T1 = Square>
inline int distance(Square x, Square y);

template<>
inline int distance<File>(Square x, Square y) { return std::abs(file_of(x) - file_of(y)); }

template<>
inline int distance<Rank>(Square x, Square y) { return std::abs(rank_of(x) - rank_of(y)); }

template<>
inline int distance<Square>(Square x, Square y) { return SquareDistance[x][y]; }

inline int edge_distance(File f) { return std::min(f, File(FILE_H - f)); }

// Destination d'un saut (cavalier, roi) : nulle si elle sort du plateau ou
// enjambe un bord. Le test de distance attrape le debordement lateral que
// le simple s + step ne voit pas.
inline Bitboard safe_destination(Square s, int step) {
    Square to = Square(s + step);
    return is_ok(to) && distance(s, to) <= 2 ? square_bb(to) : Bitboard(0);
}

// ---------------------------------------------------------------- attaques
template<PieceType Pt>
inline Bitboard attacks_bb(Square s) {
    assert((Pt != PAWN) && (is_ok(s)));
    return PseudoAttacks[Pt][s];
}

template<PieceType Pt>
inline Bitboard attacks_bb(Square s, Bitboard occupied) {
    assert((Pt != PAWN) && (is_ok(s)));

    switch (Pt)
    {
    case BISHOP :
        return BishopMagics[s].attacks_bb(occupied);
    case ROOK :
        return RookMagics[s].attacks_bb(occupied);
    case QUEEN :
        return attacks_bb<BISHOP>(s, occupied) | attacks_bb<ROOK>(s, occupied);
    default :
        return PseudoAttacks[Pt][s];
    }
}

inline Bitboard attacks_bb(PieceType pt, Square s, Bitboard occupied) {
    assert((pt != PAWN) && (is_ok(s)));

    switch (pt)
    {
    case BISHOP :
        return attacks_bb<BISHOP>(s, occupied);
    case ROOK :
        return attacks_bb<ROOK>(s, occupied);
    case QUEEN :
        return attacks_bb<BISHOP>(s, occupied) | attacks_bb<ROOK>(s, occupied);
    default :
        return PseudoAttacks[pt][s];
    }
}

// -------------------------------------------------------- primitives de bits
inline int popcount(Bitboard b) { return __builtin_popcountll(b); }

inline Square lsb(Bitboard b) {
    assert(b);
    return Square(__builtin_ctzll(b));
}

inline Square msb(Bitboard b) {
    assert(b);
    return Square(63 ^ __builtin_clzll(b));
}

// Bit de poids le plus faible, isole.
constexpr Bitboard least_significant_square_bb(Bitboard b) {
    assert(b);
    return b & -b;
}

inline Square pop_lsb(Bitboard& b) {
    assert(b);
    const Square s = lsb(b);
    b &= b - 1;
    return s;
}

// Case de destination du roi / de la tour apres le roque.
template<CastlingRights Cr>
constexpr Square king_castle_to() {
    return Cr == WHITE_OO ? SQ_G1 : Cr == WHITE_OOO ? SQ_C1 : Cr == BLACK_OO ? SQ_G8 : SQ_C8;
}

}  // namespace one
