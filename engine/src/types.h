#pragma once

#include <cstdint>
#include <cassert>
#include <algorithm>

namespace one {

using Bitboard = uint64_t;
using Key      = uint64_t;

constexpr int MAX_MOVES = 256;
constexpr int MAX_PLY   = 246;

// ---------------------------------------------------------------- couleurs
enum Color : int { WHITE, BLACK, COLOR_NB = 2 };

constexpr Color operator~(Color c) { return Color(c ^ BLACK); }

// ------------------------------------------------------------------ pieces
enum PieceType : int {
    NO_PIECE_TYPE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    ALL_PIECES    = 0,
    PIECE_TYPE_NB = 8
};

enum Piece : int {
    NO_PIECE,
    W_PAWN = PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN = PAWN + 8, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    PIECE_NB = 16
};

constexpr Piece     make_piece(Color c, PieceType pt) { return Piece((c << 3) + pt); }
constexpr PieceType type_of(Piece pc)                 { return PieceType(pc & 7); }
constexpr Color     color_of(Piece pc)                { assert(pc != NO_PIECE); return Color(pc >> 3); }

// ------------------------------------------------------------------ scores
using Value = int;

constexpr Value VALUE_ZERO      = 0;
constexpr Value VALUE_DRAW      = 0;
constexpr Value VALUE_NONE      = 32002;
constexpr Value VALUE_INFINITE  = 32001;
constexpr Value VALUE_MATE      = 32000;
constexpr Value VALUE_MATE_IN_MAX_PLY  = VALUE_MATE - MAX_PLY;
constexpr Value VALUE_MATED_IN_MAX_PLY = -VALUE_MATE_IN_MAX_PLY;
// Gains et pertes prouves par les tables de finales : VALUE_TB - ply, juste
// sous les mats, et au-dessus du seuil VALUE_TB_WIN_IN_MAX_PLY (comme chez
// Stockfish). Avant le 20-10, le gain valait VALUE_TB_WIN_IN_MAX_PLY - ply,
// c'est-a-dire SOUS le seuil : la recherche le prenait pour une evaluation.
constexpr Value VALUE_TB                 = VALUE_MATE_IN_MAX_PLY - 1;
constexpr Value VALUE_TB_WIN_IN_MAX_PLY  = VALUE_TB - MAX_PLY;
constexpr Value VALUE_TB_LOSS_IN_MAX_PLY = -VALUE_TB_WIN_IN_MAX_PLY;

constexpr Value mate_in(int ply)  { return VALUE_MATE - ply; }
constexpr Value mated_in(int ply) { return ply - VALUE_MATE; }

// ------------------------------------------------------------------ cases
// A1 = 0, B1 = 1, ... H8 = 63 (rangee-majeur)
enum Square : int {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    SQ_NONE,
    SQUARE_ZERO = 0,
    SQUARE_NB   = 64
};

enum File : int { FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_NB };
enum Rank : int { RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_NB };

enum Direction : int {
    NORTH = 8, EAST = 1, SOUTH = -NORTH, WEST = -EAST,
    NORTH_EAST = NORTH + EAST,
    SOUTH_EAST = SOUTH + EAST,
    SOUTH_WEST = SOUTH + WEST,
    NORTH_WEST = NORTH + WEST
};

// -------------------------------------------------------------- roque
enum CastlingRights : int {
    NO_CASTLING = 0,
    WHITE_OO    = 1,
    WHITE_OOO   = 2,
    BLACK_OO    = 4,
    BLACK_OOO   = 8,

    KING_SIDE      = WHITE_OO | BLACK_OO,
    QUEEN_SIDE     = WHITE_OOO | BLACK_OOO,
    WHITE_CASTLING = WHITE_OO | WHITE_OOO,
    BLACK_CASTLING = BLACK_OO | BLACK_OOO,
    ANY_CASTLING   = WHITE_CASTLING | BLACK_CASTLING,

    CASTLING_RIGHT_NB = 16
};

// ------------------------------------------------- operateurs sur les enums
#define ENABLE_INCR_OPERATORS_ON(T) \
    inline T& operator++(T& d) { return d = T(int(d) + 1); } \
    inline T& operator--(T& d) { return d = T(int(d) - 1); }

ENABLE_INCR_OPERATORS_ON(PieceType)
ENABLE_INCR_OPERATORS_ON(Square)
ENABLE_INCR_OPERATORS_ON(File)
ENABLE_INCR_OPERATORS_ON(Rank)

#undef ENABLE_INCR_OPERATORS_ON

constexpr Direction operator+(Direction d1, Direction d2) { return Direction(int(d1) + int(d2)); }
constexpr Direction operator*(int i, Direction d)         { return Direction(i * int(d)); }

constexpr Square  operator+(Square s, Direction d) { return Square(int(s) + int(d)); }
constexpr Square  operator-(Square s, Direction d) { return Square(int(s) - int(d)); }
inline    Square& operator+=(Square& s, Direction d) { return s = s + d; }
inline    Square& operator-=(Square& s, Direction d) { return s = s - d; }

constexpr Direction operator-(Square a, Square b) { return Direction(int(a) - int(b)); }

constexpr CastlingRights operator&(Color c, CastlingRights cr) {
    return CastlingRights((c == WHITE ? WHITE_CASTLING : BLACK_CASTLING) & cr);
}

// -------------------------------------------------------------- utilitaires
constexpr bool is_ok(Square s) { return s >= SQ_A1 && s <= SQ_H8; }

constexpr File file_of(Square s) { return File(s & 7); }
constexpr Rank rank_of(Square s) { return Rank(s >> 3); }

constexpr Square make_square(File f, Rank r) { return Square((r << 3) + f); }

// Case vue depuis le camp c : identite pour les blancs, miroir vertical pour les noirs
constexpr Square relative_square(Color c, Square s) { return Square(s ^ (c * 56)); }
constexpr Rank   relative_rank(Color c, Rank r)     { return Rank(r ^ (c * 7)); }
constexpr Rank   relative_rank(Color c, Square s)   { return relative_rank(c, rank_of(s)); }

constexpr Direction pawn_push(Color c) { return c == WHITE ? NORTH : SOUTH; }

// -------------------------------------------------------------------- coup
// bits  0- 5 : case d'arrivee
// bits  6-11 : case de depart
// bits 12-13 : type de promotion (KNIGHT..QUEEN) - 2
// bits 14-15 : type de coup
// Le roque est encode roi -> tour (compatible Chess960 d'entree de jeu).
enum MoveType : int {
    NORMAL     = 0,
    PROMOTION  = 1 << 14,
    EN_PASSANT = 2 << 14,
    CASTLING   = 3 << 14
};

class Move {
   public:
    Move() = default;
    constexpr explicit Move(std::uint16_t d) : data(d) {}
    constexpr Move(Square from, Square to) : data((from << 6) + to) {}

    template<MoveType T>
    static constexpr Move make(Square from, Square to, PieceType pt = KNIGHT) {
        return Move(T + ((pt - KNIGHT) << 12) + (from << 6) + to);
    }

    constexpr Square    from_sq()   const { assert(is_ok()); return Square((data >> 6) & 0x3F); }
    constexpr Square    to_sq()     const { assert(is_ok()); return Square(data & 0x3F); }
    constexpr int       from_to()   const { return data & 0xFFF; }
    constexpr MoveType  type_of()   const { return MoveType(data & (3 << 14)); }
    constexpr PieceType promotion_type() const { return PieceType(((data >> 12) & 3) + KNIGHT); }

    constexpr bool is_ok() const { return none().data != data && null().data != data; }

    static constexpr Move null() { return Move(65); }
    static constexpr Move none() { return Move(0); }

    constexpr bool operator==(const Move& m) const = default;
    constexpr explicit operator bool() const { return data != 0; }

    constexpr std::uint16_t raw() const { return data; }

   protected:
    std::uint16_t data;
};

// ------------------------------------------------------------------- bornes
enum Bound : std::uint8_t {
    BOUND_NONE  = 0,
    BOUND_UPPER = 1,
    BOUND_LOWER = 2,
    BOUND_EXACT = BOUND_UPPER | BOUND_LOWER
};

using Depth = int;

constexpr Depth DEPTH_QS       = 0;
constexpr Depth DEPTH_NONE     = -6;
constexpr Depth DEPTH_ENTRY_TT = -7;

}  // namespace one
