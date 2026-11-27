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

// Types de base du moteur (reecrits pour Onechess le 15-11-2026). Les noms
// (Square, SQ_E4, PieceType, ...) sont le vocabulaire courant des moteurs
// d'echecs ; tout le reste est notre code.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <type_traits>

namespace one {

using Bitboard = std::uint64_t;
using Key      = std::uint64_t;
using Value    = int;
using Depth    = int;

constexpr int MAX_MOVES = 256;  // plus que le maximum legal connu (218)
constexpr int MAX_PLY   = 246;

// ===================================================================== camps
enum Color : int { WHITE = 0, BLACK = 1, COLOR_NB = 2 };

constexpr Color operator~(Color c) { return Color(int(c) ^ 1); }  // l'autre camp

// ==================================================================== pieces
// Type sur 3 bits, couleur sur le bit 3 : piece = couleur * 8 + type.
enum PieceType : int { NO_PIECE_TYPE = 0, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, PIECE_TYPE_NB = 8 };
constexpr PieceType ALL_PIECES = NO_PIECE_TYPE;  // « toutes les pieces » dans les tables par type

enum Piece : int {
    NO_PIECE = 0,
    W_PAWN = 1, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN = 9, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    PIECE_NB = 16
};

constexpr Piece     make_piece(Color c, PieceType t) { return Piece(int(c) * 8 + int(t)); }
constexpr PieceType type_of(Piece p) { return PieceType(int(p) & 7); }
constexpr Color color_of(Piece p) {
    assert(p != NO_PIECE);
    return Color(int(p) >> 3);
}

// ==================================================================== scores
// Echelle (du haut vers le bas) : NONE > INFINITE > MATE > mats en N > gains
// prouves par les tables (TB) > evaluations ordinaires.
constexpr Value VALUE_ZERO     = 0;
constexpr Value VALUE_DRAW     = 0;
constexpr Value VALUE_MATE     = 32000;
constexpr Value VALUE_INFINITE = VALUE_MATE + 1;
constexpr Value VALUE_NONE     = VALUE_MATE + 2;

constexpr Value VALUE_MATE_IN_MAX_PLY  = VALUE_MATE - MAX_PLY;  // plus petit score de mat
constexpr Value VALUE_MATED_IN_MAX_PLY = -(VALUE_MATE - MAX_PLY);
// Gain prouve par les tables a la distance ply : VALUE_TB - ply, entre les
// mats et les evaluations (jusqu'au 20-10, il tombait sous le seuil et la
// recherche le traitait comme une simple evaluation).
constexpr Value VALUE_TB                 = VALUE_MATE_IN_MAX_PLY - 1;
constexpr Value VALUE_TB_WIN_IN_MAX_PLY  = VALUE_TB - MAX_PLY;
constexpr Value VALUE_TB_LOSS_IN_MAX_PLY = -(VALUE_TB - MAX_PLY);

constexpr Value mate_in(int ply) { return VALUE_MATE - ply; }    // on mate en ply demi-coups
constexpr Value mated_in(int ply) { return -VALUE_MATE + ply; }  // on est mate en ply

// ===================================================================== cases
// Numerotation par rangees : a1 = 0, b1 = 1, ..., h1 = 7, a2 = 8, ..., h8 = 63.
// clang-format off
enum Square : int {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1, SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3, SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5, SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7, SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    SQ_NONE, SQUARE_NB = 64, SQUARE_ZERO = 0
};
// clang-format on

enum File : int { FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_NB };
enum Rank : int { RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_NB };

// Pas d'une case a l'autre (+8 = une rangee vers les noirs)
enum Direction : int {
    NORTH = 8, SOUTH = -8, EAST = 1, WEST = -1,
    NORTH_EAST = 9, NORTH_WEST = 7, SOUTH_EAST = -7, SOUTH_WEST = -9
};

constexpr bool is_ok(Square s) { return unsigned(s) < 64u; }
constexpr File file_of(Square s) { return File(int(s) & 7); }
constexpr Rank rank_of(Square s) { return Rank(int(s) >> 3); }
constexpr Square make_square(File f, Rank r) { return Square(int(r) * 8 + int(f)); }

// Vu du camp c : les noirs voient l'echiquier retourne (rangee r -> 7 - r)
constexpr Rank relative_rank(Color c, Rank r) { return Rank(int(r) ^ (7 * int(c))); }
constexpr Rank relative_rank(Color c, Square s) { return relative_rank(c, rank_of(s)); }
constexpr Square relative_square(Color c, Square s) { return Square(int(s) ^ (56 * int(c))); }

constexpr Direction pawn_push(Color c) { return c == WHITE ? NORTH : SOUTH; }

// Arithmetique des cases et directions (toutes en entiers)
constexpr Direction operator+(Direction a, Direction b) { return Direction(int(a) + int(b)); }
constexpr Direction operator*(int k, Direction d) { return Direction(k * int(d)); }
constexpr Square operator+(Square s, Direction d) { return Square(int(s) + int(d)); }
constexpr Square operator-(Square s, Direction d) { return Square(int(s) - int(d)); }
constexpr Direction operator-(Square a, Square b) { return Direction(int(a) - int(b)); }
inline Square& operator+=(Square& s, Direction d) { return s = s + d; }
inline Square& operator-=(Square& s, Direction d) { return s = s - d; }

// Pas suivant / precedent pour parcourir une enumeration avec ++ et --
template<typename E>
concept Pas_a_pas = std::is_same_v<E, PieceType> || std::is_same_v<E, Square> || std::is_same_v<E, File>
                 || std::is_same_v<E, Rank>;
template<Pas_a_pas E>
inline E& operator++(E& e) { return e = E(int(e) + 1); }
template<Pas_a_pas E>
inline E& operator--(E& e) { return e = E(int(e) - 1); }

// ==================================================================== roques
// Un bit par droit : petit et grand roque, pour chaque camp.
enum CastlingRights : int {
    NO_CASTLING = 0,
    WHITE_OO = 1, WHITE_OOO = 2, BLACK_OO = 4, BLACK_OOO = 8,
    WHITE_CASTLING = 3, BLACK_CASTLING = 12, ANY_CASTLING = 15,
    KING_SIDE = WHITE_OO | BLACK_OO, QUEEN_SIDE = WHITE_OOO | BLACK_OOO,
    CASTLING_RIGHT_NB = 16
};

// Les droits d'un camp parmi ceux demandes
constexpr CastlingRights operator&(Color c, CastlingRights r) {
    return CastlingRights(int(r) & (3 << (2 * int(c))));
}

// ===================================================================== coups
// 16 bits : arrivee (bits 0-5), depart (6-11), piece de promotion moins le
// cavalier (12-13), genre du coup (14-15). Le roque se note « roi prend sa
// tour », ce qui couvre le Chess960 sans cas particulier.
enum MoveType : int { NORMAL = 0, PROMOTION = 0x4000, EN_PASSANT = 0x8000, CASTLING = 0xC000 };

class Move {
   public:
    Move() = default;
    constexpr explicit Move(std::uint16_t brut) : data(brut) {}
    constexpr Move(Square de, Square a) : data(std::uint16_t(int(de) * 64 + int(a))) {}

    template<MoveType G>
    static constexpr Move make(Square de, Square a, PieceType promue = KNIGHT) {
        return Move(std::uint16_t(int(G) | (int(promue) - int(KNIGHT)) << 12 | int(de) << 6 | int(a)));
    }

    // Deux valeurs reservees : « pas de coup » (0) et le coup nul (65 = b1 -> b1)
    static constexpr Move none() { return Move(std::uint16_t(0)); }
    static constexpr Move null() { return Move(std::uint16_t(65)); }

    constexpr Square to_sq() const {
        assert(is_ok());
        return Square(data & 63);
    }
    constexpr Square from_sq() const {
        assert(is_ok());
        return Square(data >> 6 & 63);
    }
    constexpr int       from_to() const { return data & 0x0FFF; }  // depart et arrivee ensemble
    constexpr MoveType  type_of() const { return MoveType(data & 0xC000); }
    constexpr PieceType promotion_type() const { return PieceType((data >> 12 & 3) + int(KNIGHT)); }

    constexpr bool          is_ok() const { return data != 0 && data != 65; }
    constexpr std::uint16_t raw() const { return data; }

    constexpr bool operator==(const Move&) const = default;
    constexpr explicit operator bool() const { return data != 0; }

   protected:
    std::uint16_t data;
};

// ============================================== bornes et profondeurs (table)
enum Bound : std::uint8_t { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

constexpr Depth DEPTH_QS       = 0;   // quiescence
constexpr Depth DEPTH_NONE     = -6;  // entree sans profondeur
constexpr Depth DEPTH_ENTRY_TT = -7;  // decalage du stockage en table

}  // namespace one
