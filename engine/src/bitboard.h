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

// Ensembles de cases sur 64 bits et tables d'attaques (reecrits pour Onechess
// le 15-11-2026). Un bit par case, a1 = bit 0, h8 = bit 63.

#pragma once

#include <cstdlib>
#include <string>

#include "types.h"

#if defined(USE_PEXT)
    #include <immintrin.h>
#endif

namespace one {

namespace Bitboards {
void        init();               // a appeler une fois au demarrage
std::string pretty(Bitboard b);   // dessin en texte, pour le debogage
}

// ============================================================ colonnes, rangees
constexpr Bitboard colonne_bb(int f) { return 0x0101010101010101ULL << f; }
constexpr Bitboard rangee_bb(int r) { return 0xFFULL << (8 * r); }

constexpr Bitboard FileABB = colonne_bb(0), FileBBB = colonne_bb(1), FileCBB = colonne_bb(2),
                   FileDBB = colonne_bb(3), FileEBB = colonne_bb(4), FileFBB = colonne_bb(5),
                   FileGBB = colonne_bb(6), FileHBB = colonne_bb(7);
constexpr Bitboard Rank1BB = rangee_bb(0), Rank2BB = rangee_bb(1), Rank3BB = rangee_bb(2),
                   Rank4BB = rangee_bb(3), Rank5BB = rangee_bb(4), Rank6BB = rangee_bb(5),
                   Rank7BB = rangee_bb(6), Rank8BB = rangee_bb(7);

constexpr Bitboard file_bb(File f) { return colonne_bb(int(f)); }
constexpr Bitboard file_bb(Square s) { return colonne_bb(int(file_of(s))); }
constexpr Bitboard rank_bb(Rank r) { return rangee_bb(int(r)); }
constexpr Bitboard rank_bb(Square s) { return rangee_bb(int(rank_of(s))); }

constexpr Bitboard square_bb(Square s) {
    assert(is_ok(s));
    return Bitboard(1) << int(s);
}

// Melanges Bitboard / Square : « b & s » teste une case, « b |= s » l'ajoute...
inline Bitboard operator&(Bitboard b, Square s) { return b & square_bb(s); }
inline Bitboard operator|(Bitboard b, Square s) { return b | square_bb(s); }
inline Bitboard operator^(Bitboard b, Square s) { return b ^ square_bb(s); }
inline Bitboard operator&(Square s, Bitboard b) { return square_bb(s) & b; }
inline Bitboard operator|(Square s, Bitboard b) { return square_bb(s) | b; }
inline Bitboard operator^(Square s, Bitboard b) { return square_bb(s) ^ b; }
inline Bitboard operator|(Square a, Square b) { return square_bb(a) | square_bb(b); }
inline Bitboard& operator|=(Bitboard& b, Square s) { return b = b | s; }
inline Bitboard& operator^=(Bitboard& b, Square s) { return b = b ^ s; }

// ============================================================ operations de bits
inline int popcount(Bitboard b) { return __builtin_popcountll(b); }

constexpr bool more_than_one(Bitboard b) { return (b & (b - 1)) != 0; }  // au moins deux bits

inline Square lsb(Bitboard b) {  // case du bit le plus faible
    assert(b);
    return Square(__builtin_ctzll(b));
}
inline Square msb(Bitboard b) {  // case du bit le plus fort
    assert(b);
    return Square(63 - __builtin_clzll(b));
}
constexpr Bitboard least_significant_square_bb(Bitboard b) {  // le bit le plus faible, seul
    assert(b);
    return b & (~b + 1);
}
inline Square pop_lsb(Bitboard& b) {  // retire et rend la case du bit le plus faible
    const Square s = lsb(b);
    b &= b - 1;
    return s;
}

// Tout l'ensemble decale d'un pas dans la direction D ; ce qui sortirait par
// un cote de l'echiquier disparait au lieu de reapparaitre de l'autre.
template<Direction D>
constexpr Bitboard shift(Bitboard b) {
    constexpr int      d     = int(D);
    constexpr int      col   = ((d % 8) + 8) % 8 == 1 ? 1 : ((d % 8) + 8) % 8 == 7 ? -1 : 0;  // +1 est, -1 ouest
    constexpr Bitboard garde = col == 1 ? ~FileHBB : col == -1 ? ~FileABB : ~Bitboard(0);
    return d >= 0 ? (b & garde) << d : (b & garde) >> -d;
}

// ============================================================ tables
// Precalculees par Bitboards::init().
namespace Tables {
extern Bitboard Vide[PIECE_TYPE_NB][SQUARE_NB];  // attaques sur echiquier vide (cavalier..roi)
extern Bitboard Pion[COLOR_NB][SQUARE_NB];       // cases prises par un pion
extern Bitboard Entre[SQUARE_NB][SQUARE_NB];     // voir between_bb
extern Bitboard Ligne[SQUARE_NB][SQUARE_NB];     // voir line_bb
extern std::uint8_t Ecart[SQUARE_NB][SQUARE_NB]; // distance de roi

// Acces aux attaques d'une piece glissante : on ne garde de l'occupation que
// les cases qui peuvent bloquer (masque), on en tire un index (PEXT, ou
// multiplication par un nombre « magique » puis decalage), et on lit la table.
struct Glisseur {
    Bitboard        masque;
    const Bitboard* table;
#if !defined(USE_PEXT)
    Bitboard multiplicateur;
    int      decalage;
#endif
    unsigned index(Bitboard occ) const {
#if defined(USE_PEXT)
        return unsigned(_pext_u64(occ, masque));
#else
        return unsigned(((occ & masque) * multiplicateur) >> decalage);
#endif
    }
};
extern Glisseur Fous[SQUARE_NB];
extern Glisseur Tours[SQUARE_NB];
}  // namespace Tables

template<Color C>
constexpr Bitboard pawn_attacks_bb(Bitboard pions) {
    return C == WHITE ? shift<NORTH_EAST>(pions) | shift<NORTH_WEST>(pions)
                      : shift<SOUTH_EAST>(pions) | shift<SOUTH_WEST>(pions);
}
inline Bitboard pawn_attacks_bb(Color c, Square s) { return Tables::Pion[c][s]; }

// Toute la ligne (rangee, colonne ou diagonale) qui passe par a et b, ou
// rien si elles ne sont pas alignees.
inline Bitboard line_bb(Square a, Square b) { return Tables::Ligne[a][b]; }
inline bool     aligned(Square a, Square b, Square c) { return (Tables::Ligne[a][b] & c) != 0; }

// Les cases strictement entre a et b, plus b lui-meme (b seul si elles ne sont
// pas alignees) : exactement « s'interposer ou prendre » face a un echec de b.
inline Bitboard between_bb(Square a, Square b) { return Tables::Entre[a][b]; }

template<typename Axe = Square>
inline int distance(Square a, Square b) {
    if constexpr (std::is_same_v<Axe, File>)
        return std::abs(int(file_of(a)) - int(file_of(b)));
    else if constexpr (std::is_same_v<Axe, Rank>)
        return std::abs(int(rank_of(a)) - int(rank_of(b)));
    else
        return Tables::Ecart[a][b];
}

inline int edge_distance(File f) { return std::min(int(f), 7 - int(f)); }

// Attaques sur echiquier vide (pour un cavalier, un fou, une tour, une dame, un roi)
template<PieceType P>
inline Bitboard attacks_bb(Square s) {
    static_assert(P != PAWN);
    return Tables::Vide[P][s];
}
inline Bitboard empty_attacks_bb(PieceType p, Square s) { return Tables::Vide[p][s]; }

// Attaques en tenant compte des pieces presentes (occ)
template<PieceType P>
inline Bitboard attacks_bb(Square s, Bitboard occ) {
    static_assert(P != PAWN);
    if constexpr (P == BISHOP)
        return Tables::Fous[s].table[Tables::Fous[s].index(occ)];
    else if constexpr (P == ROOK)
        return Tables::Tours[s].table[Tables::Tours[s].index(occ)];
    else if constexpr (P == QUEEN)
        return attacks_bb<BISHOP>(s, occ) | attacks_bb<ROOK>(s, occ);
    else
        return Tables::Vide[P][s];
}

inline Bitboard attacks_bb(PieceType p, Square s, Bitboard occ) {
    switch (p)
    {
    case BISHOP : return attacks_bb<BISHOP>(s, occ);
    case ROOK : return attacks_bb<ROOK>(s, occ);
    case QUEEN : return attacks_bb<QUEEN>(s, occ);
    default : return Tables::Vide[p][s];
    }
}

}  // namespace one
