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

// Construction des tables d'attaques (reecrite pour Onechess le 15-11-2026).

#include "bitboard.h"

#include <vector>

namespace one {

namespace Tables {
Bitboard     Vide[PIECE_TYPE_NB][SQUARE_NB];
Bitboard     Pion[COLOR_NB][SQUARE_NB];
Bitboard     Entre[SQUARE_NB][SQUARE_NB];
Bitboard     Ligne[SQUARE_NB][SQUARE_NB];
std::uint8_t Ecart[SQUARE_NB][SQUARE_NB];
Glisseur     Fous[SQUARE_NB];
Glisseur     Tours[SQUARE_NB];
}  // namespace Tables

namespace {

using namespace Tables;

// Une seule reserve pour les deux glisseurs : 102 400 entrees pour les tours,
// 5 248 pour les fous (somme des 2^bits des masques).
Bitboard Reserve[102400 + 5248];

// Les quatre rayons d'une piece glissante, en (colonne, rangee)
struct Rayon { int dc, dr; };
constexpr Rayon RayonsTour[4] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
constexpr Rayon RayonsFou[4]  = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}};

// Attaques reelles le long des rayons, en s'arretant sur la premiere case
// occupee (comprise). Lent : sert seulement a remplir les tables.
Bitboard lancer(const Rayon (&rayons)[4], Square depart, Bitboard occ) {
    Bitboard r = 0;
    for (const Rayon& ray : rayons)
    {
        int c = int(file_of(depart)) + ray.dc, g = int(rank_of(depart)) + ray.dr;
        for (; c >= 0 && c < 8 && g >= 0 && g < 8; c += ray.dc, g += ray.dr)
        {
            const Square s = make_square(File(c), Rank(g));
            r |= s;
            if (occ & s)
                break;
        }
    }
    return r;
}

// Melangeur 64 bits « splitmix » : suite deterministe, assez bonne pour
// tirer des multiplicateurs candidats.
struct Hasard {
    std::uint64_t etat;
    std::uint64_t suivant() {
        std::uint64_t z = (etat += 0x9E3779B97F4A7C15ULL);
        z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    // Peu de bits a 1 : ce sont les candidats qui reussissent le plus vite
    std::uint64_t clairseme() { return suivant() & suivant() & suivant(); }
};

// Remplit les glisseurs d'un type de piece a partir de « place » ; rend la
// premiere entree libre de la reserve apres eux.
Bitboard* construire(Glisseur (&g)[SQUARE_NB], const Rayon (&rayons)[4], Bitboard* place) {
    std::vector<Bitboard> occs, vraies;
#if !defined(USE_PEXT)
    std::vector<int> tour;  // numero d'essai qui a ecrit chaque entree
    Hasard           h{0x6F6E6563686573ULL};
#endif
    for (int i = 0; i < 64; ++i)
    {
        const Square s = Square(i);
        // Une case du bord ne cache rien derriere elle : elle ne compte pas
        // comme bloqueur, sauf sur la rangee ou la colonne de la piece.
        const Bitboard bords = ((Rank1BB | Rank8BB) & ~rank_bb(s)) | ((FileABB | FileHBB) & ~file_bb(s));
        Glisseur&      gl    = g[i];
        gl.masque            = lancer(rayons, s, 0) & ~bords;
        gl.table             = place;
        const int bits       = popcount(gl.masque);
        const int taille     = 1 << bits;

        // Tous les sous-ensembles du masque, et leurs vraies attaques
        occs.assign(taille, 0);
        vraies.assign(taille, 0);
        Bitboard sous = 0;
        for (int k = 0; k < taille; ++k)
        {
            occs[k]   = sous;
            vraies[k] = lancer(rayons, s, sous);
            sous      = (sous - gl.masque) & gl.masque;  // sous-ensemble suivant
        }

#if defined(USE_PEXT)
        for (int k = 0; k < taille; ++k)
            place[gl.index(occs[k])] = vraies[k];
#else
        // Cherche un multiplicateur sans collision nuisible : deux occupations
        // peuvent tomber sur la meme entree seulement si leurs attaques sont egales.
        gl.decalage = 64 - bits;
        tour.assign(taille, 0);
        for (int essai = 1;; ++essai)
        {
            do
                gl.multiplicateur = h.clairseme();
            while (popcount((gl.masque * gl.multiplicateur) & 0xFF00000000000000ULL) < 6);

            bool bon = true;
            for (int k = 0; k < taille && bon; ++k)
            {
                const unsigned j = gl.index(occs[k]);
                if (tour[j] != essai)
                {
                    tour[j]  = essai;
                    place[j] = vraies[k];
                }
                else
                    bon = place[j] == vraies[k];
            }
            if (bon)
                break;
        }
#endif
        place += taille;
    }
    return place;
}

// Destination d'un saut (cavalier ou roi), vide si elle sort de l'echiquier
Bitboard saut(Square s, int dc, int dr) {
    const int c = int(file_of(s)) + dc, g = int(rank_of(s)) + dr;
    return c >= 0 && c < 8 && g >= 0 && g < 8 ? square_bb(make_square(File(c), Rank(g))) : 0;
}

}  // namespace

void Bitboards::init() {
    for (int a = 0; a < 64; ++a)
        for (int b = 0; b < 64; ++b)
            Ecart[a][b] = std::uint8_t(std::max(distance<File>(Square(a), Square(b)),
                                                distance<Rank>(Square(a), Square(b))));

    Bitboard* libre = construire(Tours, RayonsTour, Reserve);
    construire(Fous, RayonsFou, libre);

    for (int i = 0; i < 64; ++i)
    {
        const Square s = Square(i);
        Pion[WHITE][s] = pawn_attacks_bb<WHITE>(square_bb(s));
        Pion[BLACK][s] = pawn_attacks_bb<BLACK>(square_bb(s));

        for (int dc = -2; dc <= 2; ++dc)
            for (int dr = -2; dr <= 2; ++dr)
            {
                const int n = std::abs(dc) + std::abs(dr);
                if (n == 3 && dc && dr)
                    Vide[KNIGHT][s] |= saut(s, dc, dr);
                if ((dc || dr) && std::abs(dc) <= 1 && std::abs(dr) <= 1)
                    Vide[KING][s] |= saut(s, dc, dr);
            }

        Vide[BISHOP][s] = lancer(RayonsFou, s, 0);
        Vide[ROOK][s]   = lancer(RayonsTour, s, 0);
        Vide[QUEEN][s]  = Vide[BISHOP][s] | Vide[ROOK][s];
    }

    // Lignes et segments : pour deux cases alignees, la ligne entiere est
    // l'intersection des attaques a vide des deux cases (plus elles-memes) ;
    // le segment, celle des attaques quand chacune bloque l'autre.
    for (int i = 0; i < 64; ++i)
        for (int j = 0; j < 64; ++j)
        {
            const Square a = Square(i), b = Square(j);
            Entre[a][b]    = square_bb(b);
            for (PieceType p : {BISHOP, ROOK})
                if (Vide[p][a] & b)
                {
                    Ligne[a][b] = (Vide[p][a] & Vide[p][b]) | a | b;
                    Entre[a][b] |= attacks_bb(p, a, square_bb(b)) & attacks_bb(p, b, square_bb(a));
                }
        }
}

std::string Bitboards::pretty(Bitboard b) {
    std::string t;
    for (int r = 7; r >= 0; --r)
    {
        t += char('1' + r);
        t += ' ';
        for (int c = 0; c < 8; ++c)
            t += (b & make_square(File(c), Rank(r))) ? " x" : " .";
        t += '\n';
    }
    t += "   a b c d e f g h\n";
    return t;
}

}  // namespace one
