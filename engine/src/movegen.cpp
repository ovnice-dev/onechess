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

// Generation des coups pseudo-legaux (reecrite pour Onechess le 15-11-2026).
//
// Ordre d'emission, fixe (le tri des coups departage les egalites par cet
// ordre, et la signature du banc en depend) :
//   pions : poussees simples, doubles ; promotions (prises sur la 1re
//           diagonale, sur la 2e, puis poussees) ; prises simples sur la 1re
//           diagonale, sur la 2e ; prise en passant ;
//   puis cavaliers, fous, tours, dames (case de depart puis d'arrivee, du
//   bit le plus faible au plus fort), le roi, et les deux roques.
//
// Partage des promotions entre les deux etages (voir Position::capture_stage) :
// CAPTURES recoit toute promotion en dame et toute promotion par prise,
// QUIETS les sous-promotions par poussee ; EVASIONS et NON_EVASIONS tout.

#include "movegen.h"

#include <cassert>
#include <initializer_list>

#include "bitboard.h"
#include "position.h"

namespace one {

namespace {

// Geometrie des pions d'un camp : un pas en avant, et les deux diagonales de
// prise. « Diag1 » est la diagonale traitee en premier (vers la colonne h pour
// les blancs, vers la colonne a pour les noirs).
template<Color C>
struct Geo {
    static constexpr int      Pas     = C == WHITE ? 8 : -8;
    static constexpr int      Diag1   = C == WHITE ? 9 : -9;
    static constexpr int      Diag2   = C == WHITE ? 7 : -7;
    static constexpr Bitboard Avant7  = C == WHITE ? Rank7BB : Rank2BB;  // pions qui vont promouvoir
    static constexpr Bitboard Rangee3 = C == WHITE ? Rank3BB : Rank6BB;  // apres un premier pas

    static constexpr Bitboard decale(Bitboard b, int d) { return d > 0 ? b << d : b >> -d; }

    static constexpr Bitboard avance(Bitboard b) { return decale(b, Pas); }
    // Les masques de colonne evitent qu'une prise ne « passe » d'un bord a l'autre
    static constexpr Bitboard prend1(Bitboard b) {
        return decale(b & (C == WHITE ? ~FileHBB : ~FileABB), Diag1);
    }
    static constexpr Bitboard prend2(Bitboard b) {
        return decale(b & (C == WHITE ? ~FileABB : ~FileHBB), Diag2);
    }
};

inline Square depart(Square arrivee, int delta) { return Square(int(arrivee) - delta); }

// Vide un ensemble de cases d'arrivee en coups simples depuis « arrivee - delta »
inline ExtMove* vider(Bitboard cibles, int delta, ExtMove* sortie) {
    for (; cibles; )
    {
        const Square a = pop_lsb(cibles);
        *sortie++      = Move(depart(a, delta), a);
    }
    return sortie;
}

// Promotions d'un pion : la dame d'abord, puis tour, fou, cavalier quand
// l'etage les accepte.
template<GenType T>
ExtMove* promouvoir(Square de, Square a, bool prise, ExtMove* sortie) {
    constexpr bool tout = T == EVASIONS || T == NON_EVASIONS;
    if (T != QUIETS)
        *sortie++ = Move::make<PROMOTION>(de, a, QUEEN);
    if (tout || (T == CAPTURES && prise) || (T == QUIETS && !prise))
        for (PieceType p : {ROOK, BISHOP, KNIGHT})
            *sortie++ = Move::make<PROMOTION>(de, a, p);
    return sortie;
}

template<Color C, GenType T>
ExtMove* coups_de_pions(const Position& pos, Bitboard bloque, ExtMove* sortie) {
    using G = Geo<C>;
    const Bitboard libres  = ~pos.pieces();
    const Bitboard proies  = T == EVASIONS ? pos.checkers() : pos.pieces(~C);
    const Bitboard pions   = pos.pieces(C, PAWN);
    const Bitboard promus  = pions & G::Avant7;
    const Bitboard simples = pions & ~G::Avant7;

    if (T != CAPTURES)
    {
        Bitboard un   = G::avance(simples) & libres;
        Bitboard deux = G::avance(un & G::Rangee3) & libres;
        if (T == EVASIONS)  // en echec, une poussee ne sert qu'a s'interposer
            un &= bloque, deux &= bloque;
        sortie = vider(un, G::Pas, sortie);
        sortie = vider(deux, 2 * G::Pas, sortie);
    }

    if (promus)
    {
        if (T != QUIETS)
        {
            for (Bitboard b = G::prend1(promus) & proies; b; )
            {
                const Square a = pop_lsb(b);
                sortie         = promouvoir<T>(depart(a, G::Diag1), a, true, sortie);
            }
            for (Bitboard b = G::prend2(promus) & proies; b; )
            {
                const Square a = pop_lsb(b);
                sortie         = promouvoir<T>(depart(a, G::Diag2), a, true, sortie);
            }
        }
        Bitboard pousse = G::avance(promus) & libres;
        if (T == EVASIONS)
            pousse &= bloque;
        while (pousse)
        {
            const Square a = pop_lsb(pousse);
            sortie         = promouvoir<T>(depart(a, G::Pas), a, false, sortie);
        }
    }

    if (T != QUIETS)
    {
        sortie = vider(G::prend1(simples) & proies, G::Diag1, sortie);
        sortie = vider(G::prend2(simples) & proies, G::Diag2, sortie);

        // En passant : sans filtre en parade d'echec, la legalite est verifiee
        // ensuite par Position::legal() (un filtre approche perdrait des coups).
        const Square ep = pos.state()->epSquare;
        if (ep != SQ_NONE)
        {
            Bitboard preneurs = simples & pawn_attacks_bb(~C, ep);
            assert(preneurs);  // la case n'est posee que si une prise existe
            while (preneurs)
                *sortie++ = Move::make<EN_PASSANT>(pop_lsb(preneurs), ep);
        }
    }
    return sortie;
}

template<Color C, PieceType P>
ExtMove* coups_de_pieces(const Position& pos, Bitboard cibles, ExtMove* sortie) {
    const Bitboard occ = pos.pieces();
    for (Bitboard b = pos.pieces(C, P); b; )
    {
        const Square de = pop_lsb(b);
        for (Bitboard v = attacks_bb<P>(de, occ) & cibles; v; )
            *sortie++ = Move(de, pop_lsb(v));
    }
    return sortie;
}

template<Color C, GenType T>
ExtMove* tous(const Position& pos, ExtMove* sortie) {
    const Square roi     = pos.square<KING>(C);
    const bool   double_ = T == EVASIONS && more_than_one(pos.checkers());
    Bitboard     cibles  = 0;

    if (!double_)  // en double echec, seul le roi peut jouer
    {
        switch (T)
        {
        case EVASIONS : cibles = between_bb(roi, lsb(pos.checkers())); break;
        case NON_EVASIONS : cibles = ~pos.pieces(C); break;
        case CAPTURES : cibles = pos.pieces(~C); break;
        default : cibles = ~pos.pieces(); break;
        }
        sortie = coups_de_pions<C, T>(pos, cibles, sortie);
        sortie = coups_de_pieces<C, KNIGHT>(pos, cibles, sortie);
        sortie = coups_de_pieces<C, BISHOP>(pos, cibles, sortie);
        sortie = coups_de_pieces<C, ROOK>(pos, cibles, sortie);
        sortie = coups_de_pieces<C, QUEEN>(pos, cibles, sortie);
    }

    const Bitboard pas_roi = T == EVASIONS ? ~pos.pieces(C) : cibles;
    for (Bitboard v = attacks_bb<KING>(roi) & pas_roi; v; )
        *sortie++ = Move(roi, pop_lsb(v));

    if (T == QUIETS || T == NON_EVASIONS)
        for (CastlingRights r : {C & KING_SIDE, C & QUEEN_SIDE})
            if (pos.can_castle(r) && !pos.castling_impeded(r))
                *sortie++ = Move::make<CASTLING>(roi, pos.castling_rook_square(r));
    return sortie;
}

}  // namespace

template<GenType T>
ExtMove* generate(const Position& pos, ExtMove* sortie) {
    static_assert(T != LEGAL, "LEGAL : voir la specialisation plus bas");
    assert((T == EVASIONS) == bool(pos.checkers()));
    return pos.side_to_move() == WHITE ? tous<WHITE, T>(pos, sortie) : tous<BLACK, T>(pos, sortie);
}

template ExtMove* generate<CAPTURES>(const Position&, ExtMove*);
template ExtMove* generate<QUIETS>(const Position&, ExtMove*);
template ExtMove* generate<EVASIONS>(const Position&, ExtMove*);
template ExtMove* generate<NON_EVASIONS>(const Position&, ExtMove*);

// Coups legaux : on genere, puis on ne passe a Position::legal() que les trois
// cas douteux (piece clouee, roi, en passant). Un coup refuse est remplace par
// le dernier de la liste (l'ordre qui en resulte compte pour la racine).
template<>
ExtMove* generate<LEGAL>(const Position& pos, ExtMove* sortie) {
    const Color    moi    = pos.side_to_move();
    const Bitboard cloues = pos.blockers_for_king(moi) & pos.pieces(moi);
    const Square   roi    = pos.square<KING>(moi);

    ExtMove* fin = pos.checkers() ? generate<EVASIONS>(pos, sortie) : generate<NON_EVASIONS>(pos, sortie);
    for (ExtMove* m = sortie; m < fin; )
    {
        const bool douteux = (cloues & m->from_sq()) || m->from_sq() == roi || m->type_of() == EN_PASSANT;
        if (douteux && !pos.legal(*m))
            *m = *--fin;
        else
            ++m;
    }
    return fin;
}

}  // namespace one
