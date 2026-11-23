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

// La position (reecrite pour Onechess le 15-11-2026).

#include "position.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <type_traits>

#include "movegen.h"
#include "nnue.h"
#include "prof.h"
#include "tune.h"

namespace one {

namespace {

// ---------------------------------------------------------------- hachage
// Une cle aleatoire par (piece, case), par colonne de prise en passant, par
// combinaison de droits de roque, et pour le trait.
struct Cles {
    Key piece[PIECE_NB][SQUARE_NB];
    Key ep[FILE_NB];
    Key roque[CASTLING_RIGHT_NB];
    Key noirs;
} Z;

// Coups reversibles (une piece autre qu'un pion qui va de a vers b, aller ou
// retour) ranges par leur cle : la difference de cle entre deux positions
// separees d'un seul tel coup. Adressage ouvert, sondage lineaire ; la table
// est remplie a moins de moitie (3 668 coups pour 8 192 places).
constexpr int TailleRev = 8192;
Key  RevCle[TailleRev];
Move RevCoup[TailleRev];

int place_rev(Key k) { return int((k ^ (k >> 29)) & (TailleRev - 1)); }

const Move* coup_reversible(Key k) {
    for (int i = place_rev(k);; i = (i + 1) & (TailleRev - 1))
    {
        if (!RevCoup[i])
            return nullptr;
        if (RevCle[i] == k)
            return &RevCoup[i];
    }
}

constexpr std::string_view Lettres = " PNBRQK  pnbrqk";

Square cote_roi(Color c, bool petit) { return relative_square(c, petit ? SQ_G1 : SQ_C1); }
Square cote_tour(Color c, bool petit) { return relative_square(c, petit ? SQ_F1 : SQ_D1); }

}  // namespace

// ================================================================ demarrage
void Position::init() {
    // Cles tirees d'une suite « splitmix » a graine fixe (memes cles a chaque
    // lancement, sur toutes les machines : la signature du banc en depend).
    std::uint64_t etat = 0x4F6E656368657373ULL;  // « Onechess »
    auto          tire = [&etat]() {
        std::uint64_t z = (etat += 0x9E3779B97F4A7C15ULL);
        z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    for (int p = 0; p < PIECE_NB; ++p)
        if (type_of(Piece(p)) != NO_PIECE_TYPE && type_of(Piece(p)) <= KING)
            for (int s = 0; s < 64; ++s)
                Z.piece[p][s] = tire();
    for (Key& k : Z.ep)
        k = tire();
    for (Key& k : Z.roque)
        k = tire();
    Z.noirs = tire();

    // Coups reversibles : pour chaque piece (sauf les pions) et chaque paire
    // de cases que cette piece relie sur echiquier vide.
    std::fill(std::begin(RevCoup), std::end(RevCoup), Move::none());
    int n = 0;
    for (Color c : {WHITE, BLACK})
        for (PieceType t : {KNIGHT, BISHOP, ROOK, QUEEN, KING})
            for (int a = 0; a < 64; ++a)
                for (int b = a + 1; b < 64; ++b)
                    if (empty_attacks_bb(t, Square(a)) & Square(b))
                    {
                        const Piece p = make_piece(c, t);
                        const Key   k = Z.piece[p][a] ^ Z.piece[p][b] ^ Z.noirs;
                        int         i = place_rev(k);
                        while (RevCoup[i])
                            i = (i + 1) & (TailleRev - 1);
                        RevCle[i]  = k;
                        RevCoup[i] = Move(Square(a), Square(b));
                        ++n;
                    }
    assert(n == 3668);
    (void) n;
}

// ================================================================ FEN
// Champs : placement, trait, roques (KQkq ou colonnes des tours en 960),
// prise en passant, regle des 50 coups, numero du coup.
Position& Position::set(const std::string& fen, bool chess960, StateInfo* etat) {
    static_assert(std::is_standard_layout_v<Position>);
    Thread* const garde = fil;
    std::memset(this, 0, offsetof(Position, reservoir));  // le reservoir garde son contenu
    std::memset(etat, 0, sizeof(StateInfo));
    fil = garde;
    st  = etat;
    for (Square& s : tourRoque)
        s = SQ_NONE;

    std::istringstream in(fen);
    std::string        plan, camp, roques, ep;
    in >> plan >> camp >> roques >> ep;

    // 1. Pieces, de a8 vers h1 ; tout ce qui sortirait de l'echiquier est ignore
    int r = 7, c = 0;
    for (char ch : plan)
    {
        if (ch == '/')
            --r, c = 0;
        else if (ch >= '1' && ch <= '8')
            c += ch - '0';
        else if (std::size_t i = Lettres.find(ch); i != std::string_view::npos && ch != ' ')
        {
            if (r >= 0 && r < 8 && c >= 0 && c < 8 && empty(make_square(File(c), Rank(r))))
                poser(Piece(i), make_square(File(c), Rank(r)));
            ++c;
        }
    }

    // 2. Trait
    trait = camp == "b" ? BLACK : WHITE;

    // 3. Roques : retenus seulement si roi et tour sont sur la premiere rangee
    for (char ch : roques)
    {
        if (ch == '-')
            break;
        const Color   co   = std::islower((unsigned char) ch) ? BLACK : WHITE;
        const char    u    = char(std::toupper((unsigned char) ch));
        const Rank    base = relative_rank(co, RANK_1);
        const Piece   tour = make_piece(co, ROOK);
        if (!(pieces(co, KING) & rank_bb(base)))
            continue;
        Square trouvee = SQ_NONE;
        if (u == 'K')
            for (int f = 7; f >= 0 && trouvee == SQ_NONE; --f)
            {
                if (piece_on(make_square(File(f), base)) == tour)
                    trouvee = make_square(File(f), base);
            }
        else if (u == 'Q')
            for (int f = 0; f < 8 && trouvee == SQ_NONE; ++f)
            {
                if (piece_on(make_square(File(f), base)) == tour)
                    trouvee = make_square(File(f), base);
            }
        else if (u >= 'A' && u <= 'H' && piece_on(make_square(File(u - 'A'), base)) == tour)
            trouvee = make_square(File(u - 'A'), base);
        if (trouvee != SQ_NONE)
            noter_roque(co, trouvee);
    }

    // 4. Prise en passant : gardee seulement si une prise legale existe (sinon
    //    deux positions identiques auraient des cles differentes).
    st->epSquare = SQ_NONE;
    if (ep.size() == 2 && ep[0] >= 'a' && ep[0] <= 'h' && ep[1] == (trait == WHITE ? '6' : '3'))
    {
        const Square e     = make_square(File(ep[0] - 'a'), Rank(ep[1] - '1'));
        const Square pion  = e + pawn_push(~trait);   // le pion qui vient d'avancer
        const Square avant = e + pawn_push(trait);    // sa case de depart
        if ((pawn_attacks_bb(~trait, e) & pieces(trait, PAWN)) && (pieces(~trait, PAWN) & pion)
            && !(pieces() & (square_bb(e) | square_bb(avant))) && en_passant_legal(trait, e))
            st->epSquare = e;
    }

    // 5-6. Compteurs ; demiCoups part de 0, les noirs jouent aux demi-coups impairs
    int numero = 1;
    in >> st->rule50 >> numero;
    demiCoups = std::max(2 * (numero - 1), 0) + (trait == BLACK);

    en960 = chess960;
    tout_recalculer();
    assert(pos_is_ok());
    return *this;
}

std::string Position::fen() const {
    std::ostringstream out;
    for (int r = 7; r >= 0; --r)
    {
        int vides = 0;
        for (int c = 0; c < 8; ++c)
        {
            const Piece p = piece_on(make_square(File(c), Rank(r)));
            if (p == NO_PIECE)
                ++vides;
            else
            {
                if (vides)
                    out << vides;
                vides = 0;
                out << Lettres[p];
            }
        }
        if (vides)
            out << vides;
        if (r)
            out << '/';
    }
    out << (trait == WHITE ? " w " : " b ");

    const struct { CastlingRights r; char normal; char base; } liste[4] = {
      {WHITE_OO, 'K', 'A'}, {WHITE_OOO, 'Q', 'A'}, {BLACK_OO, 'k', 'a'}, {BLACK_OOO, 'q', 'a'}};
    bool aucun = true;
    for (const auto& d : liste)
        if (can_castle(d.r))
        {
            out << (en960 ? char(d.base + file_of(tourRoque[d.r])) : d.normal);
            aucun = false;
        }
    if (aucun)
        out << '-';

    if (st->epSquare == SQ_NONE)
        out << " - ";
    else
        out << ' ' << char('a' + file_of(st->epSquare)) << char('1' + rank_of(st->epSquare)) << ' ';
    out << st->rule50 << ' ' << 1 + (demiCoups - (trait == BLACK)) / 2;
    return out.str();
}

std::ostream& operator<<(std::ostream& os, const Position& pos) {
    for (int r = 7; r >= 0; --r)
    {
        os << ' ' << char('1' + r) << " ";
        for (int c = 0; c < 8; ++c)
        {
            const Piece p = pos.piece_on(make_square(File(c), Rank(r)));
            os << ' ' << (p == NO_PIECE ? '.' : Lettres[p]);
        }
        os << '\n';
    }
    os << "    a b c d e f g h\n\nFen: " << pos.fen() << "\nCle: " << std::hex << std::uppercase
       << std::setfill('0') << std::setw(16) << pos.key() << std::setfill(' ') << std::dec << '\n';
    return os;
}

// ================================================================ etat derive
// Droit de roque du camp c avec la tour en « tour » : masques de perte des
// droits, et cases a trouver vides (entre le roi et sa case d'arrivee, entre
// la tour et la sienne, sans compter le roi et la tour eux-memes).
void Position::noter_roque(Color c, Square tour) {
    const Square         roi   = square<KING>(c);
    const bool           petit = roi < tour;
    const CastlingRights r     = c & (petit ? KING_SIDE : QUEEN_SIDE);

    st->castlingRights |= r;
    perteRoque[roi] |= r;
    perteRoque[tour] |= r;
    tourRoque[r]   = tour;
    cheminRoque[r] = (between_bb(roi, cote_roi(c, petit)) | between_bb(tour, cote_tour(c, petit)))
                   & ~(square_bb(roi) | square_bb(tour));
}

// Le camp c peut-il prendre en passant sur « ep » sans exposer son roi ?
bool Position::en_passant_legal(Color c, Square ep) const {
    const Square   victime = ep - pawn_push(c);
    const Square   roi     = square<KING>(c);
    const Bitboard eux     = pieces(~c) & ~square_bb(victime);
    for (Bitboard p = pawn_attacks_bb(~c, ep) & pieces(c, PAWN); p; )
    {
        const Square   de  = pop_lsb(p);
        const Bitboard occ = (pieces() ^ square_bb(de) ^ square_bb(victime)) | square_bb(ep);
        if (!(attackers_to(roi, occ) & eux))
            return true;
    }
    return false;
}

// Pieces qui couvrent « cible » contre les glisseurs de « tireurs » ; dans
// « cloueurs », ceux des tireurs dont la piece couverte est du camp de la cible.
Bitboard Position::bloqueurs(Bitboard tireurs, Square cible, Bitboard& cloueurs) const {
    Bitboard couvrants = 0;
    cloueurs           = 0;
    const Bitboard visent =
      tireurs & ((attacks_bb<ROOK>(cible) & pieces(ROOK, QUEEN)) | (attacks_bb<BISHOP>(cible) & pieces(BISHOP, QUEEN)));
    const Bitboard reste = pieces() & ~visent;
    const Bitboard amis  = pieces(color_of(piece_on(cible)));
    for (Bitboard v = visent; v; )
    {
        const Square   t      = pop_lsb(v);
        const Bitboard milieu = between_bb(cible, t) & reste;
        if (milieu && !more_than_one(milieu))
        {
            couvrants |= milieu;
            if (milieu & amis)
                cloueurs |= t;
        }
    }
    return couvrants;
}

void Position::calculer_echecs() {
    st->kingSq[WHITE] = square<KING>(WHITE);
    st->kingSq[BLACK] = square<KING>(BLACK);
    st->blockersForKing[WHITE] = bloqueurs(pieces(BLACK), st->kingSq[WHITE], st->pinners[BLACK]);
    st->blockersForKing[BLACK] = bloqueurs(pieces(WHITE), st->kingSq[BLACK], st->pinners[WHITE]);

    const Square   cible = st->kingSq[~trait];
    const Bitboard diag  = attacks_bb<BISHOP>(cible, pieces());
    const Bitboard droit = attacks_bb<ROOK>(cible, pieces());
    st->checkSquares[PAWN]   = pawn_attacks_bb(~trait, cible);
    st->checkSquares[KNIGHT] = attacks_bb<KNIGHT>(cible);
    st->checkSquares[BISHOP] = diag;
    st->checkSquares[ROOK]   = droit;
    st->checkSquares[QUEEN]  = diag | droit;
    st->checkSquares[KING]   = 0;
}

// Cles des corrections « mineures » et « majeures » (23-11, revue externe : elles etaient
// recalculees a chaque evaluation, ~8 % du temps) : le roi compte dans les deux.
inline bool est_mineure(Piece p) { const PieceType t = type_of(p); return t == KNIGHT || t == BISHOP || t == KING; }
inline bool est_majeure(Piece p) { const PieceType t = type_of(p); return t == ROOK || t == QUEEN || t == KING; }
inline void cles_pieces(StateInfo* s, Piece p, Key dk) {
    if (est_mineure(p))
        s->minorKey ^= dk;
    if (est_majeure(p))
        s->majorKey ^= dk;
}

// Tout ce qu'on maintient ensuite coup par coup, calcule ici de zero.
void Position::tout_recalculer() {
    st->key = st->pawnKey = 0;
    st->nonPawnKey[WHITE] = st->nonPawnKey[BLACK] = 0;
    st->minorKey = st->majorKey = 0;
    st->nonPawnMaterial[WHITE] = st->nonPawnMaterial[BLACK] = 0;
    st->checkersBB = attackers_to(square<KING>(trait)) & pieces(~trait);
    calculer_echecs();

    for (Bitboard b = pieces(); b; )
    {
        const Square s = pop_lsb(b);
        const Piece  p = piece_on(s);
        st->key ^= Z.piece[p][s];
        if (type_of(p) == PAWN)
            st->pawnKey ^= Z.piece[p][s];
        else
        {
            st->nonPawnKey[color_of(p)] ^= Z.piece[p][s];
            cles_pieces(st, p, Z.piece[p][s]);
            st->nonPawnMaterial[color_of(p)] += piece_value(p);
        }
    }
    if (st->epSquare != SQ_NONE)
        st->key ^= Z.ep[file_of(st->epSquare)];
    if (trait == BLACK)
        st->key ^= Z.noirs;
    st->key ^= Z.roque[st->castlingRights];
}

// ================================================================ attaques
Bitboard Position::attackers_to(Square s, Bitboard occ) const {
    const Bitboard dames = pieces(QUEEN);
    return (attacks_bb<ROOK>(s, occ) & (pieces(ROOK) | dames))
         | (attacks_bb<BISHOP>(s, occ) & (pieces(BISHOP) | dames))
         | (attacks_bb<KNIGHT>(s) & pieces(KNIGHT)) | (attacks_bb<KING>(s) & pieces(KING))
         | (pawn_attacks_bb(WHITE, s) & pieces(BLACK, PAWN)) | (pawn_attacks_bb(BLACK, s) & pieces(WHITE, PAWN));
}

// ================================================================ legalite
// Un coup pseudo-legal laisse-t-il notre roi hors d'atteinte ?
bool Position::legal(Move m) const {
    PROF_SCOPE(LEGAL);
    const Color  moi = trait;
    const Square de  = m.from_sq();
    const Square roi = square<KING>(moi);

    switch (m.type_of())
    {
    case CASTLING : {
        // Le roi ne traverse aucune case attaquee (depart et arrivee compris)
        const bool   petit   = m.to_sq() > de;
        const Square arrivee = cote_roi(moi, petit);
        const int    pas     = arrivee > de ? 1 : arrivee < de ? -1 : 0;
        for (Square s = de;; s = Square(int(s) + pas))
        {
            if (attackers_to(s) & pieces(~moi))
                return false;
            if (s == arrivee)
                break;
        }
        // En 960, la tour du roque pouvait masquer un clouage du roi
        return !en960 || !(blockers_for_king(moi) & m.to_sq());
    }
    case EN_PASSANT : {
        // Deux pieces quittent la rangee : on rejoue l'occupation
        const Square   a   = m.to_sq();
        const Bitboard occ = (pieces() ^ square_bb(de) ^ square_bb(a - pawn_push(moi))) | square_bb(a);
        return !(attacks_bb<ROOK>(roi, occ) & pieces(~moi, ROOK, QUEEN))
            && !(attacks_bb<BISHOP>(roi, occ) & pieces(~moi, BISHOP, QUEEN));
    }
    default :
        if (de == roi)  // le roi ne doit pas se cacher derriere lui-meme
            return !(attackers_to(m.to_sq(), pieces() ^ square_bb(de)) & pieces(~moi));
        // une piece clouee reste sur la ligne de son clouage
        return !(blockers_for_king(moi) & de) || aligned(de, m.to_sq(), roi);
    }
}

// Un coup venu d'ailleurs (table, coups meurtriers...) est-il jouable ici ?
// Doit refuser n'importe quelle valeur incoherente.
bool Position::pseudo_legal(Move m) const {
    const Color  moi = trait;
    const Square de = m.from_sq(), a = m.to_sq();
    const Piece  p  = piece_on(de);

    if (m.type_of() != NORMAL)  // rares : on demande au generateur
        return checkers() ? MoveList<EVASIONS>(*this).contains(m) : MoveList<NON_EVASIONS>(*this).contains(m);
    if (m.promotion_type() != KNIGHT)  // champ de promotion non nul sur un coup normal
        return false;
    if (p == NO_PIECE || color_of(p) != moi || (pieces(moi) & a))
        return false;

    if (type_of(p) == PAWN)
    {
        if ((Rank1BB | Rank8BB) & a)  // il faudrait une promotion
            return false;
        const Square un   = de + pawn_push(moi);
        const bool   prise = (pawn_attacks_bb(moi, de) & pieces(~moi) & a) != 0;
        const bool   pas1  = a == un && empty(a);
        const bool   pas2  = a == un + pawn_push(moi) && relative_rank(moi, de) == RANK_2 && empty(un) && empty(a);
        if (!prise && !pas1 && !pas2)
            return false;
    }
    else if (!(attacks_bb(type_of(p), de, pieces()) & a))
        return false;

    if (checkers())
    {
        if (type_of(p) == KING)
            return !(attackers_to(a, pieces() ^ square_bb(de)) & pieces(~moi));
        if (more_than_one(checkers()))  // double echec : le roi doit bouger
            return false;
        return (between_bb(square<KING>(moi), lsb(checkers())) & a) != 0;  // s'interposer ou prendre
    }
    return true;
}

bool Position::gives_check(Move m) const {
    PROF_SCOPE(GIVESCHECK);
    const Square de = m.from_sq(), a = m.to_sq();
    const Square roi = square<KING>(~trait);

    if (check_squares(type_of(piece_on(de))) & a)  // echec direct
        return true;
    if (blockers_for_king(~trait) & de)            // echec a la decouverte
        return m.type_of() == CASTLING || !aligned(de, a, roi);

    switch (m.type_of())
    {
    case PROMOTION :
        return (attacks_bb(m.promotion_type(), a, pieces() ^ square_bb(de)) & roi) != 0;
    case EN_PASSANT : {
        const Bitboard occ =
          (pieces() ^ square_bb(de) ^ square_bb(make_square(file_of(a), rank_of(de)))) | square_bb(a);
        return (attacks_bb<ROOK>(roi, occ) & pieces(trait, ROOK, QUEEN))
            || (attacks_bb<BISHOP>(roi, occ) & pieces(trait, BISHOP, QUEEN));
    }
    case CASTLING :  // seule la tour peut donner echec
        return (check_squares(ROOK) & cote_tour(trait, a > de)) != 0;
    default :
        return false;
    }
}

// ================================================================ jouer
// Releve, avant de toucher au plateau, ce qui bouge (pour le reseau)
void Position::noter_deplacements(Move m, Piece p, Piece prise, StateInfo& e) const {
    DirtyPiece&  d  = e.dirtyPiece;
    const Square de = m.from_sq(), a = m.to_sq();
    d.count         = 0;
    auto ajoute     = [&d](Piece q, Square x, Square y) {
        d.piece[d.count] = q, d.from[d.count] = x, d.to[d.count] = y;
        ++d.count;
    };

    if (m.type_of() == CASTLING)  // « a » est la case de la tour
    {
        const bool petit = a > de;
        ajoute(p, de, cote_roi(trait, petit));
        ajoute(make_piece(trait, ROOK), a, cote_tour(trait, petit));
        return;
    }
    if (prise)
        ajoute(prise, m.type_of() == EN_PASSANT ? a - pawn_push(trait) : a, SQ_NONE);
    if (m.type_of() == PROMOTION)
    {
        ajoute(p, de, SQ_NONE);
        ajoute(make_piece(trait, m.promotion_type()), SQ_NONE, a);
    }
    else
        ajoute(p, de, a);
}

// Roque dans un sens ou dans l'autre. On enleve roi et tour avant de les
// reposer : en 960 l'un peut arriver sur la case de depart de l'autre.
template<bool Jouer>
void Position::roquer(Color c, Square roi, Square& arriveeRoi, Square& departTour, Square& arriveeTour) {
    const bool petit = arriveeRoi > roi;
    departTour       = arriveeRoi;
    arriveeTour      = cote_tour(c, petit);
    arriveeRoi       = cote_roi(c, petit);
    oter(Jouer ? roi : arriveeRoi);
    oter(Jouer ? departTour : arriveeTour);
    poser(make_piece(c, KING), Jouer ? arriveeRoi : roi);
    poser(make_piece(c, ROOK), Jouer ? arriveeTour : departTour);
}

void Position::do_move(Move m, StateInfo& nouveau, bool donneEchec) {
    PROF_SCOPE(DOMOVE);
    assert(m.is_ok() && &nouveau != st);

    std::memcpy(&nouveau, st, offsetof(StateInfo, epSquare));
    nouveau.previous = st;
    st               = &nouveau;
    StateInfo& av    = *st->previous;

    const Color  moi = trait, lui = ~trait;
    const Square de  = m.from_sq();
    Square       a   = m.to_sq();
    const Piece  p   = piece_on(de);
    Piece        pris = m.type_of() == EN_PASSANT ? make_piece(lui, PAWN) : piece_on(a);
    Key          k    = av.key ^ Z.noirs;

    ++demiCoups;
    ++st->rule50;
    ++st->pliesFromNull;

    noter_deplacements(m, p, pris, *st);
    st->accGen = st->thrGen = 0;
    // Place suivante du reservoir (-1 : aucune, l'evaluation repartira de zero)
    st->accIndex = av.accIndex >= 0 && av.accIndex + 1 < NNUE::ACC_POOL ? av.accIndex + 1 : -1;

    if (m.type_of() == CASTLING)
    {
        Square td, ta;
        roquer<true>(moi, de, a, td, ta);
        const Key dk = Z.piece[pris][td] ^ Z.piece[pris][ta];
        k ^= dk;
        st->nonPawnKey[moi] ^= dk;
        cles_pieces(st, pris, dk);
        pris = NO_PIECE;
    }
    else if (pris)
    {
        const Square ou = m.type_of() == EN_PASSANT ? a - pawn_push(moi) : a;
        const Key    dk = Z.piece[pris][ou];
        if (type_of(pris) == PAWN)
            st->pawnKey ^= dk;
        else
        {
            st->nonPawnMaterial[lui] -= piece_value(pris);
            st->nonPawnKey[lui] ^= dk;
            cles_pieces(st, pris, dk);
        }
        oter(ou);
        k ^= dk;
        st->rule50 = 0;
    }

    const Key dp = Z.piece[p][de] ^ Z.piece[p][a];
    k ^= dp;
    if (type_of(p) != PAWN)
    {
        st->nonPawnKey[moi] ^= dp;
        cles_pieces(st, p, dp);
    }

    if (av.epSquare != SQ_NONE)
        k ^= Z.ep[file_of(av.epSquare)];
    st->epSquare = SQ_NONE;

    if (const int perdus = perteRoque[de] | perteRoque[a]; st->castlingRights && perdus)
    {
        k ^= Z.roque[st->castlingRights];
        st->castlingRights &= ~perdus;
        k ^= Z.roque[st->castlingRights];
    }

    if (m.type_of() != CASTLING)
        deplacer(de, a);

    if (type_of(p) == PAWN)
    {
        if ((int(a) ^ int(de)) == 16)  // double pas : prise en passant possible ?
        {
            const Square ep = a - pawn_push(moi);
            if ((pawn_attacks_bb(moi, ep) & pieces(lui, PAWN)) && en_passant_legal(lui, ep))
            {
                st->epSquare = ep;
                k ^= Z.ep[file_of(ep)];
            }
        }
        else if (m.type_of() == PROMOTION)
        {
            const Piece neuve = make_piece(moi, m.promotion_type());
            oter(a);
            poser(neuve, a);
            k ^= Z.piece[p][a] ^ Z.piece[neuve][a];
            st->pawnKey ^= Z.piece[p][a];
            st->nonPawnMaterial[moi] += piece_value(neuve);
            st->nonPawnKey[moi] ^= Z.piece[neuve][a];
            cles_pieces(st, neuve, Z.piece[neuve][a]);
        }
        st->pawnKey ^= Z.piece[p][de] ^ Z.piece[p][a];
        st->rule50 = 0;
    }

    st->capturedPiece = pris;
    st->key           = k;
    st->checkersBB    = donneEchec ? attackers_to(square<KING>(lui)) & pieces(moi) : Bitboard(0);
    trait             = lui;
    calculer_echecs();

    // Repetition : meme trait, donc de deux en deux, sans remonter au-dela de
    // la derniere prise, du dernier pas de pion ou du dernier coup nul.
    st->repetition    = 0;
    const int jusqua  = std::min(st->rule50, st->pliesFromNull);
    const StateInfo* e = st->previous->previous;
    for (int d = 4; d <= jusqua; d += 2)
    {
        e = e->previous->previous;
        if (e->key == st->key)
        {
            st->repetition = e->repetition ? -d : d;
            break;
        }
    }

    if (NNUE::FullThreatsOn)
        NNUE::ft_on_move(*this);
    assert(pos_is_ok());
}

void Position::undo_move(Move m) {
    PROF_SCOPE(UNDOMOVE);
    trait            = ~trait;
    const Color  moi = trait;
    const Square de  = m.from_sq();
    Square       a   = m.to_sq();

    if (m.type_of() == CASTLING)
    {
        Square td, ta;
        roquer<false>(moi, de, a, td, ta);
    }
    else
    {
        if (m.type_of() == PROMOTION)  // la piece promue redevient un pion
        {
            oter(a);
            poser(make_piece(moi, PAWN), a);
        }
        deplacer(a, de);
        if (st->capturedPiece)
            poser(st->capturedPiece, m.type_of() == EN_PASSANT ? a - pawn_push(moi) : a);
    }
    st = st->previous;
    --demiCoups;
    assert(pos_is_ok());
}

void Position::do_null_move(StateInfo& nouveau) {
    assert(!checkers() && &nouveau != st);
    // Pas l'accumulateur : il sera repris tel quel (rien n'a bouge)
    std::memcpy(&nouveau, st, offsetof(StateInfo, dirtyPiece));
    nouveau.previous         = st;
    nouveau.dirtyPiece.count = 0;
    nouveau.accGen = nouveau.thrGen = 0;
    nouveau.accIndex = st->accIndex >= 0 && st->accIndex + 1 < NNUE::ACC_POOL ? st->accIndex + 1 : -1;
    st               = &nouveau;

    if (st->epSquare != SQ_NONE)
    {
        st->key ^= Z.ep[file_of(st->epSquare)];
        st->epSquare = SQ_NONE;
    }
    st->key ^= Z.noirs;
    ++st->rule50;
    st->pliesFromNull = 0;
    trait             = ~trait;
    calculer_echecs();
    st->repetition = 0;

    if (NNUE::FullThreatsOn)
        NNUE::ft_on_move(*this);
    assert(pos_is_ok());
}

void Position::undo_null_move() {
    st    = st->previous;
    trait = ~trait;
}

// Cle approchee apres m (pour precharger la table) : piece deplacee, piece
// prise et trait ; les cas speciaux ne sont pas traites.
Key Position::key_after_null() const {
    return st->key ^ Z.noirs ^ (st->epSquare != SQ_NONE ? Z.ep[file_of(st->epSquare)] : 0);
}

Key Position::key_after(Move m) const {
    PROF_SCOPE(KEYAFTER);
    const Square de = m.from_sq(), a = m.to_sq();
    const Piece  p  = piece_on(de);
    Key          k  = st->key ^ Z.noirs ^ Z.piece[p][de] ^ Z.piece[p][a];
    if (const Piece q = piece_on(a))
        k ^= Z.piece[q][a];
    return k;
}

// ================================================================ echange statique
// On rejoue les prises sur la case d'arrivee, chaque camp avec sa piece la
// moins chere, en tenant compte des rayons X et des pieces clouees. « gain »
// est ce que le camp qui vient de prendre a d'avance sur le seuil ; on
// s'arrete des qu'un camp n'a plus interet a continuer.
bool Position::see_ge(Move m, int seuil) const {
    PROF_SCOPE(SEE);
    if (m.type_of() != NORMAL)  // roques, promotions, en passant : reputes neutres
        return seuil <= 0;

    const Square de = m.from_sq(), a = m.to_sq();
    int          gain = piece_value(piece_on(a)) - seuil;
    if (gain < 0)  // meme sans reprise, on n'atteint pas le seuil
        return false;
    gain = piece_value(piece_on(de)) - gain;
    if (gain <= 0)  // meme en perdant la piece, on reste au-dessus
        return true;

    Bitboard occ     = pieces() ^ square_bb(de) ^ square_bb(a);
    Bitboard tireurs = attackers_to(a, occ);
    Color    camp    = trait;
    bool     ok      = true;  // le resultat si l'echange s'arretait la

    const Bitboard diag  = pieces(BISHOP, QUEEN);
    const Bitboard droit = pieces(ROOK, QUEEN);

    for (;;)
    {
        camp = ~camp;
        tireurs &= occ;
        Bitboard miens = tireurs & pieces(camp);
        if (miens && (pinners(~camp) & occ))  // un clouage tient encore
            miens &= ~blockers_for_king(camp);
        if (!miens)
            break;
        ok = !ok;

        // La piece la moins chere de ce camp
        PieceType t = PAWN;
        while (!(miens & pieces(t)))
            ++t;
        if (t == KING)  // le roi ne prend que si plus rien ne peut le reprendre
            return (tireurs & ~pieces(camp)) ? !ok : ok;

        if ((gain = piece_value(make_piece(WHITE, t)) - gain) < int(ok))
            break;
        occ ^= least_significant_square_bb(miens & pieces(t));
        if (t == PAWN || t == BISHOP || t == QUEEN)
            tireurs |= attacks_bb<BISHOP>(a, occ) & diag;
        if (t == ROOK || t == QUEEN)
            tireurs |= attacks_bb<ROOK>(a, occ) & droit;
    }
    return ok;
}

// ================================================================ nulles
bool Position::is_draw(int ply) const {
    PROF_SCOPE(DRAW);
    // 50 coups, sauf si c'est un mat
    if (st->rule50 > 99 && (!checkers() || MoveList<LEGAL>(*this).size()))
        return true;
    // dans l'arbre, une seule repetition suffit
    return st->repetition && st->repetition < ply;
}

// Le camp au trait a-t-il un coup qui ramene une position deja vue ? On
// compare la cle actuelle a celles de la partie, de deux en deux ; si l'ecart
// est exactement la cle d'un coup reversible dont le chemin est libre, oui.
bool Position::upcoming_repetition(int ply) const {
    const int jusqua = std::min(st->rule50, st->pliesFromNull);
    if (jusqua < 3)
        return false;

    const Key        ici  = st->key;
    const StateInfo* e    = st->previous;
    Key              coups = ici ^ e->key ^ Z.noirs;  // somme des coups adverses intermediaires

    for (int d = 3; d <= jusqua; d += 2)
    {
        coups ^= e->previous->key ^ e->previous->previous->key ^ Z.noirs;
        e = e->previous->previous;
        if (coups)  // les coups de l'adversaire ne se sont pas annules
            continue;
        const Move* r = coup_reversible(ici ^ e->key);
        if (!r)
            continue;
        if ((between_bb(r->from_sq(), r->to_sq()) ^ r->to_sq()) & pieces())
            continue;  // chemin occupe
        if (ply > d || e->repetition)
            return true;
    }
    return false;
}

// ================================================================ controle
bool Position::pos_is_ok() const {
#ifdef NDEBUG
    return true;
#else
    if (count<KING>(WHITE) != 1 || count<KING>(BLACK) != 1)
        return false;
    if (attackers_to(square<KING>(~trait)) & pieces(trait))  // le roi adverse en prise
        return false;
    if ((pieces(PAWN) & (Rank1BB | Rank8BB)) || (parCamp[WHITE] & parCamp[BLACK])
        || (parCamp[WHITE] | parCamp[BLACK]) != pieces())
        return false;
    if (st->epSquare != SQ_NONE && relative_rank(trait, st->epSquare) != RANK_6)
        return false;
    for (int t1 = PAWN; t1 <= KING; ++t1)
        for (int t2 = t1 + 1; t2 <= KING; ++t2)
            if (parType[t1] & parType[t2])
                return false;
    for (int s = 0; s < 64; ++s)
        if (cases[s] != NO_PIECE && !(pieces(color_of(cases[s]), type_of(cases[s])) & Square(s)))
            return false;
    for (Color c : {WHITE, BLACK})
        for (PieceType t : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING})
            if (nombre[make_piece(c, t)] != popcount(pieces(c, t)))
                return false;
    // Cles hors pions recalculees (une erreur ici fausserait en silence les corrections)
    Key np[COLOR_NB] = {0, 0};
    StateInfo cl{};
    for (Bitboard b = pieces() & ~pieces(PAWN); b; )
    {
        const Square s = pop_lsb(b);
        np[color_of(piece_on(s))] ^= Z.piece[piece_on(s)][s];
        cles_pieces(&cl, piece_on(s), Z.piece[piece_on(s)][s]);
    }
    return np[WHITE] == st->nonPawnKey[WHITE] && np[BLACK] == st->nonPawnKey[BLACK]
        && cl.minorKey == st->minorKey && cl.majorKey == st->majorKey;
#endif
}

template void Position::roquer<true>(Color, Square, Square&, Square&, Square&);
template void Position::roquer<false>(Color, Square, Square&, Square&, Square&);

}  // namespace one
