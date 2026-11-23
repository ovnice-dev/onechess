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

// La position : plateau, trait, droits, cles de hachage, et les coups qu'on
// y joue puis defait (reecrite pour Onechess le 15-11-2026).

#pragma once

#include <cassert>
#include <deque>
#include <iosfwd>
#include <memory>
#include <string>

#include "bitboard.h"
#include "nnue_types.h"
#include "types.h"

namespace one {

class Thread;

// Valeurs des pieces, partagees par la recherche, l'echange statique et
// l'evaluation (une seule table pour tout le moteur).
constexpr Value PawnValue   = 100;
constexpr Value KnightValue = 320;
constexpr Value BishopValue = 330;
constexpr Value RookValue   = 500;
constexpr Value QueenValue  = 950;

namespace Valeurs {
constexpr Value parType[8] = {0, PawnValue, KnightValue, BishopValue, RookValue, QueenValue, 0, 0};
}
// Valeur d'une piece, quelle que soit sa couleur (le roi vaut 0)
constexpr Value piece_value(Piece p) { return Valeurs::parType[int(p) & 7]; }
struct TableValeurs {
    constexpr Value operator[](Piece p) const { return piece_value(p); }
    constexpr Value operator[](int p) const { return piece_value(Piece(p)); }
};
constexpr TableValeurs PieceValue{};

// Ce qu'on empile a chaque coup. La premiere partie se recopie telle quelle
// d'un coup au suivant ; la seconde est recalculee ; la troisieme sert au reseau.
struct StateInfo {
    // --- recopie par do_move (jusqu'a epSquare exclu)
    Key   pawnKey;
    Key   nonPawnKey[COLOR_NB];        // pieces hors pions d'un camp, roi compris
    Key   minorKey, majorKey;          // (23-11) cavaliers + fous + rois ; tours + dames + rois
    Value nonPawnMaterial[COLOR_NB];   // materiel hors pions et roi
    int   castlingRights;
    int   rule50;                      // demi-coups depuis une prise ou un pas de pion
    int   pliesFromNull;               // demi-coups depuis le dernier coup nul

    // --- recalcule a chaque coup
    Square     epSquare;               // case de prise en passant, si une prise LEGALE existe
    Key        key;
    Bitboard   checkersBB;             // pieces qui donnent echec au camp au trait
    StateInfo* previous;
    Bitboard   blockersForKing[COLOR_NB];  // pieces (des deux camps) qui couvrent le roi
    Bitboard   pinners[COLOR_NB];          // glisseurs qui clouent une piece du camp oppose
    Bitboard   checkSquares[PIECE_TYPE_NB];// d'ou chaque type de piece donnerait echec
    Piece      capturedPiece;
    int        repetition;             // ecart avec la meme position (negatif si deja repetee)
    Square     kingSq[COLOR_NB];       // les deux rois, pour le reseau

    // --- reseau (non recopie par do_null_move)
    DirtyPiece    dirtyPiece;          // pieces deplacees par le coup
    std::uint32_t accGen;              // generation du reseau qui a rempli l'accumulateur
    std::uint32_t thrGen;              // idem pour l'accumulateur des menaces completes
    int           accIndex;            // place dans le reservoir d'accumulateurs, -1 sinon
};

// Historique de la partie reelle (garde vivant entre deux recherches pour
// reconnaitre les repetitions).
// Partage entre l'interface et la recherche (22-11) : un « bench » remplacait l'historique de la
// recherche et detruisait celui sur lequel pointait encore la position de l'interface.
using StateListPtr = std::shared_ptr<std::deque<StateInfo>>;

class Position {
   public:
    static void init();  // cles de hachage et table des coups reversibles

    Position()                           = default;
    Position(const Position&)            = delete;
    Position& operator=(const Position&) = delete;

    Position&   set(const std::string& fen, bool chess960, StateInfo* etat);
    std::string fen() const;
    void        set_thread(Thread* t) { fil = t; }
    Thread*     this_thread() const { return fil; }

    // Accumulateurs du reseau, ranges par profondeur depuis la racine
    NNUE::Accumulator& accumulator(int i) const { return reservoir[i]; }

    // --- contenu
    Piece    piece_on(Square s) const { return cases[s]; }
    bool     empty(Square s) const { return cases[s] == NO_PIECE; }
    Bitboard pieces() const { return parType[0]; }
    Bitboard pieces(PieceType t) const { return parType[t]; }
    Bitboard pieces(Color c) const { return parCamp[c]; }
    template<typename... T>
    Bitboard pieces(PieceType t, T... autres) const { return (parType[t] | ... | parType[autres]); }
    template<typename... T>
    Bitboard pieces(Color c, T... types) const { return parCamp[c] & pieces(types...); }

    int count(PieceType t, Color c) const { return nombre[make_piece(c, t)]; }
    template<PieceType T>
    int count(Color c) const { return nombre[make_piece(c, T)]; }
    template<PieceType T>
    int count() const { return nombre[make_piece(WHITE, T)] + nombre[make_piece(BLACK, T)]; }
    template<PieceType T>
    Square square(Color c) const { return lsb(pieces(c, T)); }  // la piece unique de ce type

    // --- roques
    bool   can_castle(CastlingRights r) const { return (st->castlingRights & r) != 0; }
    bool   castling_impeded(CastlingRights r) const { return (pieces() & cheminRoque[r]) != 0; }
    Square castling_rook_square(CastlingRights r) const { return tourRoque[r]; }
    bool   is_chess960() const { return en960; }

    // --- echecs, clouages, attaques
    Bitboard checkers() const { return st->checkersBB; }
    Bitboard blockers_for_king(Color c) const { return st->blockersForKing[c]; }
    Bitboard pinners(Color c) const { return st->pinners[c]; }
    Bitboard check_squares(PieceType t) const { return st->checkSquares[t]; }
    Bitboard attackers_to(Square s) const { return attackers_to(s, pieces()); }
    Bitboard attackers_to(Square s, Bitboard occ) const;
    template<PieceType T>
    Bitboard attacks_by(Color c) const;

    // --- coups
    bool  legal(Move m) const;
    bool  pseudo_legal(Move m) const;
    bool  gives_check(Move m) const;
    Piece moved_piece(Move m) const { return cases[m.from_sq()]; }
    Piece captured_piece() const { return st->capturedPiece; }
    // Prise (le roque « prend » sa propre tour : ce n'en est pas une)
    bool capture(Move m) const {
        return m.type_of() == EN_PASSANT || (m.type_of() != CASTLING && !empty(m.to_sq()));
    }
    // Coups traites comme tactiques par le tri : prises et promotions en dame
    // (les sous-promotions restent avec les coups tranquilles).
    bool capture_stage(Move m) const { return capture(m) || m.promotion_type() == QUEEN; }

    void do_move(Move m, StateInfo& nouveau) { do_move(m, nouveau, gives_check(m)); }
    void do_move(Move m, StateInfo& nouveau, bool donneEchec);
    void undo_move(Move m);
    void do_null_move(StateInfo& nouveau);
    void undo_null_move();

    bool see_ge(Move m, int seuil = 0) const;  // l'echange sur la case rapporte-t-il >= seuil ?

    // --- etat
    Color      side_to_move() const { return trait; }
    int        game_ply() const { return demiCoups; }
    Key        key() const { return st->key; }
    Key        key_after(Move m) const;
    Key        key_after_null() const;  // cle apres un coup nul (prechargement de la table)
    Key        pawn_key() const { return st->pawnKey; }
    Key        non_pawn_key(Color c) const { return st->nonPawnKey[c]; }
    Key        minor_key() const { return st->minorKey; }
    Key        major_key() const { return st->majorKey; }
    Value      non_pawn_material(Color c) const { return st->nonPawnMaterial[c]; }
    Value      non_pawn_material() const { return st->nonPawnMaterial[WHITE] + st->nonPawnMaterial[BLACK]; }
    int        rule50_count() const { return st->rule50; }
    int        repetition() const { return st->repetition; }
    bool       is_draw(int ply) const;
    bool       upcoming_repetition(int ply) const;
    int        history_length() const;
    StateInfo* state() const { return st; }

    bool pos_is_ok() const;  // coherence complete (en debogage seulement)

   private:
    void poser(Piece p, Square s);
    void oter(Square s);
    void deplacer(Square de, Square a);
    template<bool Jouer>
    void roquer(Color c, Square roi, Square& arriveeRoi, Square& departTour, Square& arriveeTour);

    void     noter_roque(Color c, Square tour);
    void     tout_recalculer();
    void     calculer_echecs();
    void     noter_deplacements(Move m, Piece p, Piece prise, StateInfo& e) const;
    bool     en_passant_legal(Color c, Square ep) const;
    Bitboard bloqueurs(Bitboard tireurs, Square cible, Bitboard& cloueurs) const;

    // --- donnees : de parType a reservoir exclu, tout est remis a zero par set()
    Bitboard   parType[PIECE_TYPE_NB];  // [0] = toutes les pieces
    Bitboard   parCamp[COLOR_NB];
    Piece      cases[SQUARE_NB];
    int        nombre[PIECE_NB];        // [couleur * 8] = toutes les pieces du camp
    int        perteRoque[SQUARE_NB];   // droits perdus quand une piece quitte/atteint la case
    Square     tourRoque[CASTLING_RIGHT_NB];
    Bitboard   cheminRoque[CASTLING_RIGHT_NB];
    int        demiCoups;
    Color      trait;
    StateInfo* st;
    Thread*    fil = nullptr;
    bool       en960;

    alignas(64) mutable NNUE::Accumulator reservoir[NNUE::ACC_POOL];
};

std::ostream& operator<<(std::ostream& os, const Position& pos);

template<PieceType T>
inline Bitboard Position::attacks_by(Color c) const {
    if constexpr (T == PAWN)
        return c == WHITE ? pawn_attacks_bb<WHITE>(pieces(WHITE, PAWN)) : pawn_attacks_bb<BLACK>(pieces(BLACK, PAWN));
    else
    {
        Bitboard tout = 0;
        for (Bitboard b = pieces(c, T); b; )
            tout |= attacks_bb<T>(pop_lsb(b), pieces());
        return tout;
    }
}

// Demi-coups de la partie reelle visibles depuis la position (limite de la
// detection des repetitions).
inline int Position::history_length() const {
    int n = 0;
    for (const StateInfo* e = st->previous; e; e = e->previous)
        ++n;
    return n;
}

inline void Position::poser(Piece p, Square s) {
    const Bitboard b = square_bb(s);
    cases[s]         = p;
    parType[0] |= b;
    parType[type_of(p)] |= b;
    parCamp[color_of(p)] |= b;
    ++nombre[p];
    ++nombre[int(color_of(p)) * 8];
}

inline void Position::oter(Square s) {
    const Piece    p = cases[s];
    const Bitboard b = square_bb(s);
    parType[0] ^= b;
    parType[type_of(p)] ^= b;
    parCamp[color_of(p)] ^= b;
    cases[s] = NO_PIECE;
    --nombre[p];
    --nombre[int(color_of(p)) * 8];
}

inline void Position::deplacer(Square de, Square a) {
    const Piece    p  = cases[de];
    const Bitboard da = square_bb(de) | square_bb(a);
    parType[0] ^= da;
    parType[type_of(p)] ^= da;
    parCamp[color_of(p)] ^= da;
    cases[de] = NO_PIECE;
    cases[a]  = p;
}

}  // namespace one
