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

// Valeurs materielles de reference du moteur. Une seule table, lue par la
// recherche, le SEE et l'evaluation : le moteur Python en avait trois qui ne
// concordaient pas, et personne ne savait laquelle faisait foi.
constexpr Value PawnValue   = 100;
constexpr Value KnightValue = 320;
constexpr Value BishopValue = 330;
constexpr Value RookValue   = 500;
constexpr Value QueenValue  = 950;

constexpr Value PieceValue[PIECE_NB] = {
  0, PawnValue, KnightValue, BishopValue, RookValue, QueenValue, 0, 0,
  0, PawnValue, KnightValue, BishopValue, RookValue, QueenValue, 0, 0};

// Etat qui ne peut pas etre reconstruit depuis la position seule, et qu'il
// faut donc empiler pour pouvoir defaire un coup.
struct StateInfo {
    // Recopie tel quel par do_move()
    Key   pawnKey;
    Key   nonPawnKey[COLOR_NB];  // pieces d'une couleur hors pions, roi compris
    Value nonPawnMaterial[COLOR_NB];
    int   castlingRights;
    int   rule50;
    int   pliesFromNull;

    // Recalcule par do_move()
    Square     epSquare;
    Key        key;
    Bitboard   checkersBB;
    StateInfo* previous;
    Bitboard   blockersForKing[COLOR_NB];
    Bitboard   pinners[COLOR_NB];
    Bitboard   checkSquares[PIECE_TYPE_NB];
    Piece      capturedPiece;
    int        repetition;

    // Case des deux rois apres le coup. Le reseau en a besoin pour savoir dans
    // quel bucket se trouve chaque perspective, sans avoir acces au plateau.
    Square     kingSq[COLOR_NB];

    // Tout ce qui suit sert au reseau. Place en fin de structure pour que
    // do_move() et do_null_move() puissent recopier la seule tete utile.
    DirtyPiece    dirtyPiece;
    std::uint32_t accGen;    // generation de reseau ayant calcule l'accumulateur
    int           accIndex;  // place dans le reservoir, -1 si aucune
};

// Liste chainee d'etats, pour garder l'historique de la partie vivant entre
// deux recherches : la detection de repetition en depend.
using StateListPtr = std::unique_ptr<std::deque<StateInfo>>;

class Position {
   public:
    static void init();

    Position()                           = default;
    Position(const Position&)            = delete;
    Position& operator=(const Position&) = delete;

    // Mise en place
    Position&   set(const std::string& fenStr, bool isChess960, StateInfo* si);
    void        set_thread(Thread* t) { thisThread = t; }

    // Reservoir d'accumulateurs NNUE, indexe par la profondeur depuis la
    // racine. StateInfo n'en porte que l'index, ce qui le ramene de 2,3 Ko a
    // quelques dizaines d'octets.
    NNUE::Accumulator& accumulator(int i) const { return accPool[i]; }
    Thread*     this_thread() const { return thisThread; }
    std::string fen() const;

    // Contenu du plateau
    Piece    piece_on(Square s) const;
    Bitboard pieces(PieceType pt = ALL_PIECES) const;
    template<typename... PieceTypes>
    Bitboard pieces(PieceType pt, PieceTypes... pts) const;
    Bitboard pieces(Color c) const;
    template<typename... PieceTypes>
    Bitboard pieces(Color c, PieceTypes... pts) const;
    bool     empty(Square s) const;
    int      count(PieceType pt, Color c) const;
    template<PieceType Pt>
    int count(Color c) const;
    template<PieceType Pt>
    int count() const;
    template<PieceType Pt>
    Square square(Color c) const;

    // Roque
    bool   can_castle(CastlingRights cr) const;
    bool   castling_impeded(CastlingRights cr) const;
    Square castling_rook_square(CastlingRights cr) const;
    bool   is_chess960() const;

    // Echecs et clouages
    Bitboard checkers() const;
    Bitboard blockers_for_king(Color c) const;
    Bitboard check_squares(PieceType pt) const;
    Bitboard pinners(Color c) const;

    // Attaques
    Bitboard attackers_to(Square s) const;
    Bitboard attackers_to(Square s, Bitboard occupied) const;
    template<PieceType Pt>
    Bitboard attacks_by(Color c) const;

    // Proprietes d'un coup
    bool  legal(Move m) const;
    bool  pseudo_legal(Move m) const;
    bool  capture(Move m) const;
    bool  capture_stage(Move m) const;
    bool  gives_check(Move m) const;
    Piece moved_piece(Move m) const;
    Piece captured_piece() const;

    // Jouer et defaire
    void do_move(Move m, StateInfo& newSt);
    void do_move(Move m, StateInfo& newSt, bool givesCheck);
    void undo_move(Move m);
    void do_null_move(StateInfo& newSt);
    void undo_null_move();

    // Echange statique
    bool see_ge(Move m, int threshold = 0) const;

    // Accesseurs generaux
    Color      side_to_move() const;
    int        game_ply() const;
    Key        key() const;
    Key        key_after(Move m) const;
    Key        pawn_key() const;
    Key        non_pawn_key(Color c) const;
    Value      non_pawn_material(Color c) const;
    Value      non_pawn_material() const;
    int        rule50_count() const;
    bool       is_draw(int ply) const;
    bool       upcoming_repetition(int ply) const;
    int        repetition() const;
    int        history_length() const;
    StateInfo* state() const;

    // Verification interne, compilee uniquement en debug
    bool pos_is_ok() const;

   private:
    void set_dirty_piece(Move m, Piece pc, Piece captured, StateInfo& si) const;
    void set_castling_right(Color c, Square rfrom);
    void set_state();
    void set_check_info();

    template<bool Do>
    void do_castling(Color us, Square from, Square& to, Square& rfrom, Square& rto);

    void put_piece(Piece pc, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);

    Bitboard slider_blockers(Bitboard sliders, Square s, Bitboard& pinners) const;

    // Donnees
    Bitboard   byTypeBB[PIECE_TYPE_NB];
    Bitboard   byColorBB[COLOR_NB];
    Piece      board[SQUARE_NB];
    int        pieceCount[PIECE_NB];
    int        castlingRightsMask[SQUARE_NB];
    Square     castlingRookSquare[CASTLING_RIGHT_NB];
    Bitboard   castlingPath[CASTLING_RIGHT_NB];
    int        gamePly;
    Color      sideToMove;
    StateInfo* st;
    Thread*    thisThread = nullptr;
    bool       chess960;

    // Cache, d'ou le mutable : evaluer ne change pas la position.
    alignas(64) mutable NNUE::Accumulator accPool[NNUE::ACC_POOL];
};

std::ostream& operator<<(std::ostream& os, const Position& pos);

// --------------------------------------------------------- implementations
inline Color Position::side_to_move() const { return sideToMove; }

inline Piece Position::piece_on(Square s) const {
    assert(is_ok(s));
    return board[s];
}

inline bool Position::empty(Square s) const { return piece_on(s) == NO_PIECE; }

inline Piece Position::moved_piece(Move m) const { return piece_on(m.from_sq()); }

inline Bitboard Position::pieces(PieceType pt) const { return byTypeBB[pt]; }

template<typename... PieceTypes>
inline Bitboard Position::pieces(PieceType pt, PieceTypes... pts) const {
    return pieces(pt) | pieces(pts...);
}

inline Bitboard Position::pieces(Color c) const { return byColorBB[c]; }

template<typename... PieceTypes>
inline Bitboard Position::pieces(Color c, PieceTypes... pts) const {
    return pieces(c) & pieces(pts...);
}

inline int Position::count(PieceType pt, Color c) const { return pieceCount[make_piece(c, pt)]; }

template<PieceType Pt>
inline int Position::count(Color c) const {
    return pieceCount[make_piece(c, Pt)];
}

template<PieceType Pt>
inline int Position::count() const {
    return count<Pt>(WHITE) + count<Pt>(BLACK);
}

template<PieceType Pt>
inline Square Position::square(Color c) const {
    assert(count<Pt>(c) == 1);
    return lsb(pieces(c, Pt));
}

inline bool Position::can_castle(CastlingRights cr) const { return st->castlingRights & cr; }

inline bool Position::castling_impeded(CastlingRights cr) const {
    assert(cr == WHITE_OO || cr == WHITE_OOO || cr == BLACK_OO || cr == BLACK_OOO);
    return pieces() & castlingPath[cr];
}

inline Square Position::castling_rook_square(CastlingRights cr) const {
    assert(cr == WHITE_OO || cr == WHITE_OOO || cr == BLACK_OO || cr == BLACK_OOO);
    return castlingRookSquare[cr];
}

inline bool Position::is_chess960() const { return chess960; }

inline Bitboard Position::attackers_to(Square s) const { return attackers_to(s, pieces()); }

template<PieceType Pt>
inline Bitboard Position::attacks_by(Color c) const {
    if constexpr (Pt == PAWN)
        return c == WHITE ? pawn_attacks_bb<WHITE>(pieces(WHITE, PAWN))
                          : pawn_attacks_bb<BLACK>(pieces(BLACK, PAWN));
    else
    {
        Bitboard threats   = 0;
        Bitboard attackers = pieces(c, Pt);
        while (attackers)
            threats |= attacks_bb<Pt>(pop_lsb(attackers), pieces());
        return threats;
    }
}

inline Bitboard Position::checkers() const { return st->checkersBB; }

inline Bitboard Position::blockers_for_king(Color c) const { return st->blockersForKing[c]; }

inline Bitboard Position::pinners(Color c) const { return st->pinners[c]; }

inline Bitboard Position::check_squares(PieceType pt) const { return st->checkSquares[pt]; }

inline int Position::game_ply() const { return gamePly; }

inline int Position::rule50_count() const { return st->rule50; }

inline int Position::repetition() const { return st->repetition; }

// Nombre de demi-coups de la partie reelle accessibles depuis la racine. La
// detection de repetition ne peut pas voir plus loin que cette valeur.
inline int Position::history_length() const {
    int n = 0;
    for (const StateInfo* s = st->previous; s; s = s->previous)
        ++n;
    return n;
}

inline StateInfo* Position::state() const { return st; }

inline Key Position::key() const { return st->key; }

inline Key Position::pawn_key() const { return st->pawnKey; }
inline Key Position::non_pawn_key(Color c) const { return st->nonPawnKey[c]; }

inline Value Position::non_pawn_material(Color c) const { return st->nonPawnMaterial[c]; }

inline Value Position::non_pawn_material() const {
    return st->nonPawnMaterial[WHITE] + st->nonPawnMaterial[BLACK];
}

inline bool Position::capture(Move m) const {
    assert(m.is_ok());
    return (!empty(m.to_sq()) && m.type_of() != CASTLING) || m.type_of() == EN_PASSANT;
}

// Ce que le tri considere comme tactique : toute capture, plus la promotion
// en dame. Les sous-promotions restent des coups tranquilles -- c'est
// exactement l'anomalie qui les faisait passer avant les killers en Python.
inline bool Position::capture_stage(Move m) const {
    assert(m.is_ok());
    return capture(m) || m.promotion_type() == QUEEN;
}

inline Piece Position::captured_piece() const { return st->capturedPiece; }

inline void Position::put_piece(Piece pc, Square s) {
    board[s] = pc;
    byTypeBB[ALL_PIECES] |= byTypeBB[type_of(pc)] |= s;
    byColorBB[color_of(pc)] |= s;
    pieceCount[pc]++;
    pieceCount[make_piece(color_of(pc), ALL_PIECES)]++;
}

inline void Position::remove_piece(Square s) {
    Piece pc = board[s];
    byTypeBB[ALL_PIECES] ^= s;
    byTypeBB[type_of(pc)] ^= s;
    byColorBB[color_of(pc)] ^= s;
    board[s] = NO_PIECE;
    pieceCount[pc]--;
    pieceCount[make_piece(color_of(pc), ALL_PIECES)]--;
}

inline void Position::move_piece(Square from, Square to) {
    Piece    pc     = board[from];
    Bitboard fromTo = from | to;
    byTypeBB[ALL_PIECES] ^= fromTo;
    byTypeBB[type_of(pc)] ^= fromTo;
    byColorBB[color_of(pc)] ^= fromTo;
    board[from] = NO_PIECE;
    board[to]   = pc;
}

inline void Position::do_move(Move m, StateInfo& newSt) { do_move(m, newSt, gives_check(m)); }

}  // namespace one
