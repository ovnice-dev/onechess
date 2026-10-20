#include "position.h"
#include "prof.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <type_traits>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>

#include "movegen.h"

using std::string;

namespace one {

namespace Zobrist {
Key psq[PIECE_NB][SQUARE_NB];
Key enpassant[FILE_NB];
Key castling[CASTLING_RIGHT_NB];
Key side;
}

namespace {

constexpr std::string_view PieceToChar(" PNBRQK  pnbrqk");

// xorshift64* : meme generateur qu'a l'init des magics, graine differente.
struct PRNG {
    std::uint64_t s;
    explicit PRNG(std::uint64_t seed) : s(seed) {}
    std::uint64_t rand() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }
};

// Table de coucou : pour chaque coup REVERSIBLE (une piece non-pion qui va de
// s1 a s2 sur un echiquier vide), on range la cle Zobrist de ce coup. Deux
// fonctions de hachage, deplacement des occupants : c'est du hachage coucou,
// d'ou le nom. Elle sert a reconnaitre en O(1) qu'un coup disponible ramene a
// une position deja vue -- donc a une nulle par repetition.
Key  cuckoo[8192];
Move cuckooMove[8192];

inline int H1(Key h) { return int(h & 0x1fff); }
inline int H2(Key h) { return int((h >> 16) & 0x1fff); }

}  // namespace

std::ostream& operator<<(std::ostream& os, const Position& pos) {
    os << "\n +---+---+---+---+---+---+---+---+\n";

    for (Rank r = RANK_8; r >= RANK_1; --r)
    {
        for (File f = FILE_A; f <= FILE_H; ++f)
            os << " | " << PieceToChar[pos.piece_on(make_square(f, r))];

        os << " | " << (1 + r) << "\n +---+---+---+---+---+---+---+---+\n";
    }

    os << "   a   b   c   d   e   f   g   h\n"
       << "\nFen: " << pos.fen() << "\nCle: " << std::hex << std::uppercase << std::setfill('0')
       << std::setw(16) << pos.key() << std::setfill(' ') << std::dec << "\n";

    return os;
}

void Position::init() {
    PRNG rng(1070372);

    for (Piece pc : {W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING, B_PAWN, B_KNIGHT,
                     B_BISHOP, B_ROOK, B_QUEEN, B_KING})
        for (Square s = SQ_A1; s <= SQ_H8; ++s)
            Zobrist::psq[pc][s] = rng.rand();

    for (File f = FILE_A; f <= FILE_H; ++f)
        Zobrist::enpassant[f] = rng.rand();

    for (int cr = NO_CASTLING; cr <= ANY_CASTLING; ++cr)
        Zobrist::castling[cr] = rng.rand();

    Zobrist::side = rng.rand();

    // Remplissage de la table de coucou. 3668 coups reversibles au total :
    // 168 du cavalier, 280 du fou, 448 de la tour, 728 de la dame et 210 du
    // roi, pour chacune des deux couleurs. Le pion est exclu : il n'est
    // jamais reversible.
    std::memset(cuckoo, 0, sizeof(cuckoo));
    for (int i = 0; i < 8192; ++i)
        cuckooMove[i] = Move::none();

    int count = 0;
    for (Piece pc : {W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING, B_KNIGHT, B_BISHOP, B_ROOK,
                     B_QUEEN, B_KING})
        for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1)
            for (Square s2 = Square(s1 + 1); s2 <= SQ_H8; ++s2)
                if (attacks_bb(type_of(pc), s1, 0) & s2)
                {
                    Move move = Move(s1, s2);
                    Key  key  = Zobrist::psq[pc][s1] ^ Zobrist::psq[pc][s2] ^ Zobrist::side;
                    int  i    = H1(key);
                    while (true)
                    {
                        std::swap(cuckoo[i], key);
                        std::swap(cuckooMove[i], move);
                        if (move == Move::none())  // case libre atteinte
                            break;
                        i = (i == H1(key)) ? H2(key) : H1(key);  // on deloge l'occupant
                    }
                    ++count;
                }
    assert(count == 3668);
    (void) count;
}

// Lit une position au format FEN. Le champ de roque accepte la notation
// classique (KQkq) et la notation Shredder (colonnes des tours) pour le 960.
Position& Position::set(const string& fenStr, bool isChess960, StateInfo* si) {
    unsigned char      col, row, token;
    size_t             idx;
    Square             sq = SQ_A8;
    std::istringstream ss(fenStr);

    // memset efface aussi le thread proprietaire : on le remet apres.
    // On s'arrete avant le reservoir d'accumulateurs : sa validite est portee
    // par accGen, pas par son contenu, et l'effacer couterait 532 Ko a chaque
    // mise en place.
    static_assert(std::is_standard_layout_v<Position>,
                  "offsetof exige une disposition standard");
    Thread* const owner = thisThread;
    std::memset(this, 0, offsetof(Position, accPool));
    std::memset(si, 0, sizeof(StateInfo));
    thisThread = owner;
    st         = si;

    ss >> std::noskipws;

    // 1. Placement des pieces
    while ((ss >> token) && !isspace(token))
    {
        if (isdigit(token))
            sq += (token - '0') * EAST;

        else if (token == '/')
            sq += 2 * SOUTH;

        else if ((idx = PieceToChar.find(char(token))) != string::npos)
        {
            put_piece(Piece(idx), sq);
            ++sq;
        }
    }

    // 2. Trait
    ss >> token;
    sideToMove = (token == 'w' ? WHITE : BLACK);
    ss >> token;

    // 3. Droits de roque
    while ((ss >> token) && !isspace(token))
    {
        Square rsq;
        Color  c    = islower(token) ? BLACK : WHITE;
        Piece  rook = make_piece(c, ROOK);

        token = char(toupper(token));

        if (token == 'K')
            for (rsq = relative_square(c, SQ_H1); piece_on(rsq) != rook; --rsq) {}

        else if (token == 'Q')
            for (rsq = relative_square(c, SQ_A1); piece_on(rsq) != rook; ++rsq) {}

        else if (token >= 'A' && token <= 'H')
            rsq = make_square(File(token - 'A'), relative_rank(c, RANK_1));

        else
            continue;

        set_castling_right(c, rsq);
    }

    // 4. Case de prise en passant. On ne la retient que si une prise est
    //    reellement disponible : sinon la cle Zobrist differerait entre deux
    //    positions identiques, et la table de transposition les manquerait.
    bool enpassant = false;

    if (((ss >> col) && (col >= 'a' && col <= 'h'))
        && ((ss >> row) && (row == (sideToMove == WHITE ? '6' : '3'))))
    {
        st->epSquare = make_square(File(col - 'a'), Rank(row - '1'));

        enpassant = (pawn_attacks_bb(~sideToMove, st->epSquare) & pieces(sideToMove, PAWN))
                 && (pieces(~sideToMove, PAWN) & (st->epSquare + pawn_push(~sideToMove)))
                 && !(pieces() & (st->epSquare | (st->epSquare + pawn_push(sideToMove))));
    }

    if (!enpassant)
        st->epSquare = SQ_NONE;

    // 5-6. Demi-coups depuis la derniere prise ou poussee, puis numero de coup
    ss >> std::skipws >> st->rule50 >> gamePly;

    // gamePly compte en demi-coups depuis 0, avec les noirs a l'impair
    gamePly = std::max(2 * (gamePly - 1), 0) + (sideToMove == BLACK);

    chess960 = isChess960;
    set_state();

    assert(pos_is_ok());

    return *this;
}

// Releve les pieces qui changent de case, pour que le reseau puisse mettre son
// accumulateur a jour au lieu de le recalculer. Doit etre appele avant toute
// modification du plateau.
void Position::set_dirty_piece(Move m, Piece pc, Piece captured, StateInfo& si) const {
    DirtyPiece&  dp   = si.dirtyPiece;
    const Color  us   = sideToMove;
    const Square from = m.from_sq();
    const Square to   = m.to_sq();

    dp.count = 0;

    if (m.type_of() == CASTLING)
    {
        // to designe la tour ; le roi et la tour bougent tous les deux
        const bool   kingSide = to > from;
        const Square kto      = relative_square(us, kingSide ? SQ_G1 : SQ_C1);
        const Square rto      = relative_square(us, kingSide ? SQ_F1 : SQ_D1);

        dp.piece[0] = pc;
        dp.from[0]  = from;
        dp.to[0]    = kto;
        dp.piece[1] = make_piece(us, ROOK);
        dp.from[1]  = to;
        dp.to[1]    = rto;
        dp.count    = 2;
        return;
    }

    if (captured)
    {
        const Square capsq = m.type_of() == EN_PASSANT ? to - pawn_push(us) : to;
        dp.piece[dp.count] = captured;
        dp.from[dp.count]  = capsq;
        dp.to[dp.count]    = SQ_NONE;
        ++dp.count;
    }

    if (m.type_of() == PROMOTION)
    {
        // Le pion disparait, une autre piece apparait ailleurs
        dp.piece[dp.count] = pc;
        dp.from[dp.count]  = from;
        dp.to[dp.count]    = SQ_NONE;
        ++dp.count;

        dp.piece[dp.count] = make_piece(us, m.promotion_type());
        dp.from[dp.count]  = SQ_NONE;
        dp.to[dp.count]    = to;
        ++dp.count;
    }
    else
    {
        dp.piece[dp.count] = pc;
        dp.from[dp.count]  = from;
        dp.to[dp.count]    = to;
        ++dp.count;
    }

    assert(dp.count <= 3);
}

// Enregistre un droit de roque et precalcule le chemin a laisser libre.
void Position::set_castling_right(Color c, Square rfrom) {
    Square kfrom = square<KING>(c);
    auto   cr    = c & (kfrom < rfrom ? KING_SIDE : QUEEN_SIDE);

    st->castlingRights |= cr;
    castlingRightsMask[kfrom] |= cr;
    castlingRightsMask[rfrom] |= cr;
    castlingRookSquare[cr] = rfrom;

    Square kto = relative_square(c, cr & KING_SIDE ? SQ_G1 : SQ_C1);
    Square rto = relative_square(c, cr & KING_SIDE ? SQ_F1 : SQ_D1);

    castlingPath[cr] = (between_bb(rfrom, rto) | between_bb(kfrom, kto)) & ~(kfrom | rfrom);
}

// Recalcule tout ce qui depend de la position courante : bloqueurs, cloueurs,
// cases donnant echec.
void Position::set_check_info() {
    st->blockersForKing[WHITE] =
      slider_blockers(pieces(BLACK), square<KING>(WHITE), st->pinners[BLACK]);
    st->blockersForKing[BLACK] =
      slider_blockers(pieces(WHITE), square<KING>(BLACK), st->pinners[WHITE]);

    Square ksq = square<KING>(~sideToMove);

    st->checkSquares[PAWN]   = pawn_attacks_bb(~sideToMove, ksq);
    st->checkSquares[KNIGHT] = attacks_bb<KNIGHT>(ksq);
    st->checkSquares[BISHOP] = attacks_bb<BISHOP>(ksq, pieces());
    st->checkSquares[ROOK]   = attacks_bb<ROOK>(ksq, pieces());
    st->checkSquares[QUEEN]  = st->checkSquares[BISHOP] | st->checkSquares[ROOK];
    st->checkSquares[KING]   = 0;

    st->kingSq[WHITE] = square<KING>(WHITE);
    st->kingSq[BLACK] = square<KING>(BLACK);
}

// Reconstruit cles Zobrist et materiel a partir de zero. Appele une fois par
// set() ; ensuite tout est maintenu incrementalement par do_move().
void Position::set_state() {
    st->key        = 0;
    st->pawnKey    = 0;
    st->nonPawnKey[WHITE] = st->nonPawnKey[BLACK] = 0;
    st->checkersBB = attackers_to(square<KING>(sideToMove)) & pieces(~sideToMove);
    st->nonPawnMaterial[WHITE] = st->nonPawnMaterial[BLACK] = VALUE_ZERO;

    set_check_info();

    for (Bitboard b = pieces(); b;)
    {
        Square s  = pop_lsb(b);
        Piece  pc = piece_on(s);
        st->key ^= Zobrist::psq[pc][s];
        if (type_of(pc) != PAWN)
            st->nonPawnKey[color_of(pc)] ^= Zobrist::psq[pc][s];

        if (type_of(pc) == PAWN)
            st->pawnKey ^= Zobrist::psq[pc][s];
        else if (type_of(pc) != KING)
            st->nonPawnMaterial[color_of(pc)] += PieceValue[pc];
    }

    if (st->epSquare != SQ_NONE)
        st->key ^= Zobrist::enpassant[file_of(st->epSquare)];

    if (sideToMove == BLACK)
        st->key ^= Zobrist::side;

    st->key ^= Zobrist::castling[st->castlingRights];
}

string Position::fen() const {
    int                emptyCnt;
    std::ostringstream ss;

    for (Rank r = RANK_8; r >= RANK_1; --r)
    {
        for (File f = FILE_A; f <= FILE_H; ++f)
        {
            for (emptyCnt = 0; f <= FILE_H && empty(make_square(f, r)); ++f)
                ++emptyCnt;

            if (emptyCnt)
                ss << emptyCnt;

            if (f <= FILE_H)
                ss << PieceToChar[piece_on(make_square(f, r))];
        }

        if (r > RANK_1)
            ss << '/';
    }

    ss << (sideToMove == WHITE ? " w " : " b ");

    if (can_castle(WHITE_OO))
        ss << (chess960 ? char('A' + file_of(castling_rook_square(WHITE_OO))) : 'K');
    if (can_castle(WHITE_OOO))
        ss << (chess960 ? char('A' + file_of(castling_rook_square(WHITE_OOO))) : 'Q');
    if (can_castle(BLACK_OO))
        ss << (chess960 ? char('a' + file_of(castling_rook_square(BLACK_OO))) : 'k');
    if (can_castle(BLACK_OOO))
        ss << (chess960 ? char('a' + file_of(castling_rook_square(BLACK_OOO))) : 'q');

    if (!can_castle(ANY_CASTLING))
        ss << '-';

    ss << (st->epSquare == SQ_NONE ? " - " : " " + std::string{char('a' + file_of(st->epSquare)),
                                                               char('1' + rank_of(st->epSquare))}
                                               + " ")
       << st->rule50 << " " << 1 + (gamePly - (sideToMove == BLACK)) / 2;

    return ss.str();
}

// Pieces qui, si elles bougeaient, decouvriraient le roi en s. pinners recoit
// les glisseurs qui clouent effectivement.
Bitboard Position::slider_blockers(Bitboard sliders, Square s, Bitboard& pinners) const {
    Bitboard blockers = 0;
    pinners           = 0;

    // Glisseurs qui viseraient s si tout ce qui les separe disparaissait
    Bitboard snipers = ((attacks_bb<ROOK>(s) & pieces(QUEEN, ROOK))
                        | (attacks_bb<BISHOP>(s) & pieces(QUEEN, BISHOP)))
                     & sliders;
    Bitboard occupancy = pieces() ^ snipers;

    while (snipers)
    {
        Square   sniperSq = pop_lsb(snipers);
        Bitboard b        = between_bb(s, sniperSq) & occupancy;

        if (b && !more_than_one(b))
        {
            blockers |= b;
            if (b & pieces(color_of(piece_on(s))))
                pinners |= sniperSq;
        }
    }
    return blockers;
}

// Toutes les pieces des deux couleurs qui attaquent s, pour une occupation
// donnee (le parametre permet de simuler des echanges dans see_ge).
Bitboard Position::attackers_to(Square s, Bitboard occupied) const {
    return (pawn_attacks_bb(BLACK, s) & pieces(WHITE, PAWN))
         | (pawn_attacks_bb(WHITE, s) & pieces(BLACK, PAWN))
         | (attacks_bb<KNIGHT>(s) & pieces(KNIGHT))
         | (attacks_bb<ROOK>(s, occupied) & pieces(ROOK, QUEEN))
         | (attacks_bb<BISHOP>(s, occupied) & pieces(BISHOP, QUEEN))
         | (attacks_bb<KING>(s) & pieces(KING));
}

// Le coup pseudo-legal m laisse-t-il notre roi hors d'echec ?
bool Position::legal(Move m) const {
    PROF_SCOPE(LEGAL);
    assert(m.is_ok());

    Color  us   = sideToMove;
    Square from = m.from_sq();
    Square to   = m.to_sq();

    assert(color_of(moved_piece(m)) == us);
    assert(piece_on(square<KING>(us)) == make_piece(us, KING));

    if (m.type_of() == CASTLING)
    {
        // Ici to designe la tour. Le roi ne doit traverser aucune case
        // attaquee, case de depart comprise.
        to             = relative_square(us, to > from ? SQ_G1 : SQ_C1);
        Direction step = to > from ? WEST : EAST;

        for (Square s = to; s != from; s += step)
            if (attackers_to(s) & pieces(~us))
                return false;

        // En Chess960 la tour du roque peut elle-meme clouer le roi
        return !chess960 || !(blockers_for_king(us) & m.to_sq());
    }

    // La prise en passant est le seul coup ou deux pieces quittent la meme
    // rangee : aucun masque de clouage ne l'attrape, il faut simuler.
    if (m.type_of() == EN_PASSANT)
    {
        Square   ksq      = square<KING>(us);
        Square   capsq    = to - pawn_push(us);
        Bitboard occupied = (pieces() ^ from ^ capsq) | to;

        assert(to == st->epSquare);
        assert(moved_piece(m) == make_piece(us, PAWN));
        assert(piece_on(capsq) == make_piece(~us, PAWN));
        assert(piece_on(to) == NO_PIECE);

        return !(attacks_bb<ROOK>(ksq, occupied) & pieces(~us, QUEEN, ROOK))
            && !(attacks_bb<BISHOP>(ksq, occupied) & pieces(~us, QUEEN, BISHOP));
    }

    // Le roi ne peut pas aller sur une case attaquee, et il faut l'enlever de
    // l'occupation pour ne pas se cacher derriere lui-meme.
    if (type_of(piece_on(from)) == KING)
        return !(attackers_to(to, pieces() ^ from) & pieces(~us));

    // Une piece clouee ne peut bouger que le long de son clouage
    return !(blockers_for_king(us) & from) || aligned(from, to, square<KING>(us));
}

// Verifie qu'un coup venu de la table de transposition ou des killers est
// bien jouable dans la position courante. Doit rejeter tout coup corrompu.
bool Position::pseudo_legal(Move m) const {
    Color  us   = sideToMove;
    Square from = m.from_sq();
    Square to   = m.to_sq();
    Piece  pc   = moved_piece(m);

    if (m.type_of() != NORMAL)
        return checkers() ? MoveList<EVASIONS>(*this).contains(m)
                          : MoveList<NON_EVASIONS>(*this).contains(m);

    // Un coup NORMAL n'a jamais de champ promotion non nul
    if (m.promotion_type() - KNIGHT != NO_PIECE_TYPE)
        return false;

    if (pc == NO_PIECE || color_of(pc) != us)
        return false;

    if (pieces(us) & to)
        return false;

    if (type_of(pc) == PAWN)
    {
        // Une arrivee en derniere rangee exigerait le type PROMOTION
        if ((Rank8BB | Rank1BB) & to)
            return false;

        if (!(pawn_attacks_bb(us, from) & pieces(~us) & to)         // prise simple
            && !((from + pawn_push(us) == to) && empty(to))         // poussee d'une case
            && !((from + 2 * pawn_push(us) == to)                   // poussee double
                 && (relative_rank(us, from) == RANK_2) && empty(to)
                 && empty(to - pawn_push(us))))
            return false;
    }
    else if (!(attacks_bb(type_of(pc), from, pieces()) & to))
        return false;

    if (checkers())
    {
        if (type_of(pc) != KING)
        {
            // Un double echec ne se pare que par un coup de roi
            if (more_than_one(checkers()))
                return false;

            // Il faut bloquer ou capturer
            if (!(between_bb(square<KING>(us), lsb(checkers())) & to))
                return false;
        }
        else if (attackers_to(to, pieces() ^ from) & pieces(~us))
            return false;
    }

    return true;
}

// Le coup donne-t-il echec ? Repond sans jouer le coup.
bool Position::gives_check(Move m) const {
    PROF_SCOPE(GIVESCHECK);
    assert(m.is_ok());
    assert(color_of(moved_piece(m)) == sideToMove);

    Square from = m.from_sq();
    Square to   = m.to_sq();

    // Echec direct
    if (check_squares(type_of(piece_on(from))) & to)
        return true;

    // Echec a la decouverte
    if (blockers_for_king(~sideToMove) & from)
        return !aligned(from, to, square<KING>(~sideToMove)) || m.type_of() == CASTLING;

    switch (m.type_of())
    {
    case NORMAL :
        return false;

    case PROMOTION :
        return attacks_bb(m.promotion_type(), to, pieces() ^ from) & square<KING>(~sideToMove);

    // Le pion capture quitte sa case : cela peut ouvrir une ligne
    case EN_PASSANT : {
        Square   capsq = make_square(file_of(to), rank_of(from));
        Bitboard b     = (pieces() ^ from ^ capsq) | to;

        return (attacks_bb<ROOK>(square<KING>(~sideToMove), b) & pieces(sideToMove, QUEEN, ROOK))
             | (attacks_bb<BISHOP>(square<KING>(~sideToMove), b)
                & pieces(sideToMove, QUEEN, BISHOP));
    }

    default :  // CASTLING : seule la tour peut donner echec
    {
        Square rto = relative_square(sideToMove, to > from ? SQ_F1 : SQ_D1);
        return check_squares(ROOK) & rto;
    }
    }
}

void Position::do_move(Move m, StateInfo& newSt, bool givesCheck) {
    PROF_SCOPE(DOMOVE);
    assert(m.is_ok());
    assert(&newSt != st);

    Key k = st->key ^ Zobrist::side;

    // Recopie la partie de l'etat qui survit au coup, puis chaine
    std::memcpy(&newSt, st, offsetof(StateInfo, epSquare));
    newSt.previous = st;
    st             = &newSt;

    ++gamePly;
    ++st->rule50;
    ++st->pliesFromNull;

    Color  us       = sideToMove;
    Color  them     = ~us;
    Square from     = m.from_sq();
    Square to       = m.to_sq();
    Piece  pc       = piece_on(from);
    Piece  captured = m.type_of() == EN_PASSANT ? make_piece(them, PAWN) : piece_on(to);

    assert(color_of(pc) == us);
    assert(captured == NO_PIECE || color_of(captured) == (m.type_of() != CASTLING ? them : us));
    assert(type_of(captured) != KING);

    set_dirty_piece(m, pc, captured, *st);
    st->accGen = 0;

    // Place suivante dans le reservoir. -1 signifie "aucune" : l'evaluation
    // repartira alors de zero. Cela n'arrive qu'au-dela de ACC_POOL coups
    // joues depuis la mise en place, jamais dans la recherche.
    st->accIndex = (st->previous->accIndex >= 0
                    && st->previous->accIndex + 1 < NNUE::ACC_POOL)
                   ? st->previous->accIndex + 1
                   : -1;

    if (m.type_of() == CASTLING)
    {
        assert(pc == make_piece(us, KING));
        assert(captured == make_piece(us, ROOK));

        Square rfrom, rto;
        do_castling<true>(us, from, to, rfrom, rto);

        k ^= Zobrist::psq[captured][rfrom] ^ Zobrist::psq[captured][rto];
        st->nonPawnKey[us] ^= Zobrist::psq[captured][rfrom] ^ Zobrist::psq[captured][rto];
        captured = NO_PIECE;
    }

    if (captured)
    {
        Square capsq = to;

        if (type_of(captured) == PAWN)
        {
            if (m.type_of() == EN_PASSANT)
            {
                capsq -= pawn_push(us);

                assert(pc == make_piece(us, PAWN));
                assert(to == st->previous->epSquare);
                assert(relative_rank(us, to) == RANK_6);
                assert(piece_on(to) == NO_PIECE);
                assert(piece_on(capsq) == make_piece(them, PAWN));
            }

            st->pawnKey ^= Zobrist::psq[captured][capsq];
        }
        else
        {
            st->nonPawnMaterial[them] -= PieceValue[captured];
            st->nonPawnKey[them] ^= Zobrist::psq[captured][capsq];
        }

        remove_piece(capsq);
        k ^= Zobrist::psq[captured][capsq];

        st->rule50 = 0;
    }

    k ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
    if (type_of(pc) != PAWN)
        st->nonPawnKey[us] ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];

    // La case de prise en passant precedente disparait dans tous les cas
    if (st->previous->epSquare != SQ_NONE)
        k ^= Zobrist::enpassant[file_of(st->previous->epSquare)];
    st->epSquare = SQ_NONE;

    if (st->castlingRights && (castlingRightsMask[from] | castlingRightsMask[to]))
    {
        k ^= Zobrist::castling[st->castlingRights];
        st->castlingRights &= ~(castlingRightsMask[from] | castlingRightsMask[to]);
        k ^= Zobrist::castling[st->castlingRights];
    }

    if (m.type_of() != CASTLING)
        move_piece(from, to);

    if (type_of(pc) == PAWN)
    {
        // On ne pose la case de prise en passant que si elle est reellement
        // prenable, pour que deux positions identiques aient la meme cle.
        if ((int(to) ^ int(from)) == 16
            && (pawn_attacks_bb(us, to - pawn_push(us)) & pieces(them, PAWN)))
        {
            st->epSquare = to - pawn_push(us);
            k ^= Zobrist::enpassant[file_of(st->epSquare)];
        }

        else if (m.type_of() == PROMOTION)
        {
            Piece promotion = make_piece(us, m.promotion_type());

            assert(relative_rank(us, to) == RANK_8);
            assert(type_of(promotion) >= KNIGHT && type_of(promotion) <= QUEEN);

            remove_piece(to);
            put_piece(promotion, to);

            k ^= Zobrist::psq[pc][to] ^ Zobrist::psq[promotion][to];
            st->pawnKey ^= Zobrist::psq[pc][to];
            st->nonPawnMaterial[us] += PieceValue[promotion];
            st->nonPawnKey[us] ^= Zobrist::psq[promotion][to];
        }

        st->pawnKey ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
        st->rule50 = 0;
    }

    st->capturedPiece = captured;
    st->key           = k;
    st->checkersBB    = givesCheck ? attackers_to(square<KING>(them)) & pieces(us) : Bitboard(0);

    sideToMove = ~sideToMove;

    set_check_info();

    // Repetition : on ne remonte que jusqu'a la derniere prise, poussee de
    // pion ou coup nul, et de deux en deux (meme trait).
    st->repetition = 0;
    int end        = std::min(st->rule50, st->pliesFromNull);
    if (end >= 4)
    {
        StateInfo* stp = st->previous->previous;
        for (int i = 4; i <= end; i += 2)
        {
            stp = stp->previous->previous;
            if (stp->key == st->key)
            {
                st->repetition = stp->repetition ? -i : i;
                break;
            }
        }
    }

    assert(pos_is_ok());
}

void Position::undo_move(Move m) {
    PROF_SCOPE(UNDOMOVE);
    assert(m.is_ok());

    sideToMove = ~sideToMove;

    Color  us   = sideToMove;
    Square from = m.from_sq();
    Square to   = m.to_sq();
    Piece  pc   = piece_on(to);

    assert(empty(from) || m.type_of() == CASTLING);
    assert(type_of(st->capturedPiece) != KING);

    if (m.type_of() == PROMOTION)
    {
        assert(relative_rank(us, to) == RANK_8);
        assert(type_of(pc) == m.promotion_type());
        assert(type_of(pc) >= KNIGHT && type_of(pc) <= QUEEN);

        remove_piece(to);
        pc = make_piece(us, PAWN);
        put_piece(pc, to);
    }

    if (m.type_of() == CASTLING)
    {
        Square rfrom, rto;
        do_castling<false>(us, from, to, rfrom, rto);
    }
    else
    {
        move_piece(to, from);

        if (st->capturedPiece)
        {
            Square capsq = to;

            if (m.type_of() == EN_PASSANT)
            {
                capsq -= pawn_push(us);

                assert(type_of(pc) == PAWN);
                assert(to == st->previous->epSquare);
                assert(relative_rank(us, to) == RANK_6);
                assert(piece_on(capsq) == NO_PIECE);
                assert(st->capturedPiece == make_piece(~us, PAWN));
            }

            put_piece(st->capturedPiece, capsq);
        }
    }

    st = st->previous;
    --gamePly;

    assert(pos_is_ok());
}

// Deplace roi et tour d'un coup. Do=true joue, Do=false defait.
template<bool Do>
void Position::do_castling(Color us, Square from, Square& to, Square& rfrom, Square& rto) {
    bool kingSide = to > from;
    rfrom         = to;  // to designe la tour
    rto           = relative_square(us, kingSide ? SQ_F1 : SQ_D1);
    to            = relative_square(us, kingSide ? SQ_G1 : SQ_C1);

    // On retire les deux pieces avant de les reposer : en Chess960 elles
    // peuvent atterrir sur la case l'une de l'autre.
    remove_piece(Do ? from : to);
    remove_piece(Do ? rfrom : rto);
    board[Do ? from : to] = board[Do ? rfrom : rto] = NO_PIECE;
    put_piece(make_piece(us, KING), Do ? to : from);
    put_piece(make_piece(us, ROOK), Do ? rto : rfrom);
}

void Position::do_null_move(StateInfo& newSt) {
    assert(!checkers());
    assert(&newSt != st);

    // Seule la tete utile est recopiee : l'accumulateur pese 2 Ko et sera
    // repris paresseusement, avec un delta vide.
    std::memcpy(&newSt, st, offsetof(StateInfo, dirtyPiece));

    newSt.previous         = st;
    newSt.dirtyPiece.count = 0;
    newSt.accGen           = 0;
    newSt.accIndex = (st->accIndex >= 0 && st->accIndex + 1 < NNUE::ACC_POOL)
                     ? st->accIndex + 1
                     : -1;
    st = &newSt;

    if (st->epSquare != SQ_NONE)
    {
        st->key ^= Zobrist::enpassant[file_of(st->epSquare)];
        st->epSquare = SQ_NONE;
    }

    st->key ^= Zobrist::side;

    ++st->rule50;
    st->pliesFromNull = 0;

    sideToMove = ~sideToMove;

    set_check_info();

    st->repetition = 0;

    assert(pos_is_ok());
}

void Position::undo_null_move() {
    assert(!checkers());

    st         = st->previous;
    sideToMove = ~sideToMove;
}

// Cle Zobrist qu'aura la position apres m, sans jouer le coup. Sert au
// prefetch de la table de transposition.
Key Position::key_after(Move m) const {
    PROF_SCOPE(KEYAFTER);
    Square from     = m.from_sq();
    Square to       = m.to_sq();
    Piece  pc       = piece_on(from);
    Piece  captured = piece_on(to);
    Key    k        = st->key ^ Zobrist::side;

    if (captured)
        k ^= Zobrist::psq[captured][to];

    k ^= Zobrist::psq[pc][to] ^ Zobrist::psq[pc][from];

    return k;
}

// Echange statique : la suite des prises sur to rapporte-t-elle au moins
// threshold ? Rend la reponse sans construire l'arbre des echanges.
bool Position::see_ge(Move m, int threshold) const {
    PROF_SCOPE(SEE);
    assert(m.is_ok());

    // Les coups speciaux sont supposes bons : les evaluer correctement
    // couterait plus cher que ce que le filtre fait gagner.
    if (m.type_of() != NORMAL)
        return VALUE_ZERO >= threshold;

    Square from = m.from_sq();
    Square to   = m.to_sq();

    int swap = PieceValue[piece_on(to)] - threshold;
    if (swap < 0)
        return false;

    swap = PieceValue[piece_on(from)] - swap;
    if (swap <= 0)
        return true;

    assert(color_of(piece_on(from)) == sideToMove);

    Bitboard occupied  = pieces() ^ from ^ to;
    Color    stm       = sideToMove;
    Bitboard attackers = attackers_to(to, occupied);
    Bitboard stmAttackers, bb;
    int      res = 1;

    while (true)
    {
        stm = ~stm;
        attackers &= occupied;

        if (!(stmAttackers = attackers & pieces(stm)))
            break;

        // Une piece clouee sur le roi par un adversaire encore present ne
        // peut pas participer a l'echange.
        if (pinners(~stm) & occupied)
        {
            stmAttackers &= ~blockers_for_king(stm);

            if (!stmAttackers)
                break;
        }

        res ^= 1;

        // On prend toujours avec la piece la moins chere disponible, et on
        // redecouvre les rayons X apres chaque prise.
        if ((bb = stmAttackers & pieces(PAWN)))
        {
            if ((swap = PawnValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            attackers |= attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN);
        }

        else if ((bb = stmAttackers & pieces(KNIGHT)))
        {
            if ((swap = KnightValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);
        }

        else if ((bb = stmAttackers & pieces(BISHOP)))
        {
            if ((swap = BishopValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            attackers |= attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN);
        }

        else if ((bb = stmAttackers & pieces(ROOK)))
        {
            if ((swap = RookValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            attackers |= attacks_bb<ROOK>(to, occupied) & pieces(ROOK, QUEEN);
        }

        else if ((bb = stmAttackers & pieces(QUEEN)))
        {
            if ((swap = QueenValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            attackers |= (attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN))
                       | (attacks_bb<ROOK>(to, occupied) & pieces(ROOK, QUEEN));
        }

        else  // il ne reste que le roi : il ne peut prendre que si l'autre
              // camp n'a plus d'attaquant sur la case
            return (attackers & ~pieces(stm)) ? res ^ 1 : res;
    }

    return bool(res);
}

// Nulle par repetition, par la regle des 50 coups, ou par materiel insuffisant.
bool Position::is_draw(int ply) const {
    PROF_SCOPE(DRAW);
    if (st->rule50 > 99 && (!checkers() || MoveList<LEGAL>(*this).size()))
        return true;

    // Dans l'arbre, une seule repetition suffit a declarer nulle : la
    // deuxieme occurrence dans la partie reelle est traitee a la racine.
    return st->repetition && st->repetition < ply;
}

// Vrai si le camp au trait dispose d'un coup qui ramene a une position deja
// rencontree -- autrement dit s'il peut forcer la nulle par repetition. On ne
// genere aucun coup : on remonte l'historique de deux en deux et on demande a
// la table de coucou si l'ecart entre les deux cles correspond a un coup
// reversible unique. Reste a verifier que le chemin est libre.
bool Position::upcoming_repetition(int ply) const {

    int end = std::min(st->rule50, st->pliesFromNull);

    if (end < 3)
        return false;

    Key        originalKey = st->key;
    StateInfo* stp         = st->previous;
    Key        other       = originalKey ^ stp->key ^ Zobrist::side;

    for (int i = 3; i <= end; i += 2)
    {
        stp = stp->previous;
        other ^= stp->key ^ stp->previous->key ^ Zobrist::side;
        stp = stp->previous;

        // "other" ne s'annule que si tous les coups intermediaires de
        // l'adversaire se sont annules deux a deux : sinon ce n'est pas un
        // simple aller-retour.
        if (other != 0)
            continue;

        Key moveKey = originalKey ^ stp->key;

        if (int j = H1(moveKey); cuckoo[j] == moveKey || (j = H2(moveKey), cuckoo[j] == moveKey))
        {
            Move   move = cuckooMove[j];
            Square s1   = move.from_sq();
            Square s2   = move.to_sq();

            // between_bb inclut s2 : on le retire pour ne tester que les cases
            // traversees, la case d'arrivee etant forcement vide ici.
            if (!((between_bb(s1, s2) ^ s2) & pieces()))
            {
                if (ply > i)
                    return true;

                // A la racine ou au-dessus, il faut que ce soit une vraie
                // repetition, pas un coup qui mene a la position courante.
                if (stp->repetition)
                    return true;
            }
        }
    }
    return false;
}

// Verification complete de la coherence interne. Ne tourne qu'en debug.
bool Position::pos_is_ok() const {
#ifdef NDEBUG
    return true;
#else
    if ((sideToMove != WHITE && sideToMove != BLACK) || piece_on(square<KING>(WHITE)) != W_KING
        || piece_on(square<KING>(BLACK)) != B_KING
        || (st->epSquare != SQ_NONE && relative_rank(sideToMove, st->epSquare) != RANK_6))
        return false;

    if (count<KING>(WHITE) != 1 || count<KING>(BLACK) != 1
        || attackers_to(square<KING>(~sideToMove)) & pieces(sideToMove))
        return false;

    if ((pieces(PAWN) & (Rank1BB | Rank8BB)) || count<PAWN>(WHITE) > 8 || count<PAWN>(BLACK) > 8)
        return false;

    if ((pieces(WHITE) & pieces(BLACK)) || (pieces(WHITE) | pieces(BLACK)) != pieces()
        || popcount(pieces(WHITE)) > 16 || popcount(pieces(BLACK)) > 16)
        return false;

    for (PieceType p1 = PAWN; p1 <= KING; ++p1)
        for (PieceType p2 = PAWN; p2 <= KING; ++p2)
            if (p1 != p2 && (pieces(p1) & pieces(p2)))
                return false;

    for (Piece pc : {W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING, B_PAWN, B_KNIGHT,
                     B_BISHOP, B_ROOK, B_QUEEN, B_KING})
        if (pieceCount[pc] != popcount(pieces(color_of(pc), type_of(pc)))
            || pieceCount[pc] != std::count(board, board + SQUARE_NB, pc))
            return false;

    // Cle non-pions recalculee de zero et comparee a celle que tient do_move().
    // Une erreur de mise a jour incrementale ne planterait rien : elle
    // fausserait en silence l'historique de correction qui s'en sert.
    Key np[COLOR_NB] = {0, 0};
    for (Bitboard b = pieces() & ~pieces(PAWN); b;)
    {
        const Square s = pop_lsb(b);
        np[color_of(piece_on(s))] ^= Zobrist::psq[piece_on(s)][s];
    }
    if (np[WHITE] != st->nonPawnKey[WHITE] || np[BLACK] != st->nonPawnKey[BLACK])
        return false;

    return true;
#endif
}

}  // namespace one
