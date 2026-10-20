#include "syzygy.h"

#include <algorithm>

#include "bitboard.h"
#include "movegen.h"
#include "position.h"

// tbprobe.h porte deja son propre extern "C" et inclut <tbconfig.h> entre
// chevrons : le dossier fathom doit etre dans le chemin d'inclusion.
#include "tbprobe.h"

namespace one {
namespace Syzygy {

namespace {

std::string TbPath;
bool        Loaded = false;

// Fathom veut la case en prise en passant, ou 0 s'il n'y en a pas. Notre
// SQ_NONE n'est pas 0 : la case a1 existe et ne peut jamais etre une case de
// prise en passant, donc 0 est bien un "aucune" sans ambiguite.
inline unsigned ep_of(const Position& pos) {
    const Square e = pos.state()->epSquare;
    return e == SQ_NONE ? 0u : unsigned(e);
}

// Les quatre bits de roque de Fathom sont exactement les notres :
// K=1, Q=2, k=4, q=8. Aucune traduction n'est necessaire, et un
// static_assert le garantit si les valeurs changent un jour.
static_assert(WHITE_OO == TB_CASTLING_K && WHITE_OOO == TB_CASTLING_Q
                && BLACK_OO == TB_CASTLING_k && BLACK_OOO == TB_CASTLING_q,
              "les droits de roque ne correspondent plus a ceux de Fathom");

inline unsigned castling_of(const Position& pos) {
    return unsigned(pos.can_castle(WHITE_OO)) * TB_CASTLING_K
         | unsigned(pos.can_castle(WHITE_OOO)) * TB_CASTLING_Q
         | unsigned(pos.can_castle(BLACK_OO)) * TB_CASTLING_k
         | unsigned(pos.can_castle(BLACK_OOO)) * TB_CASTLING_q;
}

}  // namespace

void init(const std::string& p) {
    free();

    // Une ressource externe absente doit desactiver la fonction, jamais
    // empecher le moteur de demarrer.
    if (p.empty() || p == "<empty>")
        return;

    Loaded = tb_init(p.c_str());
    if (Loaded && TB_LARGEST > 0)
        TbPath = p;
    else
    {
        tb_free();
        Loaded = false;
    }
}

void free() {
    if (Loaded)
        tb_free();
    Loaded = false;
    TbPath.clear();
}

int max_pieces() { return Loaded ? int(TB_LARGEST) : 0; }

const std::string& path() { return TbPath; }

bool probe_wdl(const Position& pos, int ply, bool use50MoveRule, Value* v) {
    if (!Loaded)
        return false;

    // La table WDL suppose le compteur des 50 coups a zero et aucun droit de
    // roque : hors de ces conditions le resultat ne serait pas celui de la
    // position reellement jouee.
    if (pos.rule50_count() != 0 || castling_of(pos) != 0
        || popcount(pos.pieces()) > int(TB_LARGEST))
        return false;

    const unsigned r = tb_probe_wdl(
      pos.pieces(WHITE), pos.pieces(BLACK), pos.pieces(KING), pos.pieces(QUEEN),
      pos.pieces(ROOK), pos.pieces(BISHOP), pos.pieces(KNIGHT), pos.pieces(PAWN),
      unsigned(pos.rule50_count()), castling_of(pos), ep_of(pos),
      pos.side_to_move() == WHITE);

    if (r == TB_RESULT_FAILED)
        return false;

    // Un gain "maudit" est un gain que la regle des 50 coups annule. Il compte
    // comme une nulle tant que la regle s'applique -- et seulement alors.
    switch (r)
    {
    case TB_LOSS :
        *v = -VALUE_TB + ply;
        break;
    case TB_BLESSED_LOSS :
        *v = use50MoveRule ? VALUE_DRAW : -VALUE_TB + ply;
        break;
    case TB_CURSED_WIN :
        *v = use50MoveRule ? VALUE_DRAW : VALUE_TB - ply;
        break;
    case TB_WIN :
        *v = VALUE_TB - ply;
        break;
    default :
        *v = VALUE_DRAW;
    }
    return true;
}

Move probe_root(const Position& pos, bool use50MoveRule, int* wdl) {
    if (!Loaded || popcount(pos.pieces()) > int(TB_LARGEST))
        return Move::none();

    const unsigned r = tb_probe_root(
      pos.pieces(WHITE), pos.pieces(BLACK), pos.pieces(KING), pos.pieces(QUEEN),
      pos.pieces(ROOK), pos.pieces(BISHOP), pos.pieces(KNIGHT), pos.pieces(PAWN),
      unsigned(pos.rule50_count()), castling_of(pos), ep_of(pos),
      pos.side_to_move() == WHITE, nullptr);

    if (r == TB_RESULT_FAILED || r == TB_RESULT_CHECKMATE || r == TB_RESULT_STALEMATE)
        return Move::none();

    *wdl = int(TB_GET_WDL(r));
    if (!use50MoveRule)
    {
        if (*wdl == TB_CURSED_WIN)
            *wdl = TB_WIN;
        else if (*wdl == TB_BLESSED_LOSS)
            *wdl = TB_LOSS;
    }

    // Plutot que de reconstruire un coup a partir des champs de Fathom -- ce
    // qui demanderait de deviner notre encodage de la prise en passant et du
    // roque -- on cherche le coup correspondant parmi les notres. Un coup qui
    // ne se retrouve pas est un coup qu'on ne jouera pas : c'est le sens le
    // plus sur de l'erreur.
    const Square    from  = Square(TB_GET_FROM(r));
    const Square    to    = Square(TB_GET_TO(r));
    const unsigned  promo = TB_GET_PROMOTES(r);

    static constexpr PieceType PROMO_OF[5] = {NO_PIECE_TYPE, QUEEN, ROOK, BISHOP, KNIGHT};

    for (const auto& m : MoveList<LEGAL>(pos))
        if (m.from_sq() == from && m.to_sq() == to
            && (promo == TB_PROMOTES_NONE
                  ? m.type_of() != PROMOTION
                  : m.type_of() == PROMOTION && m.promotion_type() == PROMO_OF[promo]))
            return m;

    return Move::none();
}

}  // namespace Syzygy
}  // namespace one
