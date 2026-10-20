#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include "types.h"

namespace one {

// Une case de table d'historique. La mise a jour est un amortissement
// multiplicatif :
//
//     entry += bonus - entry * |bonus| / D
//
// La valeur converge vers +D pour des bonus repetes et vers -D pour des malus,
// sans jamais deborder. Une seule formule pour les deux sens : le malus n'a
// pas besoin d'un traitement separe, c'est le meme point fixe pris de l'autre
// cote.
template<typename T, int D>
class StatsEntry {
    T entry;

   public:
    void operator=(const T& v) { entry = v; }
    T*   operator&() { return &entry; }
    T*   operator->() { return &entry; }
    operator const T&() const { return entry; }

    void operator<<(int bonus) {
        static_assert(D <= std::numeric_limits<T>::max(), "D deborde le type de la table");

        bonus = std::clamp(bonus, -D, D);
        entry += T(bonus - int(entry) * std::abs(bonus) / D);

        assert(std::abs(int(entry)) <= D);
    }
};

// Tableau multidimensionnel de StatsEntry. L'heritage de std::array donne
// l'indexation naturelle ; fill() remplit a plat, ce qui est correct parce que
// StatsEntry<T, D> a exactement la disposition memoire de T.
template<typename T, int D, int Size, int... Sizes>
struct Stats: public std::array<Stats<T, D, Sizes...>, Size> {
    using stats = Stats<T, D, Size, Sizes...>;

    void fill(const T& v) {
        static_assert(std::is_standard_layout_v<stats>, "disposition memoire non garantie");

        using entry = StatsEntry<T, D>;
        entry* p    = reinterpret_cast<entry*>(this);
        std::fill(p, p + sizeof(*this) / sizeof(entry), v);
    }
};

template<typename T, int D, int Size>
struct Stats<T, D, Size>: public std::array<StatsEntry<T, D>, Size> {};

// D = 0 : la table stocke une valeur brute, sans amortissement.
constexpr int NOT_USED = 0;

// ------------------------------------------------------- tables concretes
//
// Chaque table a sa propre borne D. Les regrouper derriere une seule constante
// ferait passer plusieurs changements pour un seul, et rendrait le resultat
// d'un test impossible a attribuer.

// [couleur][depart * 64 + arrivee] : "ce coup tranquille marche-t-il ?"
// Avec ThreatHist, quatre fois la table : la case de depart et celle d'arrivee
// sont-elles attaquees par l'adversaire ? Sans, seul le premier quart sert, et
// le jeu est strictement celui d'avant. Voir hist_index() dans movepick.h.
using ButterflyHistory = Stats<std::int16_t, 7183, COLOR_NB, 4 * int(SQUARE_NB) * int(SQUARE_NB)>;

// [ply][depart * 64 + arrivee] : le meme historique, propre aux premiers
// niveaux de l'arbre et remis a zero a chaque recherche. Pres de la racine, les
// noeuds sont les memes d'une iteration a l'autre : ce qui y a coupe la fois
// precedente vaut plus qu'une moyenne sur tout l'arbre. Repris de Stockfish.
constexpr int LOW_PLY_HISTORY_SIZE = 5;
using LowPlyHistory =
  Stats<std::int16_t, 7183, LOW_PLY_HISTORY_SIZE, int(SQUARE_NB) * int(SQUARE_NB)>;

// [piece qui prend][case][type de piece prise] : tri fin des captures, la ou
// MVV-LVA ne departage plus.
using CapturePieceToHistory = Stats<std::int16_t, 10692, PIECE_NB, SQUARE_NB, PIECE_TYPE_NB>;

// [piece][case] : brique de l'historique de continuation.
using PieceToHistory = Stats<std::int16_t, 29952, PIECE_NB, SQUARE_NB>;

// [piece precedente][case precedente] -> PieceToHistory :
// "ce coup repond-il bien au coup precedent ?"
using ContinuationHistory = Stats<PieceToHistory, NOT_USED, PIECE_NB, SQUARE_NB>;

// [piece][case] -> coup : refutation memorisee du coup precedent.
using CounterMoveHistory = Stats<Move, NOT_USED, PIECE_NB, SQUARE_NB>;

// [structure de pions][piece][case] : ce qui marche dans ce type de position.
constexpr int PAWN_HISTORY_SIZE = 512;
using PawnHistory = Stats<std::int16_t, 8192, PAWN_HISTORY_SIZE, PIECE_NB, SQUARE_NB>;

inline int pawn_structure_index(Key pawnKey) {
    return int(pawnKey & (PAWN_HISTORY_SIZE - 1));
}

// [trait][structure de pions] : ecart appris entre evaluation statique et
// score reellement trouve par la recherche.
constexpr int CORRECTION_HISTORY_SIZE  = 16384;
constexpr int CORRECTION_HISTORY_LIMIT = 1024;
using CorrectionHistory =
  Stats<std::int16_t, CORRECTION_HISTORY_LIMIT, COLOR_NB, CORRECTION_HISTORY_SIZE>;

inline int correction_index(Key pawnKey) {
    return int(pawnKey & (CORRECTION_HISTORY_SIZE - 1));
}

// [piece][case] du coup d'avant -> [piece][case] du coup precedent : ecart
// appris apres une paire de coups donnee. Repris de Stockfish.
using PieceToCorrectionHistory = Stats<std::int16_t, CORRECTION_HISTORY_LIMIT, PIECE_NB, SQUARE_NB>;
using ContinuationCorrectionHistory =
  Stats<PieceToCorrectionHistory, NOT_USED, PIECE_NB, SQUARE_NB>;

}  // namespace one
