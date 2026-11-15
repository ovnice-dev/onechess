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

// Tables d'historique de la recherche (reecrites pour Onechess le 15-11-2026).

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

// Une note apprise, bornee par B. « note << bonus » la rapproche de +B (bonus
// positif) ou de -B (malus), d'autant moins qu'elle en est deja proche :
//     note += bonus - note * |bonus| / B
// Elle ne peut donc jamais sortir de [-B, B]. B = 0 : simple valeur stockee.
template<typename T, int B>
class Note {
   public:
    Note& operator=(const T& v) {
        val = v;
        return *this;
    }
    operator const T&() const { return val; }
    T* operator&() { return &val; }
    T* operator->() { return &val; }  // pour les tables de tables

    void operator<<(int bonus) {
        static_assert(B > 0 && B <= std::numeric_limits<T>::max(), "borne hors du type");
        const int b = std::clamp(bonus, -B, B);
        val         = T(val + b - int(val) * std::abs(b) / B);
        assert(std::abs(int(val)) <= B);
    }

   private:
    T val;
};

// Table a plusieurs dimensions de Notes : Table<T, B, 2, 64> se lit t[c][s].
template<typename T, int B, int N, int... Reste>
struct Table: std::array<Table<T, B, Reste...>, N> {
    // Remplit toutes les cases (une Note a la meme disposition qu'un T)
    void fill(const T& v) {
        static_assert(std::is_standard_layout_v<Table> && sizeof(Note<T, B>) == sizeof(T));
        auto* d = reinterpret_cast<Note<T, B>*>(this);
        std::fill(d, d + sizeof(Table) / sizeof(Note<T, B>), v);
    }
};
template<typename T, int B, int N>
struct Table<T, B, N>: std::array<Note<T, B>, N> {
    void fill(const T& v) {
        for (auto& n : *this)
            n = v;
    }
};

constexpr int NOT_USED = 0;  // borne des tables de simples valeurs

// --------------------------------------------------------- tables du moteur
// Chaque table a sa borne propre (reglee separement).

// [camp][index du coup] : le coup tranquille a-t-il coupe ? Quatre plans selon
// que ses cases de depart et d'arrivee sont menacees (hist_index, movepick.h).
using ButterflyHistory = Table<std::int16_t, 7183, COLOR_NB, 4 * 64 * 64>;

// [ply][depart*64 + arrivee] : la meme chose pres de la racine, ou les noeuds
// se retrouvent d'une iteration a l'autre (remise a zero a chaque recherche).
constexpr int LOW_PLY_HISTORY_SIZE = 5;
using LowPlyHistory = Table<std::int16_t, 7183, LOW_PLY_HISTORY_SIZE, 64 * 64>;

// [piece qui prend][case][type pris] : departage les prises de meme valeur.
using CapturePieceToHistory = Table<std::int16_t, 10692, PIECE_NB, SQUARE_NB, PIECE_TYPE_NB>;

// [piece][case] apres un coup donne ; ContinuationHistory[piece][case] du coup
// precedent donne la table a lire pour la reponse.
using PieceToHistory      = Table<std::int16_t, 29952, PIECE_NB, SQUARE_NB>;
using ContinuationHistory = Table<PieceToHistory, NOT_USED, PIECE_NB, SQUARE_NB>;

// [piece][case] -> reponse memorisee au coup precedent
using CounterMoveHistory = Table<Move, NOT_USED, PIECE_NB, SQUARE_NB>;

// [structure de pions][piece][case]
#ifndef ONE_PAWN_HIST
    #define ONE_PAWN_HIST 512
#endif
constexpr int PAWN_HISTORY_SIZE = ONE_PAWN_HIST;  // -DONE_PAWN_HIST pour essayer d'autres tailles
using PawnHistory               = Table<std::int16_t, 8192, PAWN_HISTORY_SIZE, PIECE_NB, SQUARE_NB>;
inline int pawn_structure_index(Key cle) { return int(cle % unsigned(PAWN_HISTORY_SIZE)); }

// Corrections de l'evaluation : [trait][cle de structure] = ecart appris
// entre l'evaluation et le score de la recherche.
#ifndef ONE_CORR_HIST
    #define ONE_CORR_HIST 16384
#endif
constexpr int CORRECTION_HISTORY_SIZE  = ONE_CORR_HIST;
constexpr int CORRECTION_HISTORY_LIMIT = 1024;
using CorrectionHistory = Table<std::int16_t, CORRECTION_HISTORY_LIMIT, COLOR_NB, CORRECTION_HISTORY_SIZE>;
inline int correction_index(Key cle) { return int(cle % unsigned(CORRECTION_HISTORY_SIZE)); }

// Correction apres une paire de coups : [piece][case] du coup d'avant, puis du precedent
using PieceToCorrectionHistory      = Table<std::int16_t, CORRECTION_HISTORY_LIMIT, PIECE_NB, SQUARE_NB>;
using ContinuationCorrectionHistory = Table<PieceToCorrectionHistory, NOT_USED, PIECE_NB, SQUARE_NB>;

// Anciens noms, gardes pour le reste du moteur
template<typename T, int B>
using StatsEntry = Note<T, B>;

}  // namespace one
