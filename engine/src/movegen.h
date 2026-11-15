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

// Generation des coups (reecrite pour Onechess le 15-11-2026).

#pragma once

#include <algorithm>
#include <cstddef>

#include "types.h"

namespace one {

class Position;

// Etages de generation
enum GenType {
    CAPTURES,      // prises, et promotions en dame
    QUIETS,        // le reste (dont les sous-promotions par poussee)
    EVASIONS,      // le camp au trait est en echec
    NON_EVASIONS,  // tout, hors echec
    LEGAL          // coups legaux seulement
};

// Coup accompagne de sa note de tri (le tri des coups la remplit)
struct ExtMove : public Move {
    int value;

    void  operator=(Move coup) { data = coup.raw(); }
    operator float() const   = delete;  // pas de conversion numerique cachee
};

inline bool operator<(const ExtMove& a, const ExtMove& b) { return a.value < b.value; }

// Ecrit les coups a partir de « sortie » et rend la fin de la liste
template<GenType>
ExtMove* generate(const Position& pos, ExtMove* sortie);

// Liste de coups prete a parcourir : for (auto m : MoveList<LEGAL>(pos))
template<GenType T>
class MoveList {
   public:
    explicit MoveList(const Position& pos) : fin(generate<T>(pos, coups)) {}

    const ExtMove* begin() const { return coups; }
    const ExtMove* end() const { return fin; }
    std::size_t    size() const { return std::size_t(fin - coups); }
    bool           contains(Move m) const { return std::find(begin(), end(), m) != fin; }

   private:
    ExtMove  coups[MAX_MOVES];
    ExtMove* fin;
};

}  // namespace one
