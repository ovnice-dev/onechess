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

// Gestion du temps (reecrite pour Onechess le 15-11-2026). Deux budgets par
// coup : une cible, qu'une iteration commencee peut depasser, et un plafond
// qu'on ne franchit jamais.

#pragma once

#include "misc.h"
#include "search.h"
#include "types.h"

namespace one {

class TimeManagement {
   public:
    void init(Search::LimitsType& limites, Color moi, int ply, int marge, int pieces = 32);
    void new_game() { tailleControle = 0; }
    int  tailleControle = 0;  // plus grand « movestogo » vu dans la partie (taille d'un controle)

    TimePoint optimum() const { return cible; }
    TimePoint maximum() const { return plafond; }
    TimePoint elapsed() const { return now() - debut; }

   private:
    TimePoint debut = 0, cible = 0, plafond = 0;
};

extern TimeManagement Time;

}  // namespace one
