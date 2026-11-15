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

// Point d'entree (reecrit pour Onechess le 15-11-2026).

#include <iostream>

#include "bitboard.h"
#include "evaluate.h"
#include "nnue.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"

int main(int argc, char* argv[]) {
    using namespace one;
    std::cout << engine_info() << std::endl;

    // Tables precalculees, dans l'ordre de leurs dependances
    Bitboards::init();
    Position::init();
    Eval::init();
    Search::init();
    UCI::init(Options);

    // Reseau par defaut, a cote de l'executable ; sans lui, evaluation simple
    NNUE::load("onechess.nnue");
    std::cout << NNUE::info() << std::endl;

    const int fils = int(Options["Threads"]);
    Threads.set(std::size_t(fils));
    TT.resize(std::size_t(int(Options["Hash"])), fils);

    UCI::loop(argc, argv);  // jusqu'a « quit »
    Threads.set(0);
    return 0;
}
