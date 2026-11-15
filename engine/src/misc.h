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

// Outils divers (reecrits pour Onechess le 15-11-2026).

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace one {

// Temps en millisecondes depuis une origine fixe (horloge monotone)
using TimePoint = std::int64_t;
inline TimePoint now() {
    using namespace std::chrono;
    return TimePoint(duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count());
}

// Demande au processeur de charger cette adresse dans le cache (sans attendre)
inline void prefetch(const void* adresse) { __builtin_prefetch(adresse); }

// Grands blocs (table de transposition, reseau) : alignes, et en grandes
// pages quand le systeme le permet (moins de defauts de TLB).
void* aligned_large_pages_alloc(std::size_t octets);
void  aligned_large_pages_free(void* bloc);

std::string engine_info();  // nom et version

// Option « Debug Log File » : copie l'entree et la sortie UCI dans un fichier
// (nom vide : arret).
void start_logger(const std::string& fichier);

}  // namespace one
