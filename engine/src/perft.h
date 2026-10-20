#pragma once

#include <cstdint>

#include "position.h"
#include "types.h"

namespace one {

// Compte les feuilles a profondeur fixe. C'est la seule mesure du moteur qui
// prouve plutot qu'elle n'estime : un total juste sur les positions pieges
// garantit que la generation de coups est exacte.
std::uint64_t perft(Position& pos, Depth depth);
void          perft_divide(Position& pos, Depth depth);
int           perft_suite();

}  // namespace one
