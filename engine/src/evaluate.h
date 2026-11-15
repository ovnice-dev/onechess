#pragma once

#include <string>

#include "types.h"

namespace one {

class Position;

namespace Eval {

void  init();
Value evaluate(const Position& pos);

// Valeur de la position pour le camp au trait, avant tout ajustement de
// recherche. Sert au diagnostic et aux tests.
std::string trace(const Position& pos);

}  // namespace Eval
}  // namespace one
