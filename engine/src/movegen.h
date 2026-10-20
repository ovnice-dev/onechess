#pragma once

#include <algorithm>
#include <cstddef>

#include "types.h"

namespace one {

class Position;

enum GenType {
    CAPTURES,      // prises et promotions en dame
    QUIETS,        // tout le reste
    EVASIONS,      // parades a un echec
    NON_EVASIONS,  // tous les coups, hors echec
    LEGAL          // strictement legaux, filtres
};

// Un coup plus son score de tri. Herite de Move pour rester assignable et
// comparable sans conversion explicite.
struct ExtMove : public Move {
    int value;

    void operator=(Move m) { data = m.raw(); }

    // Empeche une conversion implicite silencieuse vers un flottant
    operator float() const = delete;
};

inline bool operator<(const ExtMove& f, const ExtMove& s) { return f.value < s.value; }

template<GenType>
ExtMove* generate(const Position& pos, ExtMove* moveList);

// Enveloppe pratique : construit la liste et l'expose comme un conteneur.
template<GenType T>
struct MoveList {
    explicit MoveList(const Position& pos) : last(generate<T>(pos, moveList)) {}

    const ExtMove* begin() const { return moveList; }
    const ExtMove* end() const { return last; }
    std::size_t    size() const { return std::size_t(last - moveList); }

    bool contains(Move move) const { return std::find(begin(), end(), move) != end(); }

   private:
    ExtMove moveList[MAX_MOVES], *last;
};

}  // namespace one
