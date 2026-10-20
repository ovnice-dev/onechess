#pragma once

#include <cstdint>

#include "types.h"

namespace one {
namespace NNUE {

constexpr int INPUT_SIZE  = 768;
// Taille de la couche cachee. Fixee a la compilation : l'accumulateur et
// toutes les boucles SIMD en dependent. Pour essayer une autre taille,
// compiler un binaire a part :
//   EXTRA=-DONE_HIDDEN_SIZE=1024 OUT=bin/onechess1024.exe ./build.sh release
// Le chargeur refuse tout reseau dont l'en-tete annonce une autre taille.
#ifndef ONE_HIDDEN_SIZE
    #define ONE_HIDDEN_SIZE 512
#endif
constexpr int HIDDEN_SIZE = ONE_HIDDEN_SIZE;

// Nombre d'accumulateurs detenus par une Position. La recherche repart de
// l'index 0 a chaque racine, et ne descend jamais au-dela de MAX_PLY.
constexpr int ACC_POOL = MAX_PLY + 18;

// Une somme par perspective. C'est cet objet que la recherche transporte de
// noeud en noeud au lieu de le recalculer. Il vit dans un reservoir porte par
// la Position, et NON dans StateInfo : un StateInfo de 2,3 Ko pose sur la pile
// a chaque noeud coute 25 % du NPS, meme quand aucun reseau n'est charge.
struct alignas(64) Accumulator {
    std::int16_t v[COLOR_NB][HIDDEN_SIZE];
};

}  // namespace NNUE

// Ce qui a bouge pendant un coup, du point de vue du reseau. Trois entrees
// suffisent : le roque deplace deux pieces, une promotion capturante en fait
// disparaitre deux et apparaitre une.
//
//   from == SQ_NONE : la piece apparait   (promotion)
//   to   == SQ_NONE : la piece disparait  (capture, ou pion promu)
struct DirtyPiece {
    int    count;
    Piece  piece[3];
    Square from[3];
    Square to[3];
};

}  // namespace one
