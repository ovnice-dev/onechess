#pragma once

#include <cstddef>
#include <cstdint>

#include "misc.h"
#include "types.h"

namespace one {

// Entree de 10 octets. Trois entrees plus deux octets de bourrage tiennent
// dans 32 octets, soit une demi-ligne de cache : une sonde ne touche jamais
// deux lignes.
struct TTEntry {
    Move  move() const { return Move(move16); }
    Value value() const { return Value(value16); }
    Value eval() const { return Value(eval16); }
    Depth depth() const { return Depth(depth8) + DEPTH_ENTRY_TT; }
    bool  is_pv() const { return bool(genBound8 & 0x4); }
    Bound bound() const { return Bound(genBound8 & 0x3); }

    void save(Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, std::uint8_t gen8);

    // Age de l'entree en nombre de recherches, module le cycle du compteur.
    std::uint8_t relative_age(std::uint8_t generation8) const;

   private:
    friend class TranspositionTable;

    std::uint16_t key16;
    std::uint8_t  depth8;
    std::uint8_t  genBound8;
    std::int16_t  value16;
    std::int16_t  eval16;
    std::uint16_t move16;
};

// Bits de genBound8 : 0-1 borne, 2 drapeau PV, 3-7 generation.
constexpr unsigned      TT_GENERATION_BITS  = 3;
constexpr int           TT_GENERATION_DELTA = 1 << TT_GENERATION_BITS;
constexpr int           TT_GENERATION_CYCLE = 255 + TT_GENERATION_DELTA;
constexpr unsigned      TT_GENERATION_MASK  = (0xFF << TT_GENERATION_BITS) & 0xFF;
constexpr std::size_t   TT_CLUSTER_SIZE     = 3;

struct Cluster {
    TTEntry entry[TT_CLUSTER_SIZE];
    char    padding[2];
};

static_assert(sizeof(Cluster) == 32, "un cluster doit faire exactement 32 octets");

class TranspositionTable {
   public:
    ~TranspositionTable() { aligned_large_pages_free(table); }

    void     new_search() { generation8 += TT_GENERATION_DELTA; }
    TTEntry* probe(const Key key, bool& found) const;
    int      hashfull() const;
    void     resize(std::size_t mbSize, int threadCount);
    void     clear(int threadCount);

    std::uint8_t generation() const { return generation8; }

    // Cluster ou vit la cle : multiplication haute plutot que modulo, ce qui
    // repartit uniformement sans exiger une taille en puissance de deux.
    Cluster* first_entry(const Key key) const {
        return &table[std::size_t((__uint128_t(key) * __uint128_t(clusterCount)) >> 64)];
    }

   private:
    std::size_t  clusterCount = 0;
    Cluster*     table        = nullptr;
    std::uint8_t generation8  = 0;
};

extern TranspositionTable TT;

}  // namespace one
