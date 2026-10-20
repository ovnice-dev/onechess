#include "tt.h"
#include "prof.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace one {

TranspositionTable TT;

// Ecrit l'entree. Un coup n'est jamais efface par une entree qui n'en a pas :
// meme sans score exploitable, il reste la meilleure indication de tri.
void TTEntry::save(
  Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, std::uint8_t gen8) {
    PROF_SCOPE(TTSAVE);

    // Une evaluation deja corrigee stockee ici se ferait recorriger a chaque
    // relecture : le terme se composerait jusqu'a saturer. Seule l'evaluation
    // brute a le droit d'entrer.
    assert(ev == VALUE_NONE || std::abs(ev) < 20000);

    if (m || std::uint16_t(k) != key16)
        move16 = m.raw();

    // On ne remplace que si l'entree apporte davantage : borne exacte, cle
    // differente, ou profondeur superieure d'au moins 4 plis.
    if (b == BOUND_EXACT || std::uint16_t(k) != key16 || d - DEPTH_ENTRY_TT + 2 * pv > depth8 - 4
        || relative_age(gen8))
    {
        assert(d > DEPTH_ENTRY_TT);
        assert(d < 256 + DEPTH_ENTRY_TT);

        key16     = std::uint16_t(k);
        depth8    = std::uint8_t(d - DEPTH_ENTRY_TT);
        genBound8 = std::uint8_t(gen8 | std::uint8_t(pv) << 2 | b);
        value16   = std::int16_t(v);
        eval16    = std::int16_t(ev);
    }
}

std::uint8_t TTEntry::relative_age(std::uint8_t generation8) const {
    // TT_GENERATION_CYCLE compense le debordement du compteur 8 bits, pour que
    // la soustraction reste correcte quand la generation vient de reboucler.
    return (TT_GENERATION_CYCLE + generation8 - genBound8) & TT_GENERATION_MASK;
}

void TranspositionTable::resize(std::size_t mbSize, int threadCount) {
    aligned_large_pages_free(table);

    clusterCount = mbSize * 1024 * 1024 / sizeof(Cluster);

    table = static_cast<Cluster*>(aligned_large_pages_alloc(clusterCount * sizeof(Cluster)));
    if (!table)
    {
        clusterCount = 0;
        return;
    }

    clear(threadCount);
}

// Efface en parallele : sur une table de 1 Go, un seul thread passe plusieurs
// secondes a memset avant le premier coup.
void TranspositionTable::clear(int threadCount) {
    if (!table)
        return;

    std::vector<std::thread> workers;

    for (int idx = 0; idx < threadCount; ++idx)
        workers.emplace_back([this, idx, threadCount]() {
            const std::size_t stride = clusterCount / threadCount;
            const std::size_t start  = stride * idx;
            const std::size_t len =
              idx != threadCount - 1 ? stride : clusterCount - start;

            std::memset(&table[start], 0, len * sizeof(Cluster));
        });

    for (auto& th : workers)
        th.join();

    generation8 = 0;
}

// Cherche la cle dans son cluster. En cas d'echec, rend l'entree a remplacer :
// celle dont la profondeur, corrigee de l'age, est la plus faible.
TTEntry* TranspositionTable::probe(const Key key, bool& found) const {
    PROF_SCOPE(TTPROBE);
    TTEntry* const      tte   = first_entry(key)->entry;
    const std::uint16_t key16 = std::uint16_t(key);

    for (std::size_t i = 0; i < TT_CLUSTER_SIZE; ++i)
        if (tte[i].key16 == key16 || !tte[i].depth8)
        {
            tte[i].genBound8 =
              std::uint8_t(generation8 | (tte[i].genBound8 & (TT_GENERATION_DELTA - 1)));

            found = bool(tte[i].depth8);
            return &tte[i];
        }

    TTEntry* replace = tte;
    for (std::size_t i = 1; i < TT_CLUSTER_SIZE; ++i)
        // Une entree vieille d'une generation vaut environ 8 plis de
        // profondeur : c'est ce facteur qui evite qu'une entree profonde mais
        // perimee ne squatte le cluster indefiniment.
        if (replace->depth8 - replace->relative_age(generation8) * 2
            > tte[i].depth8 - tte[i].relative_age(generation8) * 2)
            replace = &tte[i];

    found = false;
    return replace;
}

// Taux de remplissage en pour mille, estime sur les 1000 premieres entrees.
int TranspositionTable::hashfull() const {
    if (!table)
        return 0;

    int cnt = 0;
    for (int i = 0; i < 1000; ++i)
        for (std::size_t j = 0; j < TT_CLUSTER_SIZE; ++j)
            cnt += table[i].entry[j].depth8
                && (table[i].entry[j].genBound8 & TT_GENERATION_MASK) == generation8;

    return cnt / TT_CLUSTER_SIZE;
}

}  // namespace one
