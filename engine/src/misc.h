#pragma once

#include <chrono>
#include <cstdint>
#include <string>

#if defined(_WIN32)
    #include <intrin.h>
#else
    #include <xmmintrin.h>
#endif

namespace one {

using TimePoint = std::chrono::milliseconds::rep;
static_assert(sizeof(TimePoint) == sizeof(std::int64_t), "TimePoint doit tenir sur 64 bits");

inline TimePoint now() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Charge une ligne de cache a l'avance. La sonde de table de transposition
// est la lecture la plus couteuse de la recherche : la lancer un coup plus
// tot masque la latence memoire derriere le travail utile.
inline void prefetch(const void* addr) {
#if defined(_WIN32)
    _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0);
#else
    __builtin_prefetch(addr);
#endif
}

// xorshift64* : le seul generateur du moteur, utilise pour les magics, le
// Zobrist et la diversification des threads.
class PRNG {
    std::uint64_t s;

   public:
    explicit PRNG(std::uint64_t seed) : s(seed) {}

    std::uint64_t rand() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }

    std::uint64_t sparse_rand() { return rand() & rand() & rand(); }
};

void* aligned_large_pages_alloc(std::size_t allocSize);
void  aligned_large_pages_free(void* mem);

std::string engine_info();

}  // namespace one
