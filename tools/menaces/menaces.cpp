// Indexeur des menaces completes (phase 10, C) pour l'entraineur.
// Entree : positions de 32 octets (format de binpack_vers_bin : occupation sur
// 8 octets, un quartet par piece = couleur * 6 + type, dans l'ordre des cases).
// Sortie : pour chaque position et chaque perspective (blancs, noirs), K indices
// d'entrees actives, completes par DIM (= entree nulle). Memes indices que
// tools/menaces_completes.py (la reference), verifie par menaces_completes.py --cpp.
//   g++ -O3 -std=c++17 -shared -fPIC -pthread menaces.cpp -o libmenaces.so
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr int DIM = 59808;
enum { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };
constexpr int NB_CIBLES[6] = {4, 10, 8, 8, 10, 0};
constexpr int CARTE[6][6] = {{-1, 0, -1, 1, -1, -1}, {0, 1, 2, 3, 4, -1}, {0, 1, 2, 3, -1, -1},
                             {0, 1, 2, 3, -1, -1},   {0, 1, 2, 3, 4, -1}, {-1, -1, -1, -1, -1, -1}};

uint32_t LUT1[12][12][2];   // [attaquant][attaque][from < to], piece = couleur * 6 + type
uint16_t LUT2[6][64][64];   // [type][from][to]
uint64_t ATT_VIDE[6][64];   // attaques sur echiquier vide (pion : blanc)
uint64_t ATT_PION[2][64];

inline int pc(uint64_t b) { return __builtin_popcountll(b); }

uint64_t pas(int sq, int df, int dr, uint64_t occ, bool glisse) {
    uint64_t r = 0;
    int f = sq % 8 + df, k = sq / 8 + dr;
    while (f >= 0 && f < 8 && k >= 0 && k < 8) {
        int s = k * 8 + f;
        r |= 1ULL << s;
        if (!glisse || (occ >> s & 1)) break;
        f += df; k += dr;
    }
    return r;
}

uint64_t attaques(int t, int sq, uint64_t occ, int couleur) {
    static const int DC[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    static const int DR[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    static const int DD[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    uint64_t r = 0;
    switch (t) {
    case PAWN: return ATT_PION[couleur][sq];
    case KNIGHT: for (auto& d : DC) r |= pas(sq, d[0], d[1], occ, false); return r;
    case KING:
        for (auto& d : DR) r |= pas(sq, d[0], d[1], occ, false);
        for (auto& d : DD) r |= pas(sq, d[0], d[1], occ, false);
        return r;
    case BISHOP: for (auto& d : DD) r |= pas(sq, d[0], d[1], occ, true); return r;
    case ROOK: for (auto& d : DR) r |= pas(sq, d[0], d[1], occ, true); return r;
    default:
        for (auto& d : DR) r |= pas(sq, d[0], d[1], occ, true);
        for (auto& d : DD) r |= pas(sq, d[0], d[1], occ, true);
        return r;
    }
}

struct Init {
    Init() {
        for (int sq = 0; sq < 64; ++sq) {
            ATT_PION[0][sq] = pas(sq, 1, 1, 0, false) | pas(sq, -1, 1, 0, false);
            ATT_PION[1][sq] = pas(sq, 1, -1, 0, false) | pas(sq, -1, -1, 0, false);
            for (int t = 0; t < 6; ++t) ATT_VIDE[t][sq] = attaques(t, sq, 0, 0);
        }
        int decal[6];
        for (int t = 0; t < 6; ++t) {
            int off = 0;
            for (int fr = 0; fr < 64; ++fr) {
                for (int to = 0; to < 64; ++to) {
                    uint64_t a = t != PAWN ? ATT_VIDE[t][fr] : ATT_PION[to < fr ? 1 : 0][fr];
                    LUT2[t][fr][to] = uint16_t(off + pc(((1ULL << to) - 1) & a));
                }
                if (t != PAWN) off += pc(ATT_VIDE[t][fr]);
                else if (fr >= 8 && fr <= 55) off += pc(ATT_PION[0][fr]);
            }
            decal[t] = off;
        }
        int cumul = 0;
        for (int a = 0; a < 12; ++a) {
            int ta = a % 6, cp = decal[ta], nv = NB_CIBLES[ta];
            for (int d = 0; d < 12; ++d) {
                int td = d % 6, m = CARTE[ta][td];
                bool ennemi = (a / 6) != (d / 6);
                bool semi = ta == td && (ennemi || ta != PAWN);
                uint32_t f = uint32_t(cumul + ((d / 6) * (nv / 2) + m) * cp);
                bool exclu = m < 0;
                LUT1[a][d][0] = exclu ? DIM : f;
                LUT1[a][d][1] = (exclu || semi) ? DIM : f;
            }
            cumul += nv * cp;
        }
    }
} init_;

// Une position -> indices des deux perspectives ; rend le nombre d'indices (avant troncature a K)
int une(const uint8_t* r, int K, uint16_t* w, uint16_t* b) {
    uint64_t occ;
    std::memcpy(&occ, r, 8);
    int piece[64];
    int n = 0;
    for (uint64_t o = occ; o; o &= o - 1) {
        int sq = __builtin_ctzll(o);
        int q = (r[8 + n / 2] >> (4 * (n & 1))) & 15;
        piece[sq] = q;
        ++n;
    }
    int k = 0;
    for (uint64_t o = occ; o; o &= o - 1) {
        int fr = __builtin_ctzll(o), a = piece[fr], ta = a % 6;
        if (ta == KING) continue;
        uint64_t att = attaques(ta, fr, occ, a / 6) & occ;
        for (; att; att &= att - 1) {
            int to = __builtin_ctzll(att), d = piece[to];
            // perspective blanche : telle quelle ; noire : echiquier retourne (^56), couleurs echangees
            uint32_t iw = LUT1[a][d][fr < to] + LUT2[ta][fr][to];
            int fb = fr ^ 56, tb = to ^ 56, ab = (a + 6) % 12, db = (d + 6) % 12;
            uint32_t ib = LUT1[ab][db][fb < tb] + LUT2[ta][fb][tb];
            if (iw < DIM || ib < DIM) {
                if (k < K) { w[k] = uint16_t(iw < DIM ? iw : DIM); b[k] = uint16_t(ib < DIM ? ib : DIM); }
                ++k;
            }
        }
    }
    for (int i = k; i < K; ++i) w[i] = b[i] = DIM;
    return k;
}

}  // namespace

extern "C" {
int menaces_dim() { return DIM; }

// n positions -> w, b : n * K indices chacun ; rend le plus grand nombre d'indices vu (> K : tronque)
int menaces_lot(const uint8_t* recs, long n, int K, uint16_t* w, uint16_t* b, int fils) {
    if (fils < 1) fils = int(std::thread::hardware_concurrency());
    std::vector<int> maxi(fils, 0);
    std::vector<std::thread> th;
    for (int f = 0; f < fils; ++f)
        th.emplace_back([=, &maxi] {
            for (long i = n * f / fils; i < n * (f + 1) / fils; ++i) {
                int k = une(recs + 32 * i, K, w + K * i, b + K * i);
                if (k > maxi[f]) maxi[f] = k;
            }
        });
    for (auto& t : th) t.join();
    int m = 0;
    for (int x : maxi) m = x > m ? x : m;
    return m;
}
}
