#include "nnue.h"
#include "prof.h"

#include <algorithm>
#include <x86intrin.h>
#include <iostream>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #include <unistd.h>
#endif
#include <sstream>
#include <vector>

#include "bitboard.h"
#include "misc.h"
#include "position.h"

#if defined(__AVX2__)
    #include <immintrin.h>
#elif defined(__ARM_NEON) && defined(__aarch64__)
    // ARM 64 bits (Mac M1+, Graviton, Ampere d'Oracle) : registres de 8 int16.
    // Les sommes int16 et int32 bouclent comme en AVX2 : resultat identique
    // au bit pres, quel que soit l'ordre des additions.
    #include <arm_neon.h>
    #define ONE_NEON 1
#endif

namespace one {
namespace NNUE {

// Bucket d'entree d'une perspective (roi ou materiel), defini plus bas.
PerspInfo persp_info_pos(Color p, const Position& pos, Square ksq);

namespace {

struct Network {
    int           hidden   = 0;
    int           nBuckets = 0;
    bool          mirror   = false;
    bool          material = false;  // buckets d'entree choisis par le materiel (champ miroir = 2)
    std::uint8_t  layout[64] = {};

    int           outBuckets = 1;
    std::uint32_t activation = ACT_CRELU;

    std::int16_t* featureWeights = nullptr;   // [nBuckets * 768][HIDDEN_SIZE]
    std::int8_t*  featureWeights8 = nullptr;  // copie int8 (mode ONE_FT_INT8)
    std::int16_t  featureBias[HIDDEN_SIZE] = {};

    // Paires de pions (PP_3Wide de Stockfish 19) : [PAIR_INPUTS][HIDDEN_SIZE],
    // toujours en int16. Absentes (nullptr) pour un reseau sans paires.
    bool          pairs       = false;
    std::int16_t* pairWeights = nullptr;

    // Menaces : une ligne par (piece relative, case relative) d'une piece
    // attaquee par le camp adverse. Peu actives a la fois : on les ajoute a
    // chaque evaluation, sans mise a jour incrementale.
    bool          threats       = false;
    std::int16_t* threatWeights = nullptr;
    bool          fullThreats   = false;   // menaces completes (phase 10)
    std::int8_t*  fullThreatWeights = nullptr;

    // [outBuckets][2 * HIDDEN_SIZE], sur le tas pour rester aligne
    std::int16_t* outputWeights = nullptr;
    std::int32_t  outputBias[MAX_OUT_BUCKETS] = {};

    // Couches cachees (l1 > 0) : h1w [outBuckets][l1][2 * HIDDEN_SIZE],
    // h2w [outBuckets][l2][l1], h3w [outBuckets][l2] ; biais en int32.
    int                       l1 = 0, l2 = 0;
    std::int16_t*             h1w = nullptr;
    // h1w reordonnee pour le calcul creux : [bucket][paire d'entrees p][l1][2],
    // soit W1[j][2p], W1[j][2p+1] cote a cote pour chaque neurone j.
    std::int16_t*             h1t = nullptr;
    // pairmul : poids l1 int8 reordonnes [bucket][groupe de 4 entrees][l1][4]
    bool                      pairmul = false;
    // h2w reordonnee pour le calcul vectoriel : [bucket][paire de x1][l2][2]
    std::vector<std::int16_t> h2t;
    std::int8_t*              h1t8    = nullptr;
    std::vector<std::int32_t> h1b, h2b, h3b;
    std::vector<std::int16_t> h2w, h3w;

    ~Network() {
        aligned_large_pages_free(h1w);
        aligned_large_pages_free(h1t);
        aligned_large_pages_free(h1t8);
        aligned_large_pages_free(featureWeights);
        aligned_large_pages_free(featureWeights8);
        aligned_large_pages_free(pairWeights);
        aligned_large_pages_free(threatWeights);
        aligned_large_pages_free(fullThreatWeights);
        aligned_large_pages_free(outputWeights);
    }
};

Network*      Net    = nullptr;
std::string   NetPath;
std::uint32_t NetGen = 0;

#if defined(__AVX2__)
inline std::int32_t hsum(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s          = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s          = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

// Somme horizontale elargie en 64 bits. En SCReLU un terme vaut jusqu'a
// QA^2 * |w| = 255^2 * 127, soit 8,3 millions ; 512 termes tiennent dans une
// voie int32 (528 M) mais leur somme sur les huit voies, 4,2 milliards,
// deborderait. Le cas est theorique -- il faudrait tous les neurones satures
// et tous les poids au maximum -- mais il ne coute qu'une poignee de cycles,
// une fois par evaluation, de le rendre impossible.
inline std::int64_t hsum64(__m256i v) {
    const __m256i lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(v));
    const __m256i hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(v, 1));
    const __m256i s  = _mm256_add_epi64(lo, hi);
    const __m128i s2 =
      _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    return _mm_cvtsi128_si64(s2) + _mm_extract_epi64(s2, 1);
}
#endif

// Poids de la premiere couche : int16 par defaut, ou int8 avec ONE_FT_INT8.
// En int8, une ligne de poids fait 512 octets au lieu de 1024 : deux fois
// moins de memoire a lire a chaque mise a jour de l'accumulateur, au prix
// d'une extension de signe par paquet de 16.
#ifdef ONE_FT_INT8
using FtW = std::int8_t;
    #define FT_LOAD(p) _mm256_cvtepi8_epi16(_mm_load_si128(reinterpret_cast<const __m128i*>(p)))
    #define FT_LOADN(p) vmovl_s8(vld1_s8(p))
#else
using FtW = std::int16_t;
    #define FT_LOAD(p) _mm256_load_si256(reinterpret_cast<const __m256i*>(p))
    #define FT_LOADN(p) vld1q_s16(p)
#endif

inline void add_feature(std::int16_t* acc, const FtW* w) {
#if defined(__AVX2__)
    for (int i = 0; i < HIDDEN_SIZE; i += 16)
    {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
        __m256i b = FT_LOAD(w + i);
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_add_epi16(a, b));
    }
#elif defined(ONE_NEON)
    for (int i = 0; i < HIDDEN_SIZE; i += 8)
        vst1q_s16(acc + i, vaddq_s16(vld1q_s16(acc + i), FT_LOADN(w + i)));
#else
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        acc[i] = std::int16_t(acc[i] + w[i]);
#endif
}

inline void sub_feature(std::int16_t* acc, const FtW* w) {
#if defined(__AVX2__)
    for (int i = 0; i < HIDDEN_SIZE; i += 16)
    {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
        __m256i b = FT_LOAD(w + i);
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_sub_epi16(a, b));
    }
#elif defined(ONE_NEON)
    for (int i = 0; i < HIDDEN_SIZE; i += 8)
        vst1q_s16(acc + i, vsubq_s16(vld1q_s16(acc + i), FT_LOADN(w + i)));
#else
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        acc[i] = std::int16_t(acc[i] - w[i]);
#endif
}

// Produit scalaire d'une moitie d'accumulateur avec ses poids de sortie,
// activation ReLU ecretee comprise.
inline std::int32_t forward_half(const std::int16_t* acc, const std::int16_t* w) {
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa   = _mm256_set1_epi16(QA);
    __m256i       sum  = _mm256_setzero_si256();

    for (int i = 0; i < HIDDEN_SIZE; i += 16)
    {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
        a         = _mm256_min_epi16(_mm256_max_epi16(a, zero), qa);
        __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(w + i));
        sum       = _mm256_add_epi32(sum, _mm256_madd_epi16(a, b));
    }
    return hsum(sum);
#elif defined(ONE_NEON)
    const int16x8_t zero = vdupq_n_s16(0);
    const int16x8_t qa   = vdupq_n_s16(QA);
    int32x4_t       s0 = vdupq_n_s32(0), s1 = vdupq_n_s32(0);
    for (int i = 0; i < HIDDEN_SIZE; i += 8)
    {
        const int16x8_t a = vminq_s16(vmaxq_s16(vld1q_s16(acc + i), zero), qa);
        const int16x8_t b = vld1q_s16(w + i);
        s0 = vmlal_s16(s0, vget_low_s16(a), vget_low_s16(b));
        s1 = vmlal_high_s16(s1, a, b);
    }
    return vaddvq_s32(vaddq_s32(s0, s1));
#else
    std::int32_t sum = 0;
    for (int i = 0; i < HIDDEN_SIZE; ++i)
    {
        int a = acc[i] < 0 ? 0 : (acc[i] > QA ? QA : acc[i]);
        sum += a * w[i];
    }
    return sum;
#endif
}

// Meme produit, l'accumulateur recevant d'abord n lignes de menaces.
inline std::int32_t forward_half_t(const std::int16_t* acc, const std::int16_t* const* rows, int n,
                                   const std::int16_t* w) {
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa   = _mm256_set1_epi16(QA);
    __m256i       sum  = _mm256_setzero_si256();

    for (int i = 0; i < HIDDEN_SIZE; i += 16)
    {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
        for (int k = 0; k < n; ++k)
            a = _mm256_add_epi16(a, _mm256_load_si256(reinterpret_cast<const __m256i*>(rows[k] + i)));
        a         = _mm256_min_epi16(_mm256_max_epi16(a, zero), qa);
        __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(w + i));
        sum       = _mm256_add_epi32(sum, _mm256_madd_epi16(a, b));
    }
    return hsum(sum);
#elif defined(ONE_NEON)
    const int16x8_t zero = vdupq_n_s16(0);
    const int16x8_t qa   = vdupq_n_s16(QA);
    int32x4_t       s0 = vdupq_n_s32(0), s1 = vdupq_n_s32(0);
    for (int i = 0; i < HIDDEN_SIZE; i += 8)
    {
        int16x8_t a = vld1q_s16(acc + i);
        for (int k = 0; k < n; ++k)
            a = vaddq_s16(a, vld1q_s16(rows[k] + i));
        a = vminq_s16(vmaxq_s16(a, zero), qa);
        const int16x8_t b = vld1q_s16(w + i);
        s0 = vmlal_s16(s0, vget_low_s16(a), vget_low_s16(b));
        s1 = vmlal_high_s16(s1, a, b);
    }
    return vaddvq_s32(vaddq_s32(s0, s1));
#else
    std::int32_t sum = 0;
    for (int i = 0; i < HIDDEN_SIZE; ++i)
    {
        int v = acc[i];
        for (int k = 0; k < n; ++k)
            v += rows[k][i];
        v      = std::int16_t(v);
        int a  = v < 0 ? 0 : (v > QA ? QA : v);
        sum += a * w[i];
    }
    return sum;
#endif
}

// Meme produit scalaire, activation SCReLU : clamp(a, 0, QA)^2.
//
// L'astuce du produit en trois temps evite d'avoir a elargir en int32 avant la
// multiplication. v tient dans [0, 255] et w dans [-127, 127], donc v * w tient
// tout juste dans un int16 (32 385 pour 32 767) ; madd(v, v*w) rend alors
// v^2 * w directement en int32. Le resultat est en unites QA^2 : output() lui
// retire un facteur QA avant d'ajouter le biais.
inline std::int64_t forward_half_sq(const std::int16_t* acc, const std::int16_t* w) {
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa   = _mm256_set1_epi16(QA);
    __m256i       sum  = _mm256_setzero_si256();

    for (int i = 0; i < HIDDEN_SIZE; i += 16)
    {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
        a         = _mm256_min_epi16(_mm256_max_epi16(a, zero), qa);
        __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(w + i));
        sum       = _mm256_add_epi32(sum, _mm256_madd_epi16(a, _mm256_mullo_epi16(a, b)));
    }
    return hsum64(sum);
#elif defined(ONE_NEON)
    // Comme en AVX2 : a*w tient dans un int16, sommes int32 par voie, puis
    // somme finale elargie en int64.
    const int16x8_t zero = vdupq_n_s16(0);
    const int16x8_t qa   = vdupq_n_s16(QA);
    int32x4_t       s0 = vdupq_n_s32(0), s1 = vdupq_n_s32(0);
    for (int i = 0; i < HIDDEN_SIZE; i += 8)
    {
        const int16x8_t a  = vminq_s16(vmaxq_s16(vld1q_s16(acc + i), zero), qa);
        const int16x8_t aw = vmulq_s16(a, vld1q_s16(w + i));
        s0 = vmlal_s16(s0, vget_low_s16(a), vget_low_s16(aw));
        s1 = vmlal_high_s16(s1, a, aw);
    }
    return vaddlvq_s32(s0) + vaddlvq_s32(s1);
#else
    std::int64_t sum = 0;
    for (int i = 0; i < HIDDEN_SIZE; ++i)
    {
        int a = acc[i] < 0 ? 0 : (acc[i] > QA ? QA : acc[i]);
        sum += std::int64_t(a) * a * w[i];
    }
    return sum;
#endif
}

inline const FtW* weights_of(int featureIdx) {
#ifdef ONE_FT_INT8
    return Net->featureWeights8 + std::size_t(featureIdx) * HIDDEN_SIZE;
#else
    return Net->featureWeights + std::size_t(featureIdx) * HIDDEN_SIZE;
#endif
}

// --------------------------------------------------------- paires de pions
//
// Toute paire de pions, de n'importe quelles couleurs, sur la meme colonne ou
// sur deux colonnes voisines (rangees 2 a 7). Identifiant d'un pion vu de la
// perspective p : 48 * (couleur != p) + case relative - 8. La paire {a, b} a
// pour indice hi * (hi - 1) / 2 + lo. Meme formule que tools/train.py.
Bitboard PairBand[SQUARE_NB];

void init_pair_band() {
    for (int s = 0; s < SQUARE_NB; ++s)
    {
        const int f = s & 7;
        Bitboard  b = 0;
        for (int t = 8; t < 56; ++t)
            if (t != s && std::abs((t & 7) - f) <= 1)
                b |= Bitboard(1) << t;
        PairBand[s] = b;
    }
}

inline int pawn_id(Color p, Color c, Square s) {
    const int rel = (p == WHITE) ? int(s) : int(s) ^ 56;
    return 48 * (c != p) + rel - 8;
}

inline int pair_index(Color p, Color ca, Square a, Color cb, Square b) {
    const int ia = pawn_id(p, ca, a), ib = pawn_id(p, cb, b);
    const int hi = std::max(ia, ib), lo = std::min(ia, ib);
    return hi * (hi - 1) / 2 + lo;
}

inline const std::int16_t* pair_weights(int idx) {
    return Net->pairWeights + std::size_t(idx) * HIDDEN_SIZE;
}

// Paires qui contiennent au moins un pion de « changed », parmi les pions
// pw/pb : chaque paire n'est comptee qu'une fois.
int collect_pairs(Color p, Bitboard changedW, Bitboard changedB, Bitboard pw, Bitboard pb,
                  int* out) {
    int            n         = 0;
    const Bitboard changed   = changedW | changedB;
    const Bitboard unchanged = (pw | pb) & ~changed;
    for (Bitboard u = changed; u;)
    {
        const Square   a    = pop_lsb(u);
        const Color    ca   = (pb & a) ? BLACK : WHITE;
        const Bitboard mask = PairBand[a] & (unchanged | u);
        for (Bitboard q = pw & mask; q;)
            out[n++] = pair_index(p, ca, a, WHITE, pop_lsb(q));
        for (Bitboard q = pb & mask; q;)
            out[n++] = pair_index(p, ca, a, BLACK, pop_lsb(q));
    }
    return n;
}

// Pions des deux camps avant et apres le coup d'un noeud de la chaine.
struct PawnSets {
    Bitboard before[COLOR_NB];
    Bitboard after[COLOR_NB];
};

// Le coup change-t-il le bucket ou le miroir de la perspective p ? Si oui,
// toutes ses caracteristiques changent et un delta ne suffit plus.
bool requires_refresh(const DirtyPiece& dp, Color p) {
    if ((Net->nBuckets == 1 && !Net->mirror) || Net->material)
        return false;   // en mode materiel, le changement de bande se traite dans evaluate()

    for (int i = 0; i < dp.count; ++i)
        if (type_of(dp.piece[i]) == KING && color_of(dp.piece[i]) == p
            && dp.from[i] != SQ_NONE && dp.to[i] != SQ_NONE)
        {
            const PerspInfo a = persp_info(p, dp.from[i]);
            const PerspInfo b = persp_info(p, dp.to[i]);
            if (a.bucket != b.bucket || a.flip != b.flip)
                return true;
        }

    return false;
}

// dst = src, puis application des pieces qui ont bouge, pour les seules
// perspectives dont le bucket n'a pas change.
//
// Une seule passe par perspective : on lit l'accumulateur du parent, on retire
// et on ajoute les lignes de poids dans les registres, et on n'ecrit qu'une
// fois. La version precedente copiait d'abord les 2 Ko du parent, puis relisait
// et reecrivait l'accumulateur une fois PAR PIECE deplacee : trois
// allers-retours memoire au lieu d'un pour un coup tranquille. Le resultat est
// identique au bit pres -- l'arithmetique int16 modulo 2^16 est associative --
// ce que verifient l'assertion de evaluate() en compilation de debogage et le
// nombre de noeuds a profondeur fixe.
void apply(const Position& pos, StateInfo* si, bool doWhite, bool doBlack, const PawnSets* ps) {
    PROF_SCOPE(NNUE_APPLY);
#ifdef OLD_APPLY
    // Version d'avant le 11-09, gardee pour la mesure a froid (JOURNAL, etape
    // 12) : copie des 2 Ko de l'accumulateur, puis une lecture et une ecriture
    // completes PAR PIECE deplacee.
    const DirtyPiece& dp  = si->dirtyPiece;
    Accumulator&      dst = pos.accumulator(si->accIndex);

    std::memcpy(&dst, &pos.accumulator(si->previous->accIndex), sizeof(Accumulator));

    if (dp.count == 0)
        return;

    for (Color p : {WHITE, BLACK})
    {
        if ((p == WHITE && !doWhite) || (p == BLACK && !doBlack))
            continue;

        (void) ps;
        const PerspInfo pi = persp_info(p, si->kingSq[p]);
        std::int16_t*   a  = dst.v[p];

        for (int i = 0; i < dp.count; ++i)
        {
            if (dp.from[i] != SQ_NONE)
                sub_feature(a, weights_of(feature_index(p, dp.piece[i], dp.from[i], pi)));
            if (dp.to[i] != SQ_NONE)
                add_feature(a, weights_of(feature_index(p, dp.piece[i], dp.to[i], pi)));
        }
    }
#else
    const DirtyPiece&  dp  = si->dirtyPiece;
    Accumulator&       dst = pos.accumulator(si->accIndex);
    const Accumulator& src = pos.accumulator(si->previous->accIndex);

    for (Color p : {WHITE, BLACK})
    {
        // Une perspective qui sera recalculee en entier (changement de bucket
        // ou d'expert) n'a rien a recevoir : sa copie serait ecrasee, et les
        // noeuds intermediaires ne sont alors pas declares valides.
        if (!(p == WHITE ? doWhite : doBlack))
            continue;
        if (dp.count == 0)
        {
            std::memcpy(dst.v[p], src.v[p], sizeof(dst.v[p]));
            continue;
        }

        const PerspInfo     pi = persp_info_pos(p, pos, si->kingSq[p]);
        const FtW*          subs[3];
        const FtW*          adds[3];
        int                 ns = 0, na = 0;

        // Paires de pions retirees et ajoutees par ce coup
        int pSub[96], pAdd[96], nps = 0, npa = 0;
        if (Net->pairs && ps
            && (ps->before[WHITE] != ps->after[WHITE] || ps->before[BLACK] != ps->after[BLACK]))
        {
            npa = collect_pairs(p, ps->after[WHITE] & ~ps->before[WHITE],
                                ps->after[BLACK] & ~ps->before[BLACK], ps->after[WHITE],
                                ps->after[BLACK], pAdd);
            nps = collect_pairs(p, ps->before[WHITE] & ~ps->after[WHITE],
                                ps->before[BLACK] & ~ps->after[BLACK], ps->before[WHITE],
                                ps->before[BLACK], pSub);
        }

        for (int i = 0; i < dp.count; ++i)
        {
            if (dp.from[i] != SQ_NONE)
                subs[ns++] = weights_of(feature_index(p, dp.piece[i], dp.from[i], pi));
            if (dp.to[i] != SQ_NONE)
                adds[na++] = weights_of(feature_index(p, dp.piece[i], dp.to[i], pi));
        }

        const std::int16_t* s = src.v[p];
        std::int16_t*       d = dst.v[p];

#if defined(__AVX2__)
        for (int j = 0; j < HIDDEN_SIZE; j += 16)
        {
            __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(s + j));
            for (int k = 0; k < ns; ++k)
                v = _mm256_sub_epi16(v, FT_LOAD(subs[k] + j));
            for (int k = 0; k < na; ++k)
                v = _mm256_add_epi16(v, FT_LOAD(adds[k] + j));
            for (int k = 0; k < nps; ++k)
                v = _mm256_sub_epi16(v, _mm256_load_si256(
                                          reinterpret_cast<const __m256i*>(pair_weights(pSub[k]) + j)));
            for (int k = 0; k < npa; ++k)
                v = _mm256_add_epi16(v, _mm256_load_si256(
                                          reinterpret_cast<const __m256i*>(pair_weights(pAdd[k]) + j)));
            _mm256_store_si256(reinterpret_cast<__m256i*>(d + j), v);
        }
#elif defined(ONE_NEON)
        for (int j = 0; j < HIDDEN_SIZE; j += 8)
        {
            int16x8_t v = vld1q_s16(s + j);
            for (int k = 0; k < ns; ++k)
                v = vsubq_s16(v, FT_LOADN(subs[k] + j));
            for (int k = 0; k < na; ++k)
                v = vaddq_s16(v, FT_LOADN(adds[k] + j));
            for (int k = 0; k < nps; ++k)
                v = vsubq_s16(v, vld1q_s16(pair_weights(pSub[k]) + j));
            for (int k = 0; k < npa; ++k)
                v = vaddq_s16(v, vld1q_s16(pair_weights(pAdd[k]) + j));
            vst1q_s16(d + j, v);
        }
#else
        for (int j = 0; j < HIDDEN_SIZE; ++j)
        {
            int v = s[j];
            for (int k = 0; k < ns; ++k)
                v -= subs[k][j];
            for (int k = 0; k < na; ++k)
                v += adds[k][j];
            for (int k = 0; k < nps; ++k)
                v -= pair_weights(pSub[k])[j];
            for (int k = 0; k < npa; ++k)
                v += pair_weights(pAdd[k])[j];
            d[j] = std::int16_t(v);
        }
#endif
    }
#endif
}

}  // namespace

bool          loaded() { return Net != nullptr; }
std::uint32_t generation() { return NetGen; }

void unload() {
    delete Net;
    Net = nullptr;
    FullThreatsOn = false;
    NetPath.clear();
    ++NetGen;
}

std::string info() {
    if (!Net)
        return "NNUE absente (evaluation PSQT)";

    std::ostringstream ss;
    ss << "NNUE " << Net->nBuckets << "x768x" << Net->hidden << "x2->"
       << Net->outBuckets << (Net->mirror ? " miroir" : "") << (Net->material ? " experts-materiel" : "")
       << (Net->pairs ? " paires-de-pions" : "") << (Net->threats ? " menaces" : "")
       << (Net->fullThreats ? " menaces-completes" : "")
       << (Net->activation == ACT_SCRELU ? " screlu" : " crelu") << "  " << NetPath;
    return ss.str();
}

PerspInfo persp_info(Color p, Square ksq) {
    int rel = (p == WHITE) ? int(ksq) : int(ksq) ^ 56;

    const int flip = (Net->mirror && (rel & 7) >= 4) ? 7 : 0;
    rel ^= flip;

    return {Net->layout[rel], flip};
}

// Bande de materiel : meme formule que les buckets de sortie et que
// tools/train.py (mode "mat4"). Melange d'experts a la Triumviratus 8.0.
inline int material_bucket(int pieceCount) {
    // bandes de 32 / experts pieces : (n - 2) / 8 pour 4 experts, (n - 2) / 4 pour 8
    return std::clamp((pieceCount - 2) * Net->nBuckets / 32, 0, Net->nBuckets - 1);
}

// Le bucket d'entree d'une perspective : par la case du roi, ou par le
// materiel de la position courante.
PerspInfo persp_info_pos(Color p, const Position& pos, Square ksq) {
    if (Net->material)
        return {std::uint8_t(material_bucket(popcount(pos.pieces()))), 0};
    return persp_info(p, ksq);
}

int feature_index(Color p, Piece pc, Square s, const PerspInfo& pi) {
    const int pieceIdx = (color_of(pc) != p) * 6 + (type_of(pc) - 1);
    const int rel      = (((p == WHITE) ? int(s) : int(s) ^ 56)) ^ pi.flip;

    return pi.bucket * BASE_INPUTS + pieceIdx * 64 + rel;
}

// Dossier de l'executable, pour trouver le reseau quand l'interface lance le
// moteur depuis un autre dossier courant (En Croissant, Arena...).
static std::string exe_dir() {
#ifdef _WIN32
    char buf[1024];
    DWORD n = GetModuleFileNameA(nullptr, buf, sizeof(buf));
    std::string p(buf, n < sizeof(buf) ? n : 0);
#else
    char buf[1024];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    std::string p(buf, n > 0 ? std::size_t(n) : 0);
#endif
    std::size_t k = p.find_last_of("/\\");
    return k == std::string::npos ? std::string() : p.substr(0, k + 1);
}

bool load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    // chemin relatif introuvable dans le dossier courant : a cote de l'executable
    if (!f && !path.empty() && path[0] != '/' && path[0] != char(92) && path.find(':') == std::string::npos)
        f.open(exe_dir() + path, std::ios::binary);
    if (!f)
        return false;

    std::uint32_t magic = 0, hidden = 0, nBuckets = 0, mirror = 0, outBuckets = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&hidden), 4);
    f.read(reinterpret_cast<char*>(&nBuckets), 4);
    f.read(reinterpret_cast<char*>(&mirror), 4);
    f.read(reinterpret_cast<char*>(&outBuckets), 4);

    if (!f || (magic != MAGIC && magic != MAGIC4) || int(hidden) != HIDDEN_SIZE || nBuckets < 1
        || nBuckets > 64 || outBuckets < 1 || int(outBuckets) > MAX_OUT_BUCKETS
        || 32 % outBuckets != 0)
        return false;

    // "ONN3" precede l'activation configurable : ces fichiers sont en CReLU.
    std::uint32_t activation = ACT_CRELU;
    if (magic == MAGIC4)
        f.read(reinterpret_cast<char*>(&activation), 4);

    // Bit 8 du champ activation : le reseau a des entrees paires de pions ;
    // bit 9 : des entrees menaces.
    const bool pairs   = activation & PAIRS_FLAG;
    const bool threats = activation & THREAT_FLAG;
    const bool hasHidden = activation & HIDDEN_FLAG;
    const bool pairMul   = activation & PAIRMUL_FLAG;
    const bool fullThr   = activation & FULLTHREAT_FLAG;
    activation &= ~(PAIRS_FLAG | THREAT_FLAG | HIDDEN_FLAG | PAIRMUL_FLAG | FULLTHREAT_FLAG);
    if (fullThr && (threats || hasHidden || pairs || activation != ACT_CRELU))
        return false;   // menaces completes : CReLU, seules
    if (pairMul && !hasHidden)
        return false;
    if (hasHidden && (threats || activation != ACT_CRELU))
        return false;   // couches cachees : CReLU, sans menaces

    if (!f || (activation != ACT_CRELU && activation != ACT_SCRELU))
        return false;
    if (threats && activation != ACT_CRELU)
        return false;   // menaces : seule la sortie CReLU est ecrite

    auto* net       = new Network();
    net->hidden     = int(hidden);
    net->nBuckets   = int(nBuckets);
    net->material   = mirror == 2;
    net->mirror     = mirror == 1;
    net->outBuckets = int(outBuckets);
    net->activation = activation;
    net->pairs      = pairs;
    net->threats    = threats;
    net->fullThreats = fullThr;

    f.read(reinterpret_cast<char*>(net->layout), 64);

    const std::size_t nFeatures = std::size_t(nBuckets) * BASE_INPUTS;
    const std::size_t wBytes    = nFeatures * HIDDEN_SIZE * sizeof(std::int16_t);

    net->featureWeights = static_cast<std::int16_t*>(aligned_large_pages_alloc(wBytes));
    if (!net->featureWeights)
    {
        delete net;
        return false;
    }

    const std::size_t oBytes =
      std::size_t(net->outBuckets) * 2 * HIDDEN_SIZE * sizeof(std::int16_t);
    net->outputWeights = static_cast<std::int16_t*>(aligned_large_pages_alloc(oBytes));
    if (!net->outputWeights)
    {
        delete net;
        return false;
    }

    f.read(reinterpret_cast<char*>(net->featureWeights), std::streamsize(wBytes));
#ifdef ONE_FT_INT8
    {
        // Copie int8 des poids, avec saturation. Un reseau entraine sans
        // plafond sur ses poids en perd quelques-uns ici : on le dit.
        const std::size_t n = wBytes / sizeof(std::int16_t);
        net->featureWeights8 = static_cast<std::int8_t*>(aligned_large_pages_alloc(n));
        if (!net->featureWeights8)
            return false;
        std::size_t ecretes = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const int v = std::clamp(int(net->featureWeights[i]), -127, 127);
            ecretes += v != net->featureWeights[i];
            net->featureWeights8[i] = std::int8_t(v);
        }
        std::cout << "info string poids int8 : " << ecretes << " ecretes sur " << n << std::endl;
    }
#endif
    f.read(reinterpret_cast<char*>(net->featureBias), sizeof(net->featureBias));
    if (pairs)
    {
        const std::size_t pBytes = std::size_t(PAIR_INPUTS) * HIDDEN_SIZE * sizeof(std::int16_t);
        net->pairWeights = static_cast<std::int16_t*>(aligned_large_pages_alloc(pBytes));
        if (!net->pairWeights)
        {
            delete net;
            return false;
        }
        f.read(reinterpret_cast<char*>(net->pairWeights), std::streamsize(pBytes));
        static bool bandReady = false;
        if (!bandReady)
            init_pair_band(), bandReady = true;
    }
    if (threats)
    {
        const std::size_t tBytes = std::size_t(THREAT_INPUTS) * HIDDEN_SIZE * sizeof(std::int16_t);
        net->threatWeights = static_cast<std::int16_t*>(aligned_large_pages_alloc(tBytes));
        if (!net->threatWeights)
        {
            delete net;
            return false;
        }
        f.read(reinterpret_cast<char*>(net->threatWeights), std::streamsize(tBytes));
    }
    if (fullThr)
    {
        const std::size_t tBytes = std::size_t(FULL_THREAT_DIM) * HIDDEN_SIZE;
        net->fullThreatWeights = static_cast<std::int8_t*>(aligned_large_pages_alloc(tBytes));
        if (!net->fullThreatWeights)
        {
            delete net;
            return false;
        }
        f.read(reinterpret_cast<char*>(net->fullThreatWeights), std::streamsize(tBytes));
    }
    if (hasHidden)
    {
        std::uint32_t l1 = 0, l2 = 0;
        f.read(reinterpret_cast<char*>(&l1), 4);
        f.read(reinterpret_cast<char*>(&l2), 4);
        if (!f || l1 < 8 || l1 > MAX_L1 || l1 % 8 || l2 < 8 || l2 > MAX_L2 || l2 % 8)
        {
            delete net;
            return false;
        }
        const int ob = net->outBuckets;
        net->l1 = int(l1);
        net->l2 = int(l2);
        net->pairmul = pairMul;
        if (pairMul)
        {
            // [ob][l1][HIDDEN_SIZE] int8, puis biais et couches suivantes
            const std::size_t n8 = std::size_t(ob) * l1 * HIDDEN_SIZE;
            std::vector<std::int8_t> w8(n8);
            f.read(reinterpret_cast<char*>(w8.data()), std::streamsize(n8));
            net->h1t8 = static_cast<std::int8_t*>(aligned_large_pages_alloc(n8));
            if (!net->h1t8)
            {
                delete net;
                return false;
            }
            for (int b = 0; b < ob; ++b)
                for (int c = 0; c < HIDDEN_SIZE / 4; ++c)
                    for (std::size_t j = 0; j < l1; ++j)
                        for (int k = 0; k < 4; ++k)
                            net->h1t8[((std::size_t(b) * HIDDEN_SIZE / 4 + c) * l1 + j) * 4 + k] =
                              w8[(std::size_t(b) * l1 + j) * HIDDEN_SIZE + 4 * c + k];
            net->h1b.resize(std::size_t(ob) * l1);
            net->h2w.resize(std::size_t(ob) * l2 * l1);
            net->h2b.resize(std::size_t(ob) * l2);
            net->h3w.resize(std::size_t(ob) * l2);
            net->h3b.resize(std::size_t(ob));
            f.read(reinterpret_cast<char*>(net->h1b.data()), std::streamsize(net->h1b.size() * 4));
            f.read(reinterpret_cast<char*>(net->h2w.data()), std::streamsize(net->h2w.size() * 2));
            f.read(reinterpret_cast<char*>(net->h2b.data()), std::streamsize(net->h2b.size() * 4));
            f.read(reinterpret_cast<char*>(net->h3w.data()), std::streamsize(net->h3w.size() * 2));
            f.read(reinterpret_cast<char*>(net->h3b.data()), std::streamsize(net->h3b.size() * 4));
        }
        else
        {
        const std::size_t h1Bytes = std::size_t(ob) * l1 * 2 * HIDDEN_SIZE * sizeof(std::int16_t);
        net->h1w = static_cast<std::int16_t*>(aligned_large_pages_alloc(h1Bytes));
        if (!net->h1w)
        {
            delete net;
            return false;
        }
        net->h1b.resize(std::size_t(ob) * l1);
        net->h2w.resize(std::size_t(ob) * l2 * l1);
        net->h2b.resize(std::size_t(ob) * l2);
        net->h3w.resize(std::size_t(ob) * l2);
        net->h3b.resize(std::size_t(ob));
        f.read(reinterpret_cast<char*>(net->h1w), std::streamsize(h1Bytes));
        f.read(reinterpret_cast<char*>(net->h1b.data()), std::streamsize(net->h1b.size() * 4));
        f.read(reinterpret_cast<char*>(net->h2w.data()), std::streamsize(net->h2w.size() * 2));
        f.read(reinterpret_cast<char*>(net->h2b.data()), std::streamsize(net->h2b.size() * 4));
        f.read(reinterpret_cast<char*>(net->h3w.data()), std::streamsize(net->h3w.size() * 2));
        f.read(reinterpret_cast<char*>(net->h3b.data()), std::streamsize(net->h3b.size() * 4));
        net->h1t = static_cast<std::int16_t*>(aligned_large_pages_alloc(h1Bytes));
        if (!net->h1t)
        {
            delete net;
            return false;
        }
        const std::size_t in = 2 * HIDDEN_SIZE;
        for (int b = 0; b < ob; ++b)
            for (std::size_t p = 0; p < in / 2; ++p)
                for (std::size_t j = 0; j < l1; ++j)
                    for (int k = 0; k < 2; ++k)
                        net->h1t[((std::size_t(b) * in / 2 + p) * l1 + j) * 2 + k] =
                          net->h1w[(std::size_t(b) * l1 + j) * in + 2 * p + k];
        }
        net->h2t.resize(net->h2w.size());
        for (int b = 0; b < ob; ++b)
            for (std::size_t p = 0; p < l1 / 2; ++p)
                for (std::size_t k = 0; k < l2; ++k)
                    for (int e = 0; e < 2; ++e)
                        net->h2t[((std::size_t(b) * l1 / 2 + p) * l2 + k) * 2 + e] =
                          net->h2w[(std::size_t(b) * l2 + k) * l1 + 2 * p + e];
    }
    else
    {
        f.read(reinterpret_cast<char*>(net->outputWeights), std::streamsize(oBytes));
        f.read(reinterpret_cast<char*>(net->outputBias),
               std::streamsize(net->outBuckets * sizeof(std::int32_t)));
    }

    if (!f)
    {
        delete net;
        return false;
    }

    for (int i = 0; i < 64; ++i)
        if (net->layout[i] >= net->nBuckets)
        {
            delete net;
            return false;
        }

    delete Net;
    Net     = net;
    FullThreatsOn = net->fullThreats;
    NetPath = path;
    ++NetGen;      // invalide tout accumulateur calcule avec le reseau precedent
    return true;
}

void refresh_side(Accumulator& acc, const Position& pos, Color p) {
    std::memcpy(acc.v[p], Net->featureBias, sizeof(Net->featureBias));

    const PerspInfo pi = persp_info_pos(p, pos, pos.square<KING>(p));

    for (Bitboard b = pos.pieces(); b;)
    {
        const Square s = pop_lsb(b);
        add_feature(acc.v[p], weights_of(feature_index(p, pos.piece_on(s), s, pi)));
    }

    if (Net->pairs)
    {
        const Bitboard pw = pos.pieces(WHITE, PAWN), pb = pos.pieces(BLACK, PAWN);
        int            idx[160];
        const int      n = collect_pairs(p, pw, pb, pw, pb, idx);
        // Lignes int16 ajoutees par paquets de 16 (AVX2) : un recalcul porte
        // une quarantaine de paires, et les experts par materiel en font un a
        // chaque changement de bande.
        for (int k = 0; k < n; ++k)
        {
            const std::int16_t* w = pair_weights(idx[k]);
#if defined(__AVX2__)
            for (int j = 0; j < HIDDEN_SIZE; j += 16)
            {
                __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc.v[p] + j));
                a = _mm256_add_epi16(a, _mm256_load_si256(reinterpret_cast<const __m256i*>(w + j)));
                _mm256_store_si256(reinterpret_cast<__m256i*>(acc.v[p] + j), a);
            }
#elif defined(ONE_NEON)
            for (int j = 0; j < HIDDEN_SIZE; j += 8)
                vst1q_s16(acc.v[p] + j, vaddq_s16(vld1q_s16(acc.v[p] + j), vld1q_s16(w + j)));
#else
            for (int j = 0; j < HIDDEN_SIZE; ++j)
                acc.v[p][j] = std::int16_t(acc.v[p][j] + w[j]);
#endif
        }
    }
}

void refresh(Accumulator& acc, const Position& pos) {
    refresh_side(acc, pos, WHITE);
    refresh_side(acc, pos, BLACK);
}

// Table de Finny : pour chaque perspective et chaque bucket d'entree, le
// dernier accumulateur calcule dans ce bucket et les pieces qu'il contenait.
// Un changement de bucket (expert par materiel, ou roi) ne repart plus de
// zero : on ne corrige que les pieces qui ont bouge depuis. Les poids
// s'ajoutent en int16 sans saturation, donc le resultat est identique bit
// pour bit a un recalcul complet, quel que soit l'ordre.
namespace {
struct FinnyEntry {
    alignas(64) std::int16_t acc[HIDDEN_SIZE];
    Bitboard      bb[COLOR_NB][PIECE_TYPE_NB];
    std::uint32_t gen = 0;
};
thread_local FinnyEntry Finny[COLOR_NB][64][2];
}  // namespace

void refresh_side_cached(Accumulator& acc, const Position& pos, Color p) {
    if (Net->pairs)
    {
        refresh_side(acc, pos, p);
        return;
    }

    const PerspInfo pi = persp_info_pos(p, pos, pos.square<KING>(p));
    FinnyEntry&     e  = Finny[p][pi.bucket][pi.flip != 0];

    if (e.gen != NetGen)
    {
        std::memcpy(e.acc, Net->featureBias, sizeof(e.acc));
        std::memset(e.bb, 0, sizeof(e.bb));
        e.gen = NetGen;
    }

    // Lignes a retirer et a ajouter, puis une seule passe par tuile de
    // registres : l'accumulateur n'est lu et ecrit qu'une fois, dans l'entree
    // de la table ET dans l'accumulateur du noeud (pas de copie a part).
    const FtW* subs[32];
    const FtW* adds[32];
    int        ns = 0, na = 0;
    for (Color c : {WHITE, BLACK})
        for (PieceType pt = PAWN; pt <= KING; ++pt)
        {
            const Bitboard now = pos.pieces(c, pt);
            const Piece    pc  = make_piece(c, pt);
            for (Bitboard b = e.bb[c][pt] & ~now; b;)
                subs[ns++] = weights_of(feature_index(p, pc, pop_lsb(b), pi));
            for (Bitboard b = now & ~e.bb[c][pt]; b;)
                adds[na++] = weights_of(feature_index(p, pc, pop_lsb(b), pi));
            e.bb[c][pt] = now;
        }

#if defined(__AVX2__)
    constexpr int TILE = 8;   // 8 registres de 16 valeurs : 128 valeurs par tuile
    for (int j = 0; j < HIDDEN_SIZE; j += 16 * TILE)
    {
        __m256i v[TILE];
        for (int t = 0; t < TILE; ++t)
            v[t] = _mm256_load_si256(reinterpret_cast<const __m256i*>(e.acc + j + 16 * t));
        for (int k = 0; k < ns; ++k)
            for (int t = 0; t < TILE; ++t)
                v[t] = _mm256_sub_epi16(v[t], FT_LOAD(subs[k] + j + 16 * t));
        for (int k = 0; k < na; ++k)
            for (int t = 0; t < TILE; ++t)
                v[t] = _mm256_add_epi16(v[t], FT_LOAD(adds[k] + j + 16 * t));
        for (int t = 0; t < TILE; ++t)
        {
            _mm256_store_si256(reinterpret_cast<__m256i*>(e.acc + j + 16 * t), v[t]);
            _mm256_store_si256(reinterpret_cast<__m256i*>(acc.v[p] + j + 16 * t), v[t]);
        }
    }
#elif defined(ONE_NEON)
    constexpr int TILE = 16;  // 16 des 32 registres NEON : 128 valeurs par tuile
    for (int j = 0; j < HIDDEN_SIZE; j += 8 * TILE)
    {
        int16x8_t v[TILE];
        for (int t = 0; t < TILE; ++t)
            v[t] = vld1q_s16(e.acc + j + 8 * t);
        for (int k = 0; k < ns; ++k)
            for (int t = 0; t < TILE; ++t)
                v[t] = vsubq_s16(v[t], FT_LOADN(subs[k] + j + 8 * t));
        for (int k = 0; k < na; ++k)
            for (int t = 0; t < TILE; ++t)
                v[t] = vaddq_s16(v[t], FT_LOADN(adds[k] + j + 8 * t));
        for (int t = 0; t < TILE; ++t)
        {
            vst1q_s16(e.acc + j + 8 * t, v[t]);
            vst1q_s16(acc.v[p] + j + 8 * t, v[t]);
        }
    }
#else
    for (int k = 0; k < ns; ++k)
        sub_feature(e.acc, subs[k]);
    for (int k = 0; k < na; ++k)
        add_feature(e.acc, adds[k]);
    std::memcpy(acc.v[p], e.acc, sizeof(e.acc));
#endif
}

// Bucket de sortie de la position courante.
int current_bucket(const Position& pos) {
    return output_bucket(popcount(pos.pieces()), Net->outBuckets);
}

// Lignes de menaces vues de la perspective p : chaque piece (hors rois)
// attaquee par le camp adverse.
int threat_rows(const Position& pos, Color p, const Bitboard att[COLOR_NB], const std::int16_t** rows) {
    int n = 0;
    for (Color c : {WHITE, BLACK})
        for (Bitboard b = pos.pieces(c) & ~pos.pieces(KING) & att[~c]; b;)
        {
            const Square s   = pop_lsb(b);
            const int    idx = ((c != p) * 6 + (type_of(pos.piece_on(s)) - 1)) * 64
                          + ((p == WHITE) ? int(s) : int(s) ^ 56);
            rows[n++] = Net->threatWeights + std::size_t(idx) * HIDDEN_SIZE;
        }
    return n;
}

inline Bitboard attacks_of(const Position& pos, Color c) {
    return pos.attacks_by<PAWN>(c) | pos.attacks_by<KNIGHT>(c) | pos.attacks_by<BISHOP>(c)
         | pos.attacks_by<ROOK>(c) | pos.attacks_by<QUEEN>(c) | pos.attacks_by<KING>(c);
}

Value output_threats(const Accumulator& acc, const Position& pos, int bucket) {
    const Color         stm = pos.side_to_move();
    const std::int16_t* w   = Net->outputWeights + std::size_t(bucket) * 2 * HIDDEN_SIZE;
    const Bitboard      att[COLOR_NB] = {attacks_of(pos, WHITE), attacks_of(pos, BLACK)};
    const std::int16_t* rUs[32];
    const std::int16_t* rThem[32];
    const int           nUs   = threat_rows(pos, stm, att, rUs);
    const int           nThem = threat_rows(pos, ~stm, att, rThem);

    const std::int64_t sum = std::int64_t(forward_half_t(acc.v[stm], rUs, nUs, w))
                           + forward_half_t(acc.v[~stm], rThem, nThem, w + HIDDEN_SIZE);
    const int cp = int((sum + Net->outputBias[bucket]) * SCALE / (QA * QB));
    return Value(std::clamp(cp, -12000, 12000));
}

// ------------------------------------------------------------------------
// Menaces completes (phase 10, Full Threats de Stockfish 18-19, GPL-3) : une
// entree par paire (attaquant, case de depart, piece attaquee, case visee),
// vue de chaque camp ; pas de miroir de roi. Memes indices que la reference
// tools/menaces_completes.py et l'indexeur de l'entraineur.
//
// Accumulateur a part (les menaces ne dependent pas de l'expert par
// materiel) : a chaque noeud evalue, la liste triee des menaces est comparee
// a celle du parent, et seules les lignes qui changent sont appliquees ;
// sans parent valide, recalcul complet. Poids int8, sommes int16.
// ------------------------------------------------------------------------
namespace {
constexpr int FT_MAXK = 128;   // plus de menaces actives que jamais vu (48 sur 2 M positions)

struct FtLuts {
    std::uint32_t lut1[12][12][2];
    std::uint16_t lut2[6][64][64];   // reference, sert a verifier ft_off()
    std::uint16_t base[6][64];       // depart de chaque (type, case) : 768 octets, toujours en cache
};
FtLuts* FtTab = nullptr;

inline int ft_off(int pt, int fr, int to);

void init_full_threats() {
    static const int NB_CIBLES[6] = {4, 10, 8, 8, 10, 0};
    static const int CARTE[6][6]  = {{-1, 0, -1, 1, -1, -1}, {0, 1, 2, 3, 4, -1},  {0, 1, 2, 3, -1, -1},
                                     {0, 1, 2, 3, -1, -1},   {0, 1, 2, 3, 4, -1}, {-1, -1, -1, -1, -1, -1}};
    auto* t    = new FtLuts();
    auto  vide = [](int pt, int sq, int colPion) -> Bitboard {
        const Square s = Square(sq);
        return pt == 0 ? pawn_attacks_bb(Color(colPion), s) : attacks_bb(PieceType(pt + 1), s, 0);
    };
    int decal[6];
    for (int pt = 0; pt < 6; ++pt)
    {
        int off = 0;
        for (int fr = 0; fr < 64; ++fr)
        {
            for (int to = 0; to < 64; ++to)
            {
                const Bitboard a = pt != 0 ? vide(pt, fr, 0) : vide(0, fr, to < fr ? BLACK : WHITE);
                t->lut2[pt][fr][to] = std::uint16_t(off + popcount(((Bitboard(1) << to) - 1) & a));
            }
            t->base[pt][fr] = std::uint16_t(off);
            if (pt != 0)
                off += popcount(vide(pt, fr, 0));
            else if (fr >= 8 && fr <= 55)
                off += popcount(vide(0, fr, WHITE));
        }
        decal[pt] = off;
    }
    int cumul = 0;
    for (int a = 0; a < 12; ++a)
    {
        const int ta = a % 6, cp = decal[ta], nv = NB_CIBLES[ta];
        for (int d = 0; d < 12; ++d)
        {
            const int  td     = d % 6, m = CARTE[ta][td];
            const bool ennemi = (a / 6) != (d / 6);
            const bool semi   = ta == td && (ennemi || ta != 0);
            const auto f      = std::uint32_t(cumul + ((d / 6) * (nv / 2) + m) * cp);
            t->lut1[a][d][0]  = m < 0 ? FULL_THREAT_DIM : f;
            t->lut1[a][d][1]  = (m < 0 || semi) ? FULL_THREAT_DIM : f;
        }
        cumul += nv * cp;
    }
    assert(cumul == FULL_THREAT_DIM);
    FtTab = t;
#ifndef NDEBUG
    for (int pt = 0; pt < 6; ++pt)
        for (int fr = 0; fr < 64; ++fr)
            for (int to = 0; to < 64; ++to)
                assert(ft_off(pt, fr, to) == t->lut2[pt][fr][to]);
#endif
}

inline int ft_code(Piece pc) { return (color_of(pc) == BLACK) * 6 + (type_of(pc) - 1); }

// Meme valeur que lut2[pt][fr][to], sans la table de 49 Ko (chassee du cache par
// les lignes de poids) : depart + rang de la case visee dans les attaques sur
// echiquier vide (tables d'attaques de la generation de coups, toujours chaudes).
inline int ft_off(int pt, int fr, int to) {
    const Bitboard a = pt == 0 ? pawn_attacks_bb(to < fr ? BLACK : WHITE, Square(fr))
                               : PseudoAttacks[pt + 1][fr];
    return FtTab->base[pt][fr] + popcount(a & ((Bitboard(1) << to) - 1));
}

// Listes triees des menaces actives, vues des blancs (w) et des noirs (b)
void ft_lists(const Position& pos, std::uint16_t* w, int& nw, std::uint16_t* b, int& nb) {
    nw = nb = 0;
    const Bitboard occ = pos.pieces();
    for (Bitboard o = occ & ~pos.pieces(KING); o;)
    {
        const Square   fr  = pop_lsb(o);
        const Piece    pc  = pos.piece_on(fr);
        const int      a   = ft_code(pc), ta = a % 6;
        const Bitboard att =
          (ta == 0 ? pawn_attacks_bb(color_of(pc), fr) : attacks_bb(type_of(pc), fr, occ)) & occ;
        for (Bitboard t = att; t;)
        {
            const Square        to = pop_lsb(t);
            const int           d  = ft_code(pos.piece_on(to));
            const int           f = int(fr), tt = int(to), fb = f ^ 56, tb = tt ^ 56;
            const std::uint32_t iw = FtTab->lut1[a][d][f < tt] + ft_off(ta, f, tt);
            const std::uint32_t ib =
              FtTab->lut1[(a + 6) % 12][(d + 6) % 12][fb < tb] + ft_off(ta, fb, tb);
            if (iw < std::uint32_t(FULL_THREAT_DIM) && nw < FT_MAXK)
                w[nw++] = std::uint16_t(iw);
            if (ib < std::uint32_t(FULL_THREAT_DIM) && nb < FT_MAXK)
                b[nb++] = std::uint16_t(ib);
        }
    }
    std::sort(w, w + nw);
    std::sort(b, b + nb);
}

// Changements de menaces d'un coup, codes independamment de la perspective :
// case de l'attaquant (6 bits) | case visee << 6 | code attaquant << 12 |
// code attaque << 16 | bit 31 = menace ajoutee (sinon retiree).
constexpr int FT_MAXD = 192;
#ifndef FT_SANS_ANNUL
#define FT_SANS_ANNUL 0
#endif
constexpr int FT_COMB = 256;
#ifndef FT_PF_DIST
#define FT_PF_DIST 4
#endif   // changements cumules au plus (au-dela : recalcul complet)

struct alignas(64) ThreatAcc {
    std::int16_t     v[COLOR_NB][HIDDEN_SIZE];
    std::uint32_t    dirty[FT_MAXD];
    int              nd;
    const StateInfo* dirtyOwner;   // noeud dont dirty[] decrit le coup
    const StateInfo* owner;        // noeud dont v[] est l'accumulateur
};
thread_local ThreatAcc* ThrPool = nullptr;   // une case par place du reservoir des accumulateurs

inline const std::int8_t* ft_row(int idx) {
    return Net->fullThreatWeights + std::size_t(idx) * HIDDEN_SIZE;
}

// d = s - lignes(sub) + lignes(add), poids int8 etendus en int16
void ft_apply(const std::int16_t* s, std::int16_t* d, const int* sub, int ns, const int* add, int na) {
#if defined(__AVX2__) && !defined(FT_APPLY16)
    // Tuiles de 64 valeurs : une ligne de cache entiere de chaque ligne de poids par
    // tuile, et prechargement de la ligne de cache suivante (table hors cache).
    const std::int8_t* rs[FT_COMB];
    const std::int8_t* ra[FT_COMB];
    for (int k = 0; k < ns; ++k)
        rs[k] = ft_row(sub[k]);
    for (int k = 0; k < na; ++k)
        ra[k] = ft_row(add[k]);
    for (int j = 0; j < HIDDEN_SIZE; j += 64)
    {
        if (j + FT_PF_DIST * 64 < HIDDEN_SIZE)
        {
            for (int k = 0; k < ns; ++k)
                __builtin_prefetch(rs[k] + j + FT_PF_DIST * 64, 0, 0);
            for (int k = 0; k < na; ++k)
                __builtin_prefetch(ra[k] + j + FT_PF_DIST * 64, 0, 0);
        }
        __m256i v0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(s + j));
        __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(s + j + 16));
        __m256i v2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(s + j + 32));
        __m256i v3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(s + j + 48));
        for (int k = 0; k < ns; ++k)
        {
            const __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(rs[k] + j));
            const __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(rs[k] + j + 32));
            v0 = _mm256_sub_epi16(v0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w)));
            v1 = _mm256_sub_epi16(v1, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1)));
            v2 = _mm256_sub_epi16(v2, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x)));
            v3 = _mm256_sub_epi16(v3, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x, 1)));
        }
        for (int k = 0; k < na; ++k)
        {
            const __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(ra[k] + j));
            const __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(ra[k] + j + 32));
            v0 = _mm256_add_epi16(v0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w)));
            v1 = _mm256_add_epi16(v1, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1)));
            v2 = _mm256_add_epi16(v2, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x)));
            v3 = _mm256_add_epi16(v3, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x, 1)));
        }
        _mm256_store_si256(reinterpret_cast<__m256i*>(d + j), v0);
        _mm256_store_si256(reinterpret_cast<__m256i*>(d + j + 16), v1);
        _mm256_store_si256(reinterpret_cast<__m256i*>(d + j + 32), v2);
        _mm256_store_si256(reinterpret_cast<__m256i*>(d + j + 48), v3);
    }
#elif defined(__AVX2__)
    for (int j = 0; j < HIDDEN_SIZE; j += 16)
    {
        __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(s + j));
        for (int k = 0; k < ns; ++k)
            v = _mm256_sub_epi16(
              v, _mm256_cvtepi8_epi16(_mm_load_si128(reinterpret_cast<const __m128i*>(ft_row(sub[k]) + j))));
        for (int k = 0; k < na; ++k)
            v = _mm256_add_epi16(
              v, _mm256_cvtepi8_epi16(_mm_load_si128(reinterpret_cast<const __m128i*>(ft_row(add[k]) + j))));
        _mm256_store_si256(reinterpret_cast<__m256i*>(d + j), v);
    }
#elif defined(ONE_NEON)
    for (int j = 0; j < HIDDEN_SIZE; j += 8)
    {
        int16x8_t v = vld1q_s16(s + j);
        for (int k = 0; k < ns; ++k)
            v = vsubq_s16(v, vmovl_s8(vld1_s8(ft_row(sub[k]) + j)));
        for (int k = 0; k < na; ++k)
            v = vaddq_s16(v, vmovl_s8(vld1_s8(ft_row(add[k]) + j)));
        vst1q_s16(d + j, v);
    }
#else
    for (int j = 0; j < HIDDEN_SIZE; ++j)
    {
        int v = s[j];
        for (int k = 0; k < ns; ++k)
            v -= ft_row(sub[k])[j];
        for (int k = 0; k < na; ++k)
            v += ft_row(add[k])[j];
        d[j] = std::int16_t(v);
    }
#endif
}

alignas(64) const std::int16_t FtZero[HIDDEN_SIZE] = {};

// Indice d'une menace codee, vue de la perspective p (DIM si exclue)
inline int ft_index(std::uint32_t e, Color p) {
    int fr = int(e & 63), to = int((e >> 6) & 63), a = int((e >> 12) & 15), d = int((e >> 16) & 15);
    if (p == BLACK)
        fr ^= 56, to ^= 56, a = (a + 6) % 12, d = (d + 6) % 12;
    return int(FtTab->lut1[a][d][fr < to] + ft_off(a % 6, fr, to));
}

// Attaquants (hors rois) d'une case, sur un plateau donne par ses bitboards
inline Bitboard ft_attackers(Square z, Bitboard occ, const Bitboard* bt, const Bitboard* bc) {
    return (pawn_attacks_bb(BLACK, z) & bt[PAWN] & bc[WHITE]) | (pawn_attacks_bb(WHITE, z) & bt[PAWN] & bc[BLACK])
         | (attacks_bb<KNIGHT>(z) & bt[KNIGHT]) | (attacks_bb<BISHOP>(z, occ) & (bt[BISHOP] | bt[QUEEN]))
         | (attacks_bb<ROOK>(z, occ) & (bt[ROOK] | bt[QUEEN]));
}

}  // namespace

bool FullThreatsOn = false;

// Appele a la fin de do_move et de do_null_move quand le reseau a des menaces
// completes : menaces retirees et ajoutees par le coup. Seules peuvent changer
// les menaces des pieces posees sur une case touchee (Z) et celles des pieces
// qui attaquent une case de Z avant ou apres le coup (glisseurs decouverts ou
// bloques compris) : on enumere ces menaces sur les deux plateaux et on garde
// la difference.
void ft_on_move(const Position& pos) {
    StateInfo* st = pos.state();
    if (st->accIndex < 0)
        return;
    ThreatAcc* pool = ThrPool;   // thread_local lu une fois (MinGW : chaque acces est un appel)
    if (!pool)
        pool = ThrPool = new ThreatAcc[ACC_POOL];
    if (!FtTab)
        init_full_threats();
    ThreatAcc&        t  = pool[st->accIndex];
    const DirtyPiece& dp = st->dirtyPiece;
    t.nd                 = 0;
    t.dirtyOwner         = st;
    if (dp.count == 0)
        return;

    // Plateau d'avant le coup : celui d'apres, moins les pieces arrivees, plus les pieces parties
    Bitboard bt[PIECE_TYPE_NB], bc[COLOR_NB], Z = 0;
    for (int i = 0; i < PIECE_TYPE_NB; ++i)
        bt[i] = pos.pieces(PieceType(i));
    bc[WHITE] = pos.pieces(WHITE), bc[BLACK] = pos.pieces(BLACK);
    for (int i = 0; i < dp.count; ++i)
        if (dp.to[i] != SQ_NONE)
        {
            const Bitboard b = square_bb(dp.to[i]);
            bt[type_of(dp.piece[i])] ^= b, bc[color_of(dp.piece[i])] ^= b, Z |= b;
        }
    for (int i = 0; i < dp.count; ++i)
        if (dp.from[i] != SQ_NONE)
        {
            const Bitboard b = square_bb(dp.from[i]);
            bt[type_of(dp.piece[i])] ^= b, bc[color_of(dp.piece[i])] ^= b, Z |= b;
        }
    const Bitboard occB = bc[WHITE] | bc[BLACK], occA = pos.pieces();
    Bitboard       btA[PIECE_TYPE_NB], bcA[COLOR_NB] = {pos.pieces(WHITE), pos.pieces(BLACK)};
    for (int i = 0; i < PIECE_TYPE_NB; ++i)
        btA[i] = pos.pieces(PieceType(i));

    Bitboard A = Z;
    for (Bitboard z = Z; z;)
    {
        const Square s = pop_lsb(z);
        A |= ft_attackers(s, occB, bt, bc) | ft_attackers(s, occA, btA, bcA);
    }

    auto avant = [&](Square s) -> Piece {
        if (!(Z & s))
            return pos.piece_on(s);
        for (int i = 0; i < dp.count; ++i)
            if (dp.from[i] == s)
                return dp.piece[i];
        return NO_PIECE;
    };
    auto code = [](Square s, Square to, Piece a, Piece d, bool ajout) {
        return std::uint32_t(s) | std::uint32_t(to) << 6 | std::uint32_t(ft_code(a)) << 12
             | std::uint32_t(ft_code(d)) << 16 | (ajout ? 0x80000000u : 0u);
    };
    int n = 0;
    // Pieces des cases touchees : toutes leurs menaces d'avant retirees, d'apres ajoutees
    // (le contenu d'une case touchee change toujours : rien a annuler)
    for (Bitboard z = Z; z;)
    {
        const Square s = pop_lsb(z);
        for (int etat = 0; etat < 2; ++etat)
        {
            const Piece pc = etat ? pos.piece_on(s) : avant(s);
            if (pc == NO_PIECE || type_of(pc) == KING)
                continue;
            const Bitboard occ = etat ? occA : occB;
            const Bitboard att =
              (type_of(pc) == PAWN ? pawn_attacks_bb(color_of(pc), s) : attacks_bb(type_of(pc), s, occ)) & occ;
            for (Bitboard x = att; x;)
            {
                const Square to = pop_lsb(x);
                const Piece  q  = etat ? pos.piece_on(to) : avant(to);
                if (type_of(q) != KING && n < FT_MAXD)
                    t.dirty[n++] = code(s, to, pc, q, etat == 1);
            }
        }
    }
    // Autres pieces qui attaquent une case touchee : seules changent leurs menaces
    // vers les cases touchees et le long des lignes ouvertes ou fermees
    for (Bitboard b = A & ~Z & occA; b;)
    {
        const Square   s   = pop_lsb(b);
        const Piece    pc  = pos.piece_on(s);
        if (type_of(pc) == KING)
            continue;
        const bool     glisse = type_of(pc) >= BISHOP && type_of(pc) <= QUEEN;
        const Bitboard aB = (glisse ? attacks_bb(type_of(pc), s, occB)
                             : type_of(pc) == PAWN ? pawn_attacks_bb(color_of(pc), s) : attacks_bb(type_of(pc), s, 0)) & occB;
        const Bitboard aA = (glisse ? attacks_bb(type_of(pc), s, occA)
                             : type_of(pc) == PAWN ? pawn_attacks_bb(color_of(pc), s) : attacks_bb(type_of(pc), s, 0)) & occA;
        for (Bitboard x = (aB & ~aA) | (aB & aA & Z); x;)
        {
            const Square to = pop_lsb(x);
            const Piece  q  = avant(to);
            if (type_of(q) != KING && n < FT_MAXD)
                t.dirty[n++] = code(s, to, pc, q, false);
        }
        for (Bitboard x = (aA & ~aB) | (aB & aA & Z); x;)
        {
            const Square to = pop_lsb(x);
            const Piece  q  = pos.piece_on(to);
            if (type_of(q) != KING && n < FT_MAXD)
                t.dirty[n++] = code(s, to, pc, q, true);
        }
    }
#ifdef FT_PREFETCH
    // Prechargement des lignes que l'evaluation lira (table de 61 Mo, hors cache)
    for (int k = 0; k < n; ++k)
        for (Color c : {WHITE, BLACK})
        {
            const int idx = ft_index(t.dirty[k], c);
            if (idx < FULL_THREAT_DIM)
            {
                const char* r = reinterpret_cast<const char*>(ft_row(idx));
                for (int l = 0; l < HIDDEN_SIZE; l += 64)
                    __builtin_prefetch(r + l, 0, FT_PREFETCH);
            }
        }
#endif
#ifdef FT_STATS
    { static std::uint64_t mv = 0, tot = 0; static struct B { ~B() { std::fprintf(stderr, "FT_STATS coups %llu changements/coup %.1f\n", (unsigned long long) mv, double(tot) / std::max<std::uint64_t>(1, mv)); } } aff; ++mv; tot += n; }
#endif
    t.nd = n;
}

namespace {

// Accumulateur des menaces de la position courante (calcule au besoin) : on
// remonte jusqu'au dernier ancetre calcule en cumulant les changements de
// chaque coup, puis une seule passe ; sinon recalcul complet.
const ThreatAcc& threat_acc(const Position& pos) {
    PROF_SCOPE(NNUE_APPLY);
    if (!FtTab)
        init_full_threats();
    ThreatAcc* pool = ThrPool;
    if (!pool)
        pool = ThrPool = new ThreatAcc[ACC_POOL];
    StateInfo* st = pos.state();
    if (st->accIndex < 0)
    {
        thread_local ThreatAcc scratch;
        st->thrGen = 0;
        ThreatAcc& t = scratch;
        std::uint16_t lw[FT_MAXK], lb[FT_MAXK];
        int           nw, nbk;
        ft_lists(pos, lw, nw, lb, nbk);
        int aw[FT_MAXK], ab[FT_MAXK];
        for (int k = 0; k < nw; ++k) aw[k] = lw[k];
        for (int k = 0; k < nbk; ++k) ab[k] = lb[k];
        ft_apply(FtZero, t.v[WHITE], nullptr, 0, aw, nw);
        ft_apply(FtZero, t.v[BLACK], nullptr, 0, ab, nbk);
        return t;
    }
    ThreatAcc& t = pool[st->accIndex];
    if (st->accIndex >= 0 && st->thrGen == NetGen && t.owner == st)
        return t;

#ifdef FT_STATS
    static std::uint64_t cy[4] = {0, 0, 0, 0}, nca = 0;
    static struct C { ~C() { std::fprintf(stderr, "FT_CYCLES appels %llu chaine %.0f combinaison %.0f application %.0f dont-lecture %.0f\n", (unsigned long long) nca, double(cy[0]) / nca, double(cy[1]) / nca, double(cy[2]) / nca, double(cy[3]) / nca); } } affc;
    ++nca;
    std::uint64_t c0 = __rdtsc(), cc = 0, ca = 0;
#endif
    thread_local std::int8_t* cntTabTL = nullptr;
    std::int8_t* cntTab = cntTabTL;
    if (!cntTab)
        cntTab = cntTabTL = new std::int8_t[FULL_THREAT_DIM]();
    const StateInfo* chain[16];
    int              nc  = 0;
    const StateInfo* anc = nullptr;
    for (const StateInfo* s = st; s && s->accIndex >= 0 && nc < 16; s = s->previous)
    {
        const ThreatAcc& sl = pool[s->accIndex];
        if (s != st && s->thrGen == NetGen && sl.owner == s)
        {
            anc = s;
            break;
        }
        if (sl.dirtyOwner != s)
            break;
        chain[nc++] = s;
    }

#ifdef FT_STATS
    std::uint64_t c1 = __rdtsc();
    cy[0] += c1 - c0;
#endif
    for (Color c : {WHITE, BLACK})
    {
#ifdef FT_STATS
        std::uint64_t c2 = __rdtsc();
#endif
        int add[FT_COMB], sub[FT_COMB], na = 0, ns = 0;
        const std::int16_t* src = FtZero;
        bool                full = anc == nullptr;
        if (!full)
        {
            // changements cumules ; une menace ajoutee puis retiree s'annule
            int code[FT_COMB], m = 0;
#ifdef FT_STATS
            std::uint64_t g0 = __rdtsc();
#endif
            for (int k = 0; k < nc && !full; ++k)
            {
                const ThreatAcc& sl = pool[chain[k]->accIndex];
                if (m + sl.nd > FT_COMB)
                    full = true;
                for (int q = 0; q < sl.nd && !full; ++q)
                {
                    const int idx = ft_index(sl.dirty[q], c);
                    if (idx < FULL_THREAT_DIM)
                        code[m++] = idx * 2 + int(sl.dirty[q] >> 31);
                }
            }
#ifdef FT_STATS
            cy[3] += __rdtsc() - g0;
#endif
            if (!full && (nc == 1 || FT_SANS_ANNUL))   // sans tri : un ajout et un retrait de la meme ligne s annulent dans la somme
            {
                for (int q = 0; q < m; ++q)
                    if (code[q] & 1)
                        add[na++] = code[q] >> 1;
                    else
                        sub[ns++] = code[q] >> 1;
                src = pool[anc->accIndex].v[c];
            }
            else if (!full)
            {
                // Annulation sans tri : compteur par menace (60 Ko, remis a zero apres usage)
                std::int8_t* cnt = cntTab;
                for (int q = 0; q < m; ++q)
                    cnt[code[q] >> 1] += (code[q] & 1) ? 1 : -1;
                for (int q = 0; q < m; ++q)
                {
                    const int idx = code[q] >> 1;
                    for (int net = cnt[idx]; net > 0; --net)
                        add[na++] = idx;
                    for (int net = cnt[idx]; net < 0; ++net)
                        sub[ns++] = idx;
                    cnt[idx] = 0;
                }
                src = pool[anc->accIndex].v[c];
                if (na + ns > 48)   // repartir de zero coute moins cher
                    full = true;
            }
        }
        if (full)
        {
            std::uint16_t lw[FT_MAXK], lb[FT_MAXK];
            int           nw, nbk;
            ft_lists(pos, lw, nw, lb, nbk);
            src = FtZero, ns = 0, na = 0;
            const std::uint16_t* l = c == WHITE ? lw : lb;
            for (int k = 0; k < (c == WHITE ? nw : nbk); ++k)
                add[na++] = l[k];
        }
#ifdef FT_STATS
        { static std::uint64_t ev = 0, fu = 0, ro = 0, ch = 0;
          static struct A { ~A() { std::fprintf(stderr, "FT_STATS persp-evals %llu complets %.1f %% lignes %.1f chaine %.2f\n", (unsigned long long) ev, 100.0 * fu / std::max<std::uint64_t>(1, ev), double(ro) / std::max<std::uint64_t>(1, ev), double(ch) / std::max<std::uint64_t>(1, ev)); } } aff;
          ++ev; fu += full; ro += na + ns; ch += nc; }
#endif
#ifdef FT_STATS
        std::uint64_t c3 = __rdtsc();
        cc += c3 - c2;
#endif
#ifndef FT_NOAPPLY
        ft_apply(src, t.v[c], sub, ns, add, na);
#ifdef FT_STATS
        ca += __rdtsc() - c3;
#endif
#else
        (void) src;
#endif
    }
#ifdef FT_STATS
    cy[1] += cc, cy[2] += ca;
#endif
    t.owner    = st;
    st->thrGen = NetGen;
#ifndef NDEBUG
    // Preuve : le calcul par changements egale le recalcul complet
    {
        thread_local ThreatAcc ref;
        std::uint16_t          lw[FT_MAXK], lb[FT_MAXK];
        int                    nw, nbk;
        ft_lists(pos, lw, nw, lb, nbk);
        for (Color c : {WHITE, BLACK})
        {
            int add[FT_MAXK];
            const int n = c == WHITE ? nw : nbk;
            for (int k = 0; k < n; ++k)
                add[k] = (c == WHITE ? lw : lb)[k];
            ft_apply(FtZero, ref.v[c], nullptr, 0, add, n);
            assert(std::memcmp(ref.v[c], t.v[c], sizeof(ref.v[c])) == 0 && "menaces desynchronisees");
        }
    }
#endif
    return t;
}

// Sortie : accumulateur des pieces + accumulateur des menaces, puis la sortie habituelle
Value output_full(const Accumulator& acc, const Position& pos, int bucket) {
    const ThreatAcc&       t = threat_acc(pos);
    alignas(64) Accumulator sum;
    for (Color c : {WHITE, BLACK})
    {
#if defined(__AVX2__)
        for (int j = 0; j < HIDDEN_SIZE; j += 16)
            _mm256_store_si256(reinterpret_cast<__m256i*>(sum.v[c] + j),
                               _mm256_add_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(acc.v[c] + j)),
                                                _mm256_load_si256(reinterpret_cast<const __m256i*>(t.v[c] + j))));
#elif defined(ONE_NEON)
        for (int j = 0; j < HIDDEN_SIZE; j += 8)
            vst1q_s16(sum.v[c] + j, vaddq_s16(vld1q_s16(acc.v[c] + j), vld1q_s16(t.v[c] + j)));
#else
        for (int j = 0; j < HIDDEN_SIZE; ++j)
            sum.v[c][j] = std::int16_t(acc.v[c][j] + t.v[c][j]);
#endif
    }
    return output(sum, pos.side_to_move(), bucket);
}
}  // namespace

// Premiere couche cachee apres produit par paires (Stockfish) :
//   p  = (clamp(a[i], 0, QA) * clamp(a[i + H/2], 0, QA)) >> 9   dans [0, QH]
//   x1 = clamp((W1 p + b1) / QB, 0, QH)                          W1 en int8
// Calcul creux par groupes de 4 entrees (maddubs) : exact, en entiers.
void pairmul_l1(const Accumulator& acc, Color stm, int bucket, std::int32_t* x1) {
    const int l1 = Net->l1;
    alignas(32) std::uint8_t in[HIDDEN_SIZE];
    const std::int8_t*  wt = Net->h1t8 + std::size_t(bucket) * (HIDDEN_SIZE / 4) * l1 * 4;
    const std::int32_t* b1 = Net->h1b.data() + std::size_t(bucket) * l1;
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa   = _mm256_set1_epi16(QA);
    for (int h = 0; h < 2; ++h)
    {
        const std::int16_t* src = acc.v[h == 0 ? stm : ~stm];
        for (int i = 0; i < HIDDEN_SIZE / 2; i += 32)
        {
            __m256i p[2];
            for (int k = 0; k < 2; ++k)
            {
                __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(src + i + 16 * k));
                __m256i b = _mm256_load_si256(
                  reinterpret_cast<const __m256i*>(src + HIDDEN_SIZE / 2 + i + 16 * k));
                a    = _mm256_min_epi16(_mm256_max_epi16(a, zero), qa);
                b    = _mm256_min_epi16(_mm256_max_epi16(b, zero), qa);
                p[k] = _mm256_srli_epi16(_mm256_mullo_epi16(a, b), 9);
            }
            const __m256i packed =
              _mm256_permute4x64_epi64(_mm256_packus_epi16(p[0], p[1]), 0xD8);
            _mm256_store_si256(reinterpret_cast<__m256i*>(in + h * HIDDEN_SIZE / 2 + i), packed);
        }
    }
    std::uint16_t nnz[HIDDEN_SIZE / 4];
    int           n = 0;
    for (int i = 0; i < HIDDEN_SIZE; i += 32)
    {
        const __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(in + i));
        unsigned      m =
          ~unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, zero)))) & 0xFF;
        while (m)
        {
            nnz[n++] = std::uint16_t(i / 4 + __builtin_ctz(m));
            m &= m - 1;
        }
    }
#ifdef ONE_NNZ_STATS
    // Diagnostic (phase 10, A3) : part des groupes de 4 entrees non nuls et
    // des octets non nuls, affichee a la sortie du programme.
    {
        struct Stats {
            std::uint64_t groups = 0, nzGroups = 0, bytes = 0, nzBytes = 0;
            ~Stats() {
                std::fprintf(stderr, "NNZ groupes %.2f %% non nuls, octets %.2f %% non nuls (%llu evals)\n",
                             100.0 * double(nzGroups) / double(groups ? groups : 1),
                             100.0 * double(nzBytes) / double(bytes ? bytes : 1),
                             (unsigned long long)(groups / (HIDDEN_SIZE / 4)));
            }
        };
        static Stats st;
        st.groups += HIDDEN_SIZE / 4;
        st.nzGroups += std::uint64_t(n);
        st.bytes += HIDDEN_SIZE;
        for (int i = 0; i < HIDDEN_SIZE; ++i)
            st.nzBytes += in[i] != 0;
        // ONE_NNZ_DUMP=fichier : masques des entrees non nulles (1 bit par
        // entree, les deux perspectives), pour tools/ftperm.py.
        static std::FILE* dump = std::getenv("ONE_NNZ_DUMP") ? std::fopen(std::getenv("ONE_NNZ_DUMP"), "wb")
                                                              : nullptr;
        if (dump)
        {
            std::uint8_t bits[HIDDEN_SIZE / 8] = {};
            for (int i = 0; i < HIDDEN_SIZE; ++i)
                bits[i / 8] |= std::uint8_t((in[i] != 0) << (i % 8));
            std::fwrite(bits, 1, sizeof(bits), dump);
        }
    }
#endif
    const int     nr   = l1 / 8;
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i       sum[MAX_L1 / 8];
    for (int r = 0; r < nr; ++r)
        sum[r] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b1 + 8 * r));
    const std::int32_t* in32 = reinterpret_cast<const std::int32_t*>(in);
    for (int t = 0; t < n; ++t)
    {
        const int          c = nnz[t];
        const __m256i      v = _mm256_set1_epi32(in32[c]);
        const std::int8_t* w = wt + std::size_t(c) * l1 * 4;
        for (int r = 0; r < nr; ++r)
        {
            const __m256i wv = _mm256_load_si256(reinterpret_cast<const __m256i*>(w + 32 * r));
            sum[r] = _mm256_add_epi32(sum[r], _mm256_madd_epi16(_mm256_maddubs_epi16(v, wv), ones));
        }
    }
    alignas(32) std::int32_t s1[MAX_L1];
    for (int r = 0; r < nr; ++r)
        _mm256_store_si256(reinterpret_cast<__m256i*>(s1 + 8 * r), sum[r]);
    for (int j = 0; j < l1; ++j)
        x1[j] = std::clamp(s1[j] / QB, 0, QH);
#else
    for (int h = 0; h < 2; ++h)
    {
        const std::int16_t* src = acc.v[h == 0 ? stm : ~stm];
        for (int i = 0; i < HIDDEN_SIZE / 2; ++i)
        {
            const int a = std::clamp<int>(src[i], 0, QA);
            const int b = std::clamp<int>(src[i + HIDDEN_SIZE / 2], 0, QA);
            in[h * HIDDEN_SIZE / 2 + i] = std::uint8_t((a * b) >> 9);
        }
    }
    for (int j = 0; j < l1; ++j)
    {
        std::int32_t s = b1[j];
        for (int c = 0; c < HIDDEN_SIZE / 4; ++c)
            for (int k = 0; k < 4; ++k)
                s += in[4 * c + k] * wt[(std::size_t(c) * l1 + j) * 4 + k];
        x1[j] = std::clamp(s / QB, 0, QH);
    }
#endif
}

// Couches cachees, en entiers. Meme formule que tools/check_nnue.py :
//   a  = clamp(acc, 0, QA)                      (2 * HIDDEN_SIZE entrees)
//   x1 = clamp(s1 * QH / (QA * QB), 0, QH)      s1 = W1 a + b1
//   x2 = clamp(s2 / QB, 0, QH)                  s2 = W2 x1 + b2
//   cp = s3 * SCALE / (QH * QB)                 s3 = W3 x2 + b3
Value output_hidden(const Accumulator& acc, Color stm, int bucket) {
    PROF_SCOPE(NNUE_OUT);
    const int l1 = Net->l1, l2 = Net->l2;
    const std::int16_t* w1 = Net->h1w + std::size_t(bucket) * l1 * 2 * HIDDEN_SIZE;

    std::int32_t x1[MAX_L1];
    if (Net->pairmul)
        pairmul_l1(acc, stm, bucket, x1);
    else
    {
#if defined(__AVX2__)
    // Calcul creux (Stockfish) : les activations sont ecretees une fois, on
    // ne garde que les paires d'entrees non nulles, et chacune s'ajoute aux l1
    // neurones a la fois. Sommes entieres : resultat identique au calcul dense.
    (void) w1;
    alignas(32) std::int16_t a[2 * HIDDEN_SIZE];
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa   = _mm256_set1_epi16(QA);
    for (int h = 0; h < 2; ++h)
    {
        const std::int16_t* src = acc.v[h == 0 ? stm : ~stm];
        for (int i = 0; i < HIDDEN_SIZE; i += 16)
        {
            __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(src + i));
            v = _mm256_min_epi16(_mm256_max_epi16(v, zero), qa);
            _mm256_store_si256(reinterpret_cast<__m256i*>(a + h * HIDDEN_SIZE + i), v);
        }
    }
    std::uint16_t nnz[HIDDEN_SIZE];   // paires non nulles (2 * HIDDEN_SIZE / 2 au plus)
    int           n = 0;
    for (int i = 0; i < 2 * HIDDEN_SIZE; i += 16)
    {
        const __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(a + i));
        unsigned m = ~unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, zero)))) & 0xFF;
        while (m)
        {
            nnz[n++] = std::uint16_t(i / 2 + __builtin_ctz(m));
            m &= m - 1;
        }
    }
    const int      nr = l1 / 8;
    __m256i        sum[MAX_L1 / 8];
    for (int r = 0; r < nr; ++r)
        sum[r] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Net->h1b.data() + std::size_t(bucket) * l1 + 8 * r));
    const std::int32_t* a32 = reinterpret_cast<const std::int32_t*>(a);
    const std::int16_t* wt  = Net->h1t + std::size_t(bucket) * HIDDEN_SIZE * l1 * 2;
    for (int t = 0; t < n; ++t)
    {
        const int           p  = nnz[t];
        const __m256i       ap = _mm256_set1_epi32(a32[p]);
        const std::int16_t* w  = wt + std::size_t(p) * l1 * 2;
        for (int r = 0; r < nr; ++r)
            sum[r] = _mm256_add_epi32(sum[r], _mm256_madd_epi16(ap,
                       _mm256_load_si256(reinterpret_cast<const __m256i*>(w + 16 * r))));
    }
    alignas(32) std::int32_t s1[MAX_L1];
    for (int r = 0; r < nr; ++r)
        _mm256_store_si256(reinterpret_cast<__m256i*>(s1 + 8 * r), sum[r]);
    for (int j = 0; j < l1; ++j)
        x1[j] = int(std::clamp<std::int64_t>(std::int64_t(s1[j]) * QH / (QA * QB), 0, QH));
#else
    for (int j = 0; j < l1; ++j)
    {
        const std::int16_t* w = w1 + std::size_t(j) * 2 * HIDDEN_SIZE;
        const std::int64_t  s = std::int64_t(forward_half(acc.v[stm], w))
                             + forward_half(acc.v[~stm], w + HIDDEN_SIZE)
                             + Net->h1b[std::size_t(bucket) * l1 + j];
        x1[j] = int(std::clamp<std::int64_t>(s * QH / (QA * QB), 0, QH));
    }
#endif
    }

    alignas(32) std::int32_t x2[MAX_L2];
#if defined(__AVX2__)
    {
        // Paires de x1 diffusees, poids [paire][l2][2] : l2 / 8 sommes a la fois
        const int           nr = l2 / 8;
        const std::int16_t* wt = Net->h2t.data() + std::size_t(bucket) * (l1 / 2) * l2 * 2;
        __m256i             sum[MAX_L2 / 8];
        for (int r = 0; r < nr; ++r)
            sum[r] = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(Net->h2b.data() + std::size_t(bucket) * l2 + 8 * r));
        for (int p = 0; p < l1 / 2; ++p)
        {
            const __m256i v = _mm256_set1_epi32(x1[2 * p] | (x1[2 * p + 1] << 16));
            const std::int16_t* w = wt + std::size_t(p) * l2 * 2;
            for (int r = 0; r < nr; ++r)
                sum[r] = _mm256_add_epi32(sum[r], _mm256_madd_epi16(v,
                           _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 16 * r))));
        }
        for (int r = 0; r < nr; ++r)
            _mm256_store_si256(reinterpret_cast<__m256i*>(x2 + 8 * r), sum[r]);
        for (int k = 0; k < l2; ++k)
            x2[k] = std::clamp(x2[k] / QB, 0, QH);
    }
#else
    for (int k = 0; k < l2; ++k)
    {
        const std::int16_t* w = Net->h2w.data() + (std::size_t(bucket) * l2 + k) * l1;
        std::int32_t        s = Net->h2b[std::size_t(bucket) * l2 + k];
        for (int j = 0; j < l1; ++j)
            s += x1[j] * w[j];
        x2[k] = std::clamp(s / QB, 0, QH);
    }
#endif

    const std::int16_t* w3 = Net->h3w.data() + std::size_t(bucket) * l2;
    std::int64_t        s3 = Net->h3b[bucket];
    for (int k = 0; k < l2; ++k)
        s3 += x2[k] * w3[k];
    const int cp = int(s3 * SCALE / (QH * QB));
    return Value(std::clamp(cp, -12000, 12000));
}

Value output(const Accumulator& acc, Color stm, int bucket) {
    if (Net->l1)
        return output_hidden(acc, stm, bucket);
    PROF_SCOPE(NNUE_OUT);
    const std::int16_t* w = Net->outputWeights + std::size_t(bucket) * 2 * HIDDEN_SIZE;

    // Le biais de sortie est exprime en unites QA * QB dans les deux cas :
    // en SCReLU la somme arrive en QA^2 * QB et doit d'abord perdre un QA.
    // La division tronque vers zero, comme en C ; tools/check_nnue.py reproduit
    // exactement cette troncature.
    const std::int64_t sum =
      Net->activation == ACT_SCRELU
        ? (forward_half_sq(acc.v[stm], w) + forward_half_sq(acc.v[~stm], w + HIDDEN_SIZE)) / QA
        : std::int64_t(forward_half(acc.v[stm], w) + forward_half(acc.v[~stm], w + HIDDEN_SIZE));

    const int cp = int((sum + Net->outputBias[bucket]) * SCALE / (QA * QB));

    // Un reseau mal quantifie ne doit pas pouvoir empoisonner la recherche
    return Value(std::clamp(cp, -12000, 12000));
}

// Evaluation paresseuse : on remonte jusqu'au dernier ancetre dont
// l'accumulateur est valide, et on n'applique que les deltas necessaires.
// Au-dela de quelques plis, repartir de zero coute moins cher.
Value evaluate(const Position& pos) {
    StateInfo* st = pos.state();

    // Aucune place dans le reservoir : on repart de zero dans un accumulateur
    // de service. Cela ne peut arriver que sur une partie plus longue que le
    // reservoir, jamais dans la recherche.
    if (st->accIndex < 0)
    {
        thread_local Accumulator scratch;
        refresh(scratch, pos);
        return Net->fullThreats ? output_full(scratch, pos, current_bucket(pos))
             : Net->threats     ? output_threats(scratch, pos, current_bucket(pos))
                                : output(scratch, pos.side_to_move(), current_bucket(pos));
    }

    if (st->accGen != NetGen)
    {
        StateInfo* chain[8];
        int        n = 0;
        StateInfo* s = st;

        while (n < 7 && s->previous && s->previous->accIndex >= 0
               && s->previous->accGen != NetGen)
        {
            chain[n++] = s;
            s          = s->previous;
        }

        const bool haveAncestor = s->previous && s->previous->accIndex >= 0
                               && s->previous->accGen == NetGen;
        if (haveAncestor)
            chain[n++] = s;

        // Une perspective dont le roi a change de bucket doit etre recalculee ;
        // faute de plateau pour les noeuds intermediaires, on le fait ici.
        bool refreshSide[COLOR_NB] = {!haveAncestor, !haveAncestor};

        if (haveAncestor)
            for (int i = 0; i < n; ++i)
                for (Color c : {WHITE, BLACK})
                    if (requires_refresh(chain[i]->dirtyPiece, c))
                        refreshSide[c] = true;

        // Mode materiel : si une prise de la chaine a fait changer de bande,
        // tout l'accumulateur change d'expert, pour les deux perspectives.
        // Le nombre de pieces ne fait que baisser : comparer les deux bouts
        // de la chaine suffit.
        if (haveAncestor && Net->material)
        {
            int removed = 0;
            for (int i = 0; i < n; ++i)
            {
                const DirtyPiece& dp = chain[i]->dirtyPiece;
                for (int k = 0; k < dp.count; ++k)
                    removed += (dp.to[k] == SQ_NONE) - (dp.from[k] == SQ_NONE);
            }
            const int now = popcount(pos.pieces());
            if (material_bucket(now) != material_bucket(now + removed))
                refreshSide[WHITE] = refreshSide[BLACK] = true;
        }

        // Pions de chaque noeud de la chaine, retrouves en defaisant les coups
        // depuis la position courante (les paires ne dependent que d'eux).
        PawnSets ps[8];
        if (haveAncestor && Net->pairs)
        {
            Bitboard cur[COLOR_NB] = {pos.pieces(WHITE, PAWN), pos.pieces(BLACK, PAWN)};
            for (int i = 0; i < n; ++i)
            {
                ps[i].after[WHITE] = cur[WHITE];
                ps[i].after[BLACK] = cur[BLACK];
                const DirtyPiece& dp = chain[i]->dirtyPiece;
                for (int k = 0; k < dp.count; ++k)
                    if (type_of(dp.piece[k]) == PAWN)
                    {
                        const Color c = color_of(dp.piece[k]);
                        if (dp.to[k] != SQ_NONE)
                            cur[c] &= ~(Bitboard(1) << dp.to[k]);
                        if (dp.from[k] != SQ_NONE)
                            cur[c] |= Bitboard(1) << dp.from[k];
                    }
                ps[i].before[WHITE] = cur[WHITE];
                ps[i].before[BLACK] = cur[BLACK];
            }
        }

        if (haveAncestor)
            for (int i = n - 1; i >= 0; --i)
                apply(pos, chain[i], !refreshSide[WHITE], !refreshSide[BLACK],
                      Net->pairs ? &ps[i] : nullptr);

        for (Color c : {WHITE, BLACK})
            if (refreshSide[c])
                refresh_side_cached(pos.accumulator(st->accIndex), pos, c);

        // Les noeuds intermediaires ne sont declares valides que si les DEUX
        // perspectives y ont ete calculees par delta.
        if (haveAncestor && !refreshSide[WHITE] && !refreshSide[BLACK])
            for (int i = 0; i < n; ++i)
                chain[i]->accGen = NetGen;

        st->accGen = NetGen;

#ifndef NDEBUG
        // Preuve, et non estimation : l'accumulateur doit etre identique bit
        // pour bit a un recalcul complet.
        {
            Accumulator ref;
            refresh(ref, pos);
            assert(std::memcmp(&ref, &pos.accumulator(st->accIndex), sizeof(Accumulator)) == 0
                   && "accumulateur incremental desynchronise");
        }
#endif
    }

    return Net->fullThreats ? output_full(pos.accumulator(st->accIndex), pos, current_bucket(pos))
         : Net->threats     ? output_threats(pos.accumulator(st->accIndex), pos, current_bucket(pos))
                            : output(pos.accumulator(st->accIndex), pos.side_to_move(), current_bucket(pos));
}

}  // namespace NNUE
}  // namespace one
