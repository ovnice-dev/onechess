#include "nnue.h"
#include "prof.h"

#include <algorithm>
#include <iostream>
#include <cassert>
#include <cstring>
#include <fstream>
#include <sstream>

#include "bitboard.h"
#include "misc.h"
#include "position.h"

#if defined(__AVX2__)
    #include <immintrin.h>
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

    // [outBuckets][2 * HIDDEN_SIZE], sur le tas pour rester aligne
    std::int16_t* outputWeights = nullptr;
    std::int32_t  outputBias[MAX_OUT_BUCKETS] = {};

    ~Network() {
        aligned_large_pages_free(featureWeights);
        aligned_large_pages_free(featureWeights8);
        aligned_large_pages_free(pairWeights);
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
#else
using FtW = std::int16_t;
    #define FT_LOAD(p) _mm256_load_si256(reinterpret_cast<const __m256i*>(p))
#endif

inline void add_feature(std::int16_t* acc, const FtW* w) {
#if defined(__AVX2__)
    for (int i = 0; i < HIDDEN_SIZE; i += 16)
    {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
        __m256i b = FT_LOAD(w + i);
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_add_epi16(a, b));
    }
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
        const bool update = (p == WHITE ? doWhite : doBlack) && dp.count > 0;
        if (!update)
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
    NetPath.clear();
    ++NetGen;
}

std::string info() {
    if (!Net)
        return "NNUE absente (evaluation PSQT)";

    std::ostringstream ss;
    ss << "NNUE " << Net->nBuckets << "x768x" << Net->hidden << "x2->"
       << Net->outBuckets << (Net->mirror ? " miroir" : "") << (Net->material ? " experts-materiel" : "")
       << (Net->pairs ? " paires-de-pions" : "")
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
    return std::clamp((pieceCount - 2) / 8, 0, Net->nBuckets - 1);
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

bool load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
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

    // Bit 8 du champ activation : le reseau a des entrees paires de pions
    const bool pairs = activation & PAIRS_FLAG;
    activation &= ~PAIRS_FLAG;

    if (!f || (activation != ACT_CRELU && activation != ACT_SCRELU))
        return false;

    auto* net       = new Network();
    net->hidden     = int(hidden);
    net->nBuckets   = int(nBuckets);
    net->material   = mirror == 2;
    net->mirror     = mirror == 1;
    net->outBuckets = int(outBuckets);
    net->activation = activation;
    net->pairs      = pairs;

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
    f.read(reinterpret_cast<char*>(net->outputWeights), std::streamsize(oBytes));
    f.read(reinterpret_cast<char*>(net->outputBias),
           std::streamsize(net->outBuckets * sizeof(std::int32_t)));

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
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < HIDDEN_SIZE; ++j)
                acc.v[p][j] = std::int16_t(acc.v[p][j] + pair_weights(idx[k])[j]);
    }
}

void refresh(Accumulator& acc, const Position& pos) {
    refresh_side(acc, pos, WHITE);
    refresh_side(acc, pos, BLACK);
}

// Bucket de sortie de la position courante.
int current_bucket(const Position& pos) {
    return output_bucket(popcount(pos.pieces()), Net->outBuckets);
}

Value output(const Accumulator& acc, Color stm, int bucket) {
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
        return output(scratch, pos.side_to_move(), current_bucket(pos));
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
                refresh_side(pos.accumulator(st->accIndex), pos, c);

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

    return output(pos.accumulator(st->accIndex), pos.side_to_move(), current_bucket(pos));
}

}  // namespace NNUE
}  // namespace one
