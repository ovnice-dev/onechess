// Convertit un .binpack de Stockfish (donnees publiques reetiquetees par
// Leela BT4, voir RECHERCHES 9 ter) en enregistrements Onechess de 32 octets.
//
// Le lecteur binpack (lib/) est celui de nnue-pytorch (GPLv3, comme Onechess),
// copie sans modification depuis official-stockfish/nnue-pytorch, commit
// 568c7fb, data_loader/cpp/lib.
//
// Filtres, ceux du chargeur de Stockfish : score absent (32002), zero de
// remplissage, coup joue = prise, trait en echec, debut de partie (--ply-min).
// --garde p ne garde qu'une position sur 1/p, tiree au hasard : un binpack de
// 6 Go fait ~2 G positions, soit ~70 Go a 32 octets.
// La sortie est decoupee en fichiers de --par positions (prefixe_0000.bin...).
//
//   binpack_vers_bin fichier.binpack prefixe [--garde 0.1] [--par 50000000]
//                    [--ply-min 16] [--max N] [--graine 1] [--fens N]
//
// Score ecrit : centipions du point de vue des blancs, a l'echelle de
// Stockfish (100 * valeur interne / 208, comme le modele de victoire du
// chargeur). Resultat : 0 = noirs gagnent, 1 = nulle, 2 = blancs gagnent.

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>

#include "lib/binpack.h"

using namespace chess;

namespace {

struct Record {
    std::uint64_t occ;
    std::uint8_t  pcs[16];
    std::int16_t  score;
    std::uint8_t  stm, result, rule50;
    std::uint8_t  pad[3];
};
static_assert(sizeof(Record) == 32, "32 octets");

void pack(const binpack::TrainingDataEntry& e, Record& r) {
    std::memset(&r, 0, sizeof r);
    int k = 0;
    for (int s = 0; s < 64; ++s) {
        const Piece pc = e.pos.pieceAt(Square(s));
        if (pc == Piece::none())
            continue;
        r.occ |= std::uint64_t(1) << s;
        const int code = ordinal(pc.type()) + 6 * ordinal(pc.color());
        r.pcs[k >> 1] |= std::uint8_t(code << ((k & 1) * 4));
        ++k;
    }
    const bool blanc = e.pos.sideToMove() == Color::White;
    long cp = std::lround(100.0 * e.score / 208.0);
    if (cp > 32000) cp = 32000;
    if (cp < -32000) cp = -32000;
    r.score  = std::int16_t(blanc ? cp : -cp);
    r.stm    = blanc ? 0 : 1;
    const int res = blanc ? e.result : -e.result;   // resultat vu des blancs
    r.result = std::uint8_t(res + 1);
    r.rule50 = std::uint8_t(std::min<int>(e.pos.rule50Counter(), 255));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage : binpack_vers_bin fichier.binpack prefixe [options]\n");
        return 1;
    }
    const std::string src = argv[1], prefixe = argv[2];
    double garde = 1.0;
    long long par = 50000000, maxi = -1;
    int ply_min = 16;
    unsigned graine = 1;
    long long fens = 0;   // controle : FEN du lecteur officiel des N premieres positions ecrites
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string o = argv[i];
        if (o == "--garde") garde = std::atof(argv[i + 1]);
        else if (o == "--par") par = std::atoll(argv[i + 1]);
        else if (o == "--max") maxi = std::atoll(argv[i + 1]);
        else if (o == "--ply-min") ply_min = std::atoi(argv[i + 1]);
        else if (o == "--graine") graine = unsigned(std::atoi(argv[i + 1]));
        else if (o == "--fens") fens = std::atoll(argv[i + 1]);
        else { std::fprintf(stderr, "option inconnue : %s\n", o.c_str()); return 1; }
    }

    std::mt19937_64 rng(graine);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    binpack::CompressedTrainingDataEntryReader lecteur(src, std::ios_base::in | std::ios_base::binary);

    const auto t0 = std::chrono::steady_clock::now();
    long long lues = 0, ecrites = 0, rejet_score = 0, rejet_tactique = 0, rejet_debut = 0;
    int last_ply = -1, last_score = 32002;
    std::FILE* out = nullptr;
    int part = 0;
    Record r;
    while (lecteur.hasNext()) {
        const binpack::TrainingDataEntry e = lecteur.next();
        ++lues;
        const bool zero_remplissage = e.ply > last_ply && last_score != 32002
                                   && std::abs(last_score) > 100 && e.result != 0 && e.score == 0;
        last_ply = e.ply;
        if (e.score == 32002 || zero_remplissage) { ++rejet_score; continue; }
        last_score = e.score;
        if (e.ply <= ply_min) { ++rejet_debut; continue; }
        if (e.isCapturingMove() || e.isInCheck()) { ++rejet_tactique; continue; }
        if (garde < 1.0 && u(rng) >= garde) continue;

        if (!out || ecrites % par == 0) {
            if (out) std::fclose(out);
            char nom[1024];
            std::snprintf(nom, sizeof nom, "%s_%04d.bin", prefixe.c_str(), part++);
            out = std::fopen(nom, "wb");
            if (!out) { std::fprintf(stderr, "impossible d'ecrire %s\n", nom); return 1; }
        }
        pack(e, r);
        std::fwrite(&r, sizeof r, 1, out);
        if (ecrites < fens)
            std::printf("FEN %s|%d|%d|%d\n", e.pos.fen().c_str(), int(e.score), int(e.result), int(e.ply));
        ++ecrites;
        if (ecrites % 10000000 == 0) {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("%lld lues, %lld ecrites, %.0f lues/s\n", lues, ecrites, lues / s);
            std::fflush(stdout);
        }
        if (maxi >= 0 && ecrites >= maxi) break;
    }
    if (out) std::fclose(out);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("FIN : %lld lues, %lld ecrites en %d fichiers ; rejets : score %lld, prise/echec %lld, "
                "debut %lld ; %.0f s (%.0f lues/s)\n",
                lues, ecrites, part, rejet_score, rejet_tactique, rejet_debut, s, lues / s);
    return 0;
}
