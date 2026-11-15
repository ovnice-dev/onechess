#include "perft.h"

#include <chrono>
#include <iostream>
#include <vector>

#include "movegen.h"
#include "uci.h"

namespace one {

namespace {

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

struct PerftCase {
    const char*   fen;
    Depth         depth;
    std::uint64_t expected;
    const char*   nom;
};

// Les six positions de reference de la Chess Programming Wiki, puis la liste
// des pieges connus : prise en passant illegale, roque donnant echec,
// promotions, pat, double echec.
const std::vector<PerftCase> Suite = {
  {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324, "position initiale"},
  {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690,
   "Kiwipete"},
  {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 7, 178633661, "finale de pions"},
  {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 6, 706045033, "promotions"},
  {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 5, 89941194, "position 5"},
  {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 5, 164075551,
   "position 6"},
  {"3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1", 6, 1134888, "prise en passant illegale 1"},
  {"8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1", 6, 1015133, "prise en passant illegale 2"},
  {"8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1", 6, 1440467, "prise en passant donnant echec"},
  {"5k2/8/8/8/8/8/8/4K2R w K - 0 1", 6, 661072, "petit roque donnant echec"},
  {"3k4/8/8/8/8/8/8/R3K3 w Q - 0 1", 6, 803711, "grand roque donnant echec"},
  {"r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1", 4, 1274206, "droits de roque"},
  {"r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1", 4, 1720476, "droits de roque perdus"},
  {"2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1", 6, 3821001, "promotion donnant echec"},
  {"8/8/1P2K3/8/2n5/1q6/8/5k2 b - - 0 1", 5, 1004658, "promotion capturante"},
  {"4k3/1P6/8/8/8/8/K7/8 w - - 0 1", 6, 217342, "sous-promotion echec"},
  {"8/P1k5/K7/8/8/8/8/8 w - - 0 1", 6, 92683, "auto-pat par promotion"},
  {"K1k5/8/P7/8/8/8/8/8 w - - 0 1", 6, 2217, "pat"},
  {"8/k1P5/8/1K6/8/8/8/8 w - - 0 1", 7, 567584, "double echec par promotion"},
  {"8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1", 4, 23527, "double echec"},
};

}  // namespace

std::uint64_t perft(Position& pos, Depth depth) {
    StateInfo     st;
    std::uint64_t nodes = 0;
    const bool    leaf  = (depth == 2);

    for (const auto& m : MoveList<LEGAL>(pos))
    {
        pos.do_move(m, st);
        nodes += leaf ? MoveList<LEGAL>(pos).size() : perft(pos, depth - 1);
        pos.undo_move(m);
    }

    return nodes;
}

void perft_divide(Position& pos, Depth depth) {
    StateInfo     st;
    std::uint64_t total = 0;
    const auto    t0    = std::chrono::steady_clock::now();

    for (const auto& m : MoveList<LEGAL>(pos))
    {
        std::uint64_t cnt = 1;

        if (depth > 1)
        {
            pos.do_move(m, st);
            cnt = depth == 2 ? MoveList<LEGAL>(pos).size() : perft(pos, depth - 1);
            pos.undo_move(m);
        }

        total += cnt;
        std::cout << UCI::move(m, pos.is_chess960()) << ": " << cnt << std::endl;
    }

    const double ms = ms_since(t0);
    std::cout << "\nNoeuds: " << total << "   " << int(ms) << " ms   "
              << std::uint64_t(ms > 0 ? total / ms : 0) << " knps" << std::endl;
}

int perft_suite() {
    Position      pos;
    StateInfo     si;
    int           echecs = 0;
    std::uint64_t total  = 0;
    const auto    t0     = std::chrono::steady_clock::now();

    for (const auto& c : Suite)
    {
        pos.set(c.fen, false, &si);

        const auto          t1 = std::chrono::steady_clock::now();
        const std::uint64_t n  = perft(pos, c.depth);
        const double        ms = ms_since(t1);

        total += n;
        const bool ok = (n == c.expected);
        echecs += !ok;

        std::cout << (ok ? "  OK  " : " ECHEC") << "  d" << c.depth << "  " << n;
        if (!ok)
            std::cout << "  (attendu " << c.expected << ")";
        std::cout << "   " << int(ms) << " ms   " << std::uint64_t(ms > 0 ? n / ms : 0)
                  << " knps   " << c.nom << std::endl;
    }

    const double ms = ms_since(t0);
    std::cout << "\n" << Suite.size() - echecs << "/" << Suite.size() << " positions correctes   "
              << total << " noeuds en " << int(ms) << " ms   " << std::uint64_t(total / ms)
              << " knps global" << std::endl;

    return echecs;
}

}  // namespace one
