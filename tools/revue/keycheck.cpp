// Banc de coherence : parties aleatoires, verification a chaque coup de tout
// ce que do_move() maintient incrementalement, contre un recalcul complet
// (set(fen())). Verifie aussi undo_move(), gives_check() et pseudo_legal().
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "bitboard.h"
#include "movegen.h"
#include "position.h"
#include "uci.h"

using namespace one;

static std::mt19937_64 rng(12345);

static std::string chess960_fen(int n) {
    // Numerotation de Scharnagl
    char  rank[8];
    bool  used[8] = {};
    int   b1 = n % 4; n /= 4;  // fou cases claires : b,d,f,h
    rank[2 * b1 + 1] = 'B'; used[2 * b1 + 1] = true;
    int   b2 = n % 4; n /= 4;  // fou cases sombres : a,c,e,g
    rank[2 * b2] = 'B'; used[2 * b2] = true;
    int   q = n % 6; n /= 6;
    auto place = [&](int k, char c) {
        for (int i = 0; i < 8; ++i)
            if (!used[i] && k-- == 0) { rank[i] = c; used[i] = true; return; }
    };
    place(q, 'Q');
    static const int knights[10][2] = {{0,1},{0,2},{0,3},{0,4},{1,2},{1,3},{1,4},{2,3},{2,4},{3,4}};
    int k1 = knights[n][0], k2 = knights[n][1];
    // placer les cavaliers sur les cases libres k1 et k2 (indices parmi les libres)
    std::vector<int> fr;
    for (int i = 0; i < 8; ++i) if (!used[i]) fr.push_back(i);
    rank[fr[k1]] = 'N'; used[fr[k1]] = true;
    rank[fr[k2]] = 'N'; used[fr[k2]] = true;
    fr.clear();
    for (int i = 0; i < 8; ++i) if (!used[i]) fr.push_back(i);
    rank[fr[0]] = 'R'; rank[fr[1]] = 'K'; rank[fr[2]] = 'R';
    std::string w(rank, 8), b = w;
    for (auto& c : b) c = char(tolower(c));
    std::string cast;
    cast += char('A' + fr[2]); cast += char('A' + fr[0]);
    cast += char('a' + fr[2]); cast += char('a' + fr[0]);
    return b + "/pppppppp/8/8/8/8/PPPPPPPP/" + w + " w " + cast + " - 0 1";
}

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { ++fails; if (fails < 30) std::printf("ECHEC %s : %s\n  fen %s\n", msg, #c, pos.fen().c_str()); } } while (0)

int main(int argc, char** argv) {
    Bitboards::init();
    Position::init();
    int games = argc > 1 ? std::atoi(argv[1]) : 2000;

    auto pos_ptr = std::make_unique<Position>();
    auto ref_ptr = std::make_unique<Position>();
    Position& pos = *pos_ptr;
    Position& ref = *ref_ptr;

    std::vector<std::string> starts = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
      "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    };

    long long plies = 0, garbage = 0, garbageOk = 0;
    for (int g = 0; g < games; ++g)
    {
        const bool c960 = g % 2 == 1;
        std::string fen = c960 ? chess960_fen(int(rng() % 960)) : starts[g / 2 % starts.size()];

        std::deque<StateInfo> states(1);
        pos.set(fen, c960, &states.back());

        for (int ply = 0; ply < 300; ++ply)
        {
            MoveList<LEGAL> legal(pos);
            if (!legal.size())
                break;

            // pseudo_legal doit accepter tous les coups legaux
            for (const auto& m : legal)
                CHECK(pos.pseudo_legal(m) && pos.legal(m), "legal -> pseudo_legal");

            // et ne laisser passer aucun coup absurde
            for (int k = 0; k < 64; ++k)
            {
                Move m{static_cast<std::uint16_t>(rng())};
                if (!m.is_ok())
                    continue;
                ++garbage;
                if (pos.pseudo_legal(m) && pos.legal(m))
                {
                    ++garbageOk;
                    CHECK(legal.contains(m), "coup corrompu accepte");
                }
            }

            Move m = *(legal.begin() + rng() % legal.size());
            // Les prises, promotions et roques plus souvent : c'est la qu'est le risque
            for (int t = 0; t < 3; ++t)
            {
                Move c = *(legal.begin() + rng() % legal.size());
                if (pos.capture(c) || c.type_of() != NORMAL)
                {
                    m = c;
                    break;
                }
            }

            const std::string before    = pos.fen();
            const Key         keyBefore = pos.key();
            const bool        gc        = pos.gives_check(m);
            const Key         ka        = pos.key_after(m);
            (void) ka;

            states.emplace_back();
            pos.do_move(m, states.back(), gc);
            ++plies;

            CHECK(bool(pos.checkers()) == gc, "gives_check");

            std::deque<StateInfo> rs(1);
            ref.set(pos.fen(), c960, &rs.back());
            CHECK(ref.fen() == pos.fen(), "fen aller-retour");
            CHECK(ref.key() == pos.key(), "cle Zobrist");
            CHECK(ref.pawn_key() == pos.pawn_key(), "cle des pions");
            CHECK(ref.non_pawn_key(WHITE) == pos.non_pawn_key(WHITE), "cle non-pions blancs");
            CHECK(ref.non_pawn_key(BLACK) == pos.non_pawn_key(BLACK), "cle non-pions noirs");
            CHECK(ref.non_pawn_material(WHITE) == pos.non_pawn_material(WHITE), "materiel blanc");
            CHECK(ref.non_pawn_material(BLACK) == pos.non_pawn_material(BLACK), "materiel noir");
            CHECK(ref.checkers() == pos.checkers(), "checkers");
            CHECK(ref.blockers_for_king(WHITE) == pos.blockers_for_king(WHITE), "clouages blancs");
            CHECK(ref.blockers_for_king(BLACK) == pos.blockers_for_king(BLACK), "clouages noirs");
            CHECK(ref.rule50_count() == pos.rule50_count(), "regle des 50 coups");

            // undo puis redo : retour exact
            pos.undo_move(m);
            CHECK(pos.fen() == before && pos.key() == keyBefore, "undo_move");
            states.pop_back();
            states.emplace_back();
            pos.do_move(m, states.back(), gc);
        }
    }

    std::printf("%d parties, %lld coups, %lld coups aleatoires testes (%lld acceptes), %d echecs\n",
                games, plies, garbage, garbageOk, fails);
    return fails != 0;
}
