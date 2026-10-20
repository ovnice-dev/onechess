#include "uci.h"
#include "prof.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "evaluate.h"
#include "misc.h"
#include "movegen.h"
#include "nnue.h"
#include "position.h"
#include "search.h"
#include "syzygy.h"
#include "thread.h"
#include "timeman.h"
#include "tt.h"
#include "perft.h"
#include "tune.h"

namespace one {

UCI::OptionsMap Options;

namespace {

const std::string StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Positions du banc : un echantillon large, ouvertures, milieux tactiques et
// finales. Le total de noeuds a profondeur fixe est la mesure de NPS du
// moteur, reproductible d'une compilation a l'autre.
const std::vector<std::string> BenchFEN = {
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
  "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
  "rq3rk1/ppp2ppp/1bnpb3/3N2B1/3NP3/7P/PPPQ1PP1/2KR3R w - - 7 14",
  "r1bq1r1k/1pp1n1pp/1p1p4/4p2Q/4Pp2/1BNP4/PPP2PPP/3R1RK1 w - - 2 14",
  "r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 2 15",
  "r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13",
  "r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1 16",
  "4r1k1/r1q2ppp/ppp2n2/4P3/5Rb1/1N1BQ3/PPP3PP/R5K1 w - - 1 17",
  "2rqkb1r/ppp2p2/2npb1p1/1N1Nn2p/2P1PP2/8/PP2B1PP/R1BQK2R b KQ - 0 11",
  "r1bq1r1k/b1p1npp1/p2p3p/1p6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 1 16",
  "3r1rk1/p4pp1/2p1b1p1/1p1n4/2P1PPQ1/P1BqB1P1/1PN2R1P/6K1 b - - 0 26",
  "8/8/8/8/8/6k1/6p1/6K1 w - - 0 1",
  "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
  "8/8/1P6/5pr1/8/4R3/7k/2K5 w - - 0 1",
  "6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
  "8/3p3B/5p2/5P2/p7/PP5b/k7/6K1 w - - 0 1",
  "5k2/7R/4P2p/5K2/p1r2P1p/8/8/8 b - - 0 1",
  "6k1/6p1/P6p/r1N5/5p2/7P/1b3PP1/4R1K1 w - - 0 1",
  "1r3k2/4q3/2Pp3b/3Bp3/2Q2p2/1p1P2P1/1P2KP2/3N4 w - - 0 1",
};

Search::LimitsType parse_go(Position& pos, std::istringstream& is) {
    Search::LimitsType limits;
    std::string        token;

    limits.startTime = now();

    while (is >> token)
        if (token == "searchmoves")
            while (is >> token)
                limits.searchmoves.push_back(UCI::to_move(pos, token));
        else if (token == "wtime")
            is >> limits.time[WHITE];
        else if (token == "btime")
            is >> limits.time[BLACK];
        else if (token == "winc")
            is >> limits.inc[WHITE];
        else if (token == "binc")
            is >> limits.inc[BLACK];
        else if (token == "movestogo")
            is >> limits.movestogo;
        else if (token == "depth")
            is >> limits.depth;
        else if (token == "nodes")
            is >> limits.nodes;
        else if (token == "movetime")
            is >> limits.movetime;
        else if (token == "mate")
            is >> limits.mate;
        else if (token == "perft")
            is >> limits.perft;
        else if (token == "infinite")
            limits.infinite = 1;

    return limits;
}

void go(Position& pos, std::istringstream& is, StateListPtr& states) {
    Search::LimitsType limits = parse_go(pos, is);

    Time.init(limits, pos.side_to_move(), pos.game_ply(), int(Options["Move Overhead"]));
    TT.new_search();

    // On lance et on rend la main : la boucle UCI doit rester capable de lire
    // "stop" pendant que la recherche tourne. C'est le thread principal de
    // recherche qui affichera le coup.
    Threads.start_thinking(pos, states, limits);
}

void position(Position& pos, std::istringstream& is, StateListPtr& states) {
    std::string token, fen;

    is >> token;

    if (token == "startpos")
    {
        fen = StartFEN;
        is >> token;  // consomme "moves"
    }
    else if (token == "fen")
        while (is >> token && token != "moves")
            fen += token + " ";
    else
        return;

    states = StateListPtr(new std::deque<StateInfo>(1));
    pos.set(fen, false, &states->back());

    Move m;
    while (is >> token && (m = UCI::to_move(pos, token)) != Move::none())
    {
        states->emplace_back();
        pos.do_move(m, states->back());
    }
}

// Banc de mesure : profondeur fixe sur un echantillon de positions. Deux
// compilations qui rendent le meme total de noeuds sont equivalentes du point
// de vue de la recherche -- c'est la forme de preuve la plus solide qu'un
// changement ne touche que la vitesse.
void bench(std::istringstream& is) {
    int depth = 13;
    is >> depth;

    Prof::reset();

    std::uint64_t total   = 0;
    const TimePoint start = now();

    for (const std::string& fen : BenchFEN)
    {
        Position     pos;
        StateListPtr states(new std::deque<StateInfo>(1));
        pos.set(fen, false, &states->back());

        Search::LimitsType limits;
        limits.startTime = now();
        limits.depth     = depth;
        limits.silent    = true;

        TT.clear(int(Options["Threads"]));
        Time.init(limits, pos.side_to_move(), pos.game_ply(), int(Options["Move Overhead"]));
        TT.new_search();

        Threads.start_thinking(pos, states, limits);
        Threads.wait_for_search_finished();

        total += Threads.nodes_searched();
    }

    const TimePoint elapsed = std::max<TimePoint>(now() - start, 1);

    std::cout << "\n==========================="
              << "\nProfondeur     : " << depth
              << "\nPositions      : " << BenchFEN.size()
              << "\nNoeuds totaux  : " << total
              << "\nTemps          : " << elapsed << " ms"
              << "\nNPS            : " << 1000 * total / elapsed << std::endl;
    Prof::report(total);
}

}  // namespace

// --------------------------------------------------------------- les options

void UCI::init(OptionsMap& o) {
    constexpr int MaxHashMB = 33554432;

    o["Hash"] << Option(64, 1, MaxHashMB, [](const Option& opt) {
        TT.resize(std::size_t(int(opt)), int(Options["Threads"]));
    });

    o["Threads"] << Option(1, 1, 512, [](const Option& opt) {
        Threads.set(std::size_t(int(opt)));
        TT.resize(std::size_t(int(Options["Hash"])), int(opt));
    });

    o["Clear Hash"] << Option(OnChange([](const Option&) { TT.clear(int(Options["Threads"])); }));

    o["Move Overhead"] << Option(10, 0, 5000);

    o["EvalFile"] << Option("onechess.nnue", [](const Option& opt) {
        const std::string path = opt;
        if (path.empty() || path == "<empty>")
            NNUE::unload();
        else if (!NNUE::load(path))
            std::cout << "info string reseau introuvable ou invalide : " << path
                      << " (evaluation PSQT conservee)" << std::endl;
        else
            std::cout << "info string " << NNUE::info() << std::endl;
    });

    o["Ponder"] << Option(false);

    o["UCI_Chess960"] << Option(false);

    // Tables de finale Syzygy. Chemin vide = desactivees : aucune ressource
    // externe ne doit etre obligatoire pour que le moteur tourne.
    o["SyzygyPath"] << Option("<empty>", [](const Option& opt) {
        const std::string path = opt;
        Syzygy::init(path);
        if (Syzygy::max_pieces())
            std::cout << "info string Syzygy : jusqu'a " << Syzygy::max_pieces()
                      << " pieces  " << Syzygy::path() << std::endl;
        else if (!path.empty() && path != "<empty>")
            std::cout << "info string Syzygy introuvable : " << path << std::endl;
    });

    // En dessous de cette profondeur, sonder coute plus cher que chercher :
    // chaque sonde est un acces disque, la recherche non.
    o["SyzygyProbeDepth"] << Option(1, 1, 100);

    // Permet de brider volontairement les tables a moins de pieces qu'elles
    // n'en contiennent -- utile pour mesurer leur apport.
    o["SyzygyProbeLimit"] << Option(7, 0, 7);

    // Faux : les gains "maudits" comptent comme des gains. A ne changer que
    // pour une analyse, jamais en partie.
    o["Syzygy50MoveRule"] << Option(true);

    // Toutes les constantes de recherche, exposees pour le SPRT
#define TUNABLE_OPTION(name, def, lo, hi)                          \
    o[#name] << Option(def, lo, hi, [](const Option& opt) {        \
        Tune::name = int(opt);                                     \
        Search::init();                                            \
    });
    TUNABLE_LIST(TUNABLE_OPTION)
#undef TUNABLE_OPTION
}

UCI::Option::Option(OnChange f) : type("button"), on_change(std::move(f)) {}

UCI::Option::Option(bool v, OnChange f) : type("check"), on_change(std::move(f)) {
    defaultValue = currentValue = (v ? "true" : "false");
}

UCI::Option::Option(const char* v, OnChange f) : type("string"), on_change(std::move(f)) {
    defaultValue = currentValue = v;
}

UCI::Option::Option(int v, int minv, int maxv, OnChange f) :
    type("spin"),
    min(minv),
    max(maxv),
    on_change(std::move(f)) {
    defaultValue = currentValue = std::to_string(v);
}

UCI::Option::operator int() const {
    assert(type == "check" || type == "spin");
    return type == "spin" ? std::stoi(currentValue) : currentValue == "true";
}

UCI::Option::operator std::string() const {
    assert(type == "string");
    return currentValue;
}

// Conserve l'ordre d'insertion, pour que "uci" affiche les options dans un
// ordre stable.
void UCI::Option::operator<<(const Option& o) {
    static std::size_t insertOrder = 0;
    *this                          = o;
    idx                            = insertOrder++;
}

UCI::Option& UCI::Option::operator=(const std::string& v) {
    assert(!type.empty());

    if ((type != "button" && type != "string" && v.empty())
        || (type == "check" && v != "true" && v != "false")
        || (type == "spin" && (std::stoi(v) < min || std::stoi(v) > max)))
        return *this;

    if (type != "button")
        currentValue = v;

    if (on_change)
        on_change(*this);

    return *this;
}

bool UCI::CaseInsensitiveLess::operator()(const std::string& s1, const std::string& s2) const {
    return std::lexicographical_compare(
      s1.begin(), s1.end(), s2.begin(), s2.end(),
      [](char c1, char c2) { return std::tolower(c1) < std::tolower(c2); });
}

std::ostream& operator<<(std::ostream& os, const UCI::OptionsMap& om) {
    for (std::size_t idx = 0; idx < om.size(); ++idx)
        for (const auto& it : om)
            if (it.second.idx == idx)
            {
                const UCI::Option& o = it.second;
                os << "\noption name " << it.first << " type " << o.type;

                if (o.type == "check" || o.type == "combo")
                    os << " default " << o.defaultValue;
                else if (o.type == "string")
                    os << " default " << o.defaultValue;
                else if (o.type == "spin")
                    os << " default " << o.defaultValue << " min " << o.min << " max " << o.max;

                break;
            }

    return os;
}

// ------------------------------------------------------------ conversions

std::string UCI::square(Square s) {
    return std::string{char('a' + file_of(s)), char('1' + rank_of(s))};
}

std::string UCI::move(Move m, bool chess960) {
    if (m == Move::none())
        return "(none)";
    if (m == Move::null())
        return "0000";

    Square from = m.from_sq();
    Square to   = m.to_sq();

    // Hors Chess960, le roque se note par la case reelle du roi
    if (m.type_of() == CASTLING && !chess960)
        to = make_square(to > from ? FILE_G : FILE_C, rank_of(from));

    std::string s = square(from) + square(to);

    if (m.type_of() == PROMOTION)
        s += " pnbrqk"[m.promotion_type()];

    return s;
}

Move UCI::to_move(const Position& pos, std::string& str) {
    if (str.length() == 5)
        str[4] = char(std::tolower(str[4]));

    for (const auto& m : MoveList<LEGAL>(pos))
        if (str == UCI::move(m, pos.is_chess960()))
            return m;

    return Move::none();
}

std::string UCI::value(Value v) {
    assert(-VALUE_INFINITE < v && v < VALUE_INFINITE);

    std::stringstream ss;

    if (std::abs(v) < VALUE_TB_WIN_IN_MAX_PLY)
        ss << "cp " << v;
    // Gain ou perte des tables de finales : pas un mat compte en coups. On
    // l'annonce en centipions, 20 000 moins la distance, comme Stockfish.
    else if (std::abs(v) < VALUE_MATE_IN_MAX_PLY)
        ss << "cp " << (v > 0 ? 20000 - (VALUE_TB - v) : -20000 + (VALUE_TB + v));
    else
        ss << "mate " << (v > 0 ? VALUE_MATE - v + 1 : -VALUE_MATE - v) / 2;

    return ss.str();
}

std::string UCI::pv(const Thread& th, Depth depth) {
    std::stringstream ss;

    const TimePoint     elapsed = Time.elapsed() + 1;
    const std::uint64_t nodes   = Threads.nodes_searched();
    const auto&         rm      = th.rootMoves[0];

    if (rm.score == -VALUE_INFINITE)
        return "";

    ss << "info depth " << depth << " seldepth " << rm.selDepth << " score " << UCI::value(rm.score)
       << " nodes " << nodes << " nps " << nodes * 1000 / elapsed << " hashfull " << TT.hashfull()
       << " time " << elapsed << " pv";

    for (Move m : rm.pv)
        ss << " " << UCI::move(m, th.rootPos.is_chess960());

    return ss.str();
}

// ------------------------------------------------------------- boucle UCI

void UCI::loop(int argc, char* argv[]) {
    Position     pos;
    StateListPtr states(new std::deque<StateInfo>(1));

    pos.set(StartFEN, false, &states->back());

    std::string cmd;
    for (int i = 1; i < argc; ++i)
        cmd += std::string(argv[i]) + " ";

    do
    {
        if (argc == 1 && !std::getline(std::cin, cmd))
            cmd = "fin";

        std::istringstream is(cmd);
        std::string        token;
        is >> std::skipws >> token;

        if (token == "quit" || token == "stop")
        {
            Threads.stop = true;
            if (token == "quit")
            {
                Threads.wait_for_search_finished();
                break;
            }
        }

        // Fin de l'entree standard : on laisse la recherche en cours aller au
        // bout, sinon un script en tube n'obtiendrait jamais son coup. Seule
        // une recherche infinie doit etre interrompue, elle ne finirait pas.
        else if (token == "fin")
        {
            if (Search::Limits.infinite)
                Threads.stop = true;
            Threads.wait_for_search_finished();
            break;
        }

        else if (token == "uci")
            std::cout << "id name " << engine_info() << "\nid author ovnice" << Options
                      << "\nuciok" << std::endl;

        else if (token == "setoption")
        {
            std::string name, value;

            is >> token;  // consomme "name"

            while (is >> token && token != "value")
                name += (name.empty() ? "" : " ") + token;

            while (is >> token)
                value += (value.empty() ? "" : " ") + token;

            if (Options.count(name))
                Options[name] = value;
            else
                std::cout << "option inconnue: " << name << std::endl;
        }

        else if (token == "ucinewgame")
        {
            Threads.clear();
            TT.clear(int(Options["Threads"]));
            Time.new_game();
        }

        else if (token == "isready")
            std::cout << "readyok" << std::endl;

        else if (token == "ponderhit")
            Search::Limits.infinite = 0;

        else if (token == "position")
            position(pos, is, states);

        else if (token == "go")
            go(pos, is, states);

        else if (token == "bench")
            bench(is);

        else if (token == "perft")
        {
            int d = 1;
            is >> d;
            perft_divide(pos, d);
        }

        else if (token == "perfttest")
            perft_suite();

        else if (token == "eval")
            std::cout << Eval::trace(pos) << std::endl;

        else if (token == "d")
        {
            std::cout << pos << "Interface  : historique " << pos.history_length()
                      << " demi-coups, repetition=" << pos.repetition() << std::endl;

            // Ce qui compte reellement : ce que le thread de recherche voit
            // depuis SA racine. C'est la seule vue qui alimente is_draw().
            const Position& rp = Threads.main()->rootPos;
            std::cout << "Recherche  : historique " << rp.history_length()
                      << " demi-coups, repetition=" << rp.repetition() << std::endl;
        }

        else if (!token.empty())
            std::cout << "commande inconnue: " << token << std::endl;

    } while (argc == 1);
}

}  // namespace one
