/*
  Onechess, a UCI chess engine.
  Copyright (C) 2026 ovnice

  Onechess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Onechess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Protocole UCI (reecrit pour Onechess le 15-11-2026).

#include "uci.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include "evaluate.h"
#include "misc.h"
#include "movegen.h"
#include "nnue.h"
#include "perft.h"
#include "position.h"
#include "prof.h"
#include "search.h"
#include "syzygy.h"
#include "thread.h"
#include "timeman.h"
#include "tt.h"
#include "tune.h"

namespace one {

UCI::OptionsMap Options;

namespace {

const std::string Depart = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Positions du banc (15-11-2026) : 13 ouvertures de notre recueil de tests
// (UHO), et 7 finales composees pour Onechess. Le total de noeuds a
// profondeur fixe est la « signature » du moteur : deux compilations qui la
// donnent jouent exactement pareil.
const char* const Banc[] = {
  "r1bq1rk1/1ppnppb1/3p1npp/p5B1/2PPP3/2NB1N1P/PP3PP1/R2QK2R w KQ - 0 1",
  "rn1qk2r/p1pp1ppp/4pn2/8/1bbP4/5NP1/P2NPPBP/R1BQK2R w KQkq - 0 1",
  "r2qk2r/pbpn1p1p/1p1p2pn/3Pp3/2P1P3/2NB4/PP2NPPP/R2QK2R w KQkq - 0 1",
  "rnbqkb1r/pp2pppp/1np5/4p3/2PP4/2N2N2/PP3PPP/R1BQKB1R w KQkq - 0 1",
  "r1bqk2r/pp1n1pb1/2pp1npp/4p3/P2PP3/2N1BN2/1PP1BPPP/R2QK2R w KQkq e6 0 1",
  "r1bqkb1r/2pn1ppp/pp2p3/3n4/P1Q5/5NP1/1P1PPPBP/RNB2RK1 w kq - 0 1",
  "r2q1rk1/pppb1ppp/2n1pn2/8/1bNP4/2N3P1/PP2PPBP/R1BQK2R w KQ - 0 1",
  "r1nqk2r/pp1bbppp/2n1p3/2ppP3/3P4/2P2N2/PP2BPPP/RNBQR1K1 w kq - 0 1",
  "r3kbnr/pbpq1ppp/1pn1p3/3pP2P/3P2Q1/2N1B3/PPP2PP1/R3KBNR w KQkq - 0 1",
  "r2qk2r/pbpnbpp1/1p1ppn1p/8/3P1B2/3BPN1P/PPPNQPP1/R3K2R w KQkq - 0 1",
  "r1bqkb1r/ppp2p1p/1n2p3/5p2/3P4/3B4/PPP1NPPP/R1BQK2R w KQkq - 0 1",
  "r2qkbnr/1p1b2pp/p1n1p3/3pPp2/2pP4/P1P2N2/1PB2PPP/RNBQK2R w KQkq - 0 1",
  "rnbqk2r/5ppp/p1pb1p2/1p6/2PP4/P3B3/1P3PPP/R2QKBNR w KQkq - 0 1",
  "8/8/4k3/3p4/3P4/4K3/8/8 w - - 0 1",
  "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
  "8/6k1/6p1/3r4/5R2/6K1/7P/8 w - - 0 1",
  "4k3/8/8/2b5/8/4N3/3P4/4K3 w - - 0 1",
  "8/8/2k5/8/1P6/2K5/8/7q w - - 0 1",
  "r5k1/5ppp/8/8/8/8/5PPP/1Q4K1 w - - 0 1",
  "8/1p3k2/p1p5/P1P5/1P6/5K2/8/8 w - - 0 1",
};

// Entier lu sans exception (compilation -fno-exceptions)
bool lire_entier(const std::string& t, int& sortie) {
    if (t.empty())
        return false;
    char*      fin = nullptr;
    const long v   = std::strtol(t.c_str(), &fin, 10);
    if (*fin || fin == t.c_str() || v < INT_MIN || v > INT_MAX)
        return false;
    sortie = int(v);
    return true;
}

// Le placement d'un FEN est-il plausible (8 x 8, un roi par camp, pas de
// pion au bord) ? Verifie avant Position::set, qui suppose les deux rois.
bool placement_valide(const std::string& fen) {
    std::istringstream in(fen);
    std::string        plan, camp;
    in >> plan >> camp;
    if (camp != "w" && camp != "b")
        return false;
    int rangee = 0, col = 0, rois[2] = {0, 0};
    for (char c : plan)
    {
        if (c == '/')
        {
            if (col != 8 || ++rangee > 7)
                return false;
            col = 0;
            continue;
        }
        if (c >= '1' && c <= '8')
            col += c - '0';
        else
        {
            const std::size_t i = std::string("PNBRQKpnbrqk").find(c);
            if (i == std::string::npos || ((c == 'P' || c == 'p') && (rangee == 0 || rangee == 7)))
                return false;
            if (c == 'K' || c == 'k')
                ++rois[c == 'k'];
            ++col;
        }
        if (col > 8)
            return false;
    }
    return rangee == 7 && col == 8 && rois[0] == 1 && rois[1] == 1;
}

// « position [startpos | fen ...] [moves ...] »
void cmd_position(Position& pos, std::istringstream& in, StateListPtr& etats) {
    std::string mot, fen;
    in >> mot;
    if (mot == "startpos")
    {
        fen = Depart;
        in >> mot;  // « moves »
    }
    else if (mot == "fen")
        while (in >> mot && mot != "moves")
            fen += mot + ' ';
    else
        return;

    const bool chess960 = bool(Options["UCI_Chess960"]);
    if (!placement_valide(fen))
    {
        std::cout << "info string FEN invalide, position initiale : " << fen << std::endl;
        fen = Depart;
        in.setstate(std::ios::eofbit);  // les coups ne s'appliquent plus
    }
    etats = StateListPtr(new std::deque<StateInfo>(1));
    pos.set(fen, chess960, &etats->back());

    // le camp qui n'a pas le trait ne peut pas etre en echec
    if (pos.attackers_to(pos.square<KING>(~pos.side_to_move())) & pos.pieces(pos.side_to_move()))
    {
        std::cout << "info string FEN invalide (roi adverse en echec), position initiale" << std::endl;
        etats = StateListPtr(new std::deque<StateInfo>(1));
        pos.set(Depart, chess960, &etats->back());
        return;
    }
    for (Move m; in >> mot && (m = UCI::to_move(pos, mot));)
    {
        etats->emplace_back();
        pos.do_move(m, etats->back());
    }
}

// « go ... » : lit les contraintes et lance la recherche (sans attendre ;
// c'est le fil principal qui annoncera le coup).
void cmd_go(Position& pos, std::istringstream& in, StateListPtr& etats) {
    Search::LimitsType lim;
    lim.startTime = now();
    for (std::string mot; in >> mot;)
    {
        if (mot == "searchmoves")
            while (in >> mot)
                lim.searchmoves.push_back(UCI::to_move(pos, mot));
        else if (mot == "wtime")      in >> lim.time[WHITE];
        else if (mot == "btime")      in >> lim.time[BLACK];
        else if (mot == "winc")       in >> lim.inc[WHITE];
        else if (mot == "binc")       in >> lim.inc[BLACK];
        else if (mot == "movestogo")  in >> lim.movestogo;
        else if (mot == "depth")      in >> lim.depth;
        else if (mot == "nodes")      in >> lim.nodes;
        else if (mot == "movetime")   in >> lim.movetime;
        else if (mot == "mate")       in >> lim.mate;
        else if (mot == "perft")      in >> lim.perft;
        else if (mot == "infinite")   lim.infinite = 1;
        else if (mot == "ponder")     lim.ponderMode = true;
    }
    // « go perft N » (commande courante des testeurs) : comptage, pas de recherche
    // (avant le 19-11, la valeur etait lue puis ignoree : recherche sans fin)
    if (lim.perft)
    {
        perft_divide(pos, Depth(lim.perft));
        return;
    }
    Time.init(lim, pos.side_to_move(), pos.game_ply(), int(Options["Move Overhead"]), popcount(pos.pieces()));
    TT.new_search();
    Search::Ponder          = lim.ponderMode;
    Search::StopOnPonderhit = false;
    Search::Active          = !lim.silent;
    Threads.start_thinking(pos, etats, lim);
}

// « bench [profondeur] » : chaque position du banc a profondeur fixe, table
// videe entre deux ; seul le temps de recherche est compte.
void cmd_bench(std::istringstream& in) {
    int prof = 13;
    in >> prof;
    Prof::reset();
    Search::Ponder          = false;  // un « stop » apres « go ponder » le laissait a vrai : bench bloque
    Search::StopOnPonderhit = false;

    std::uint64_t                       noeuds = 0;
    std::chrono::steady_clock::duration duree{};
    for (const char* fen : Banc)
    {
        Position     pos;
        StateListPtr etats(new std::deque<StateInfo>(1));
        pos.set(fen, false, &etats->back());

        Search::LimitsType lim;
        lim.startTime = now();
        lim.depth     = prof;
        lim.silent    = true;
        TT.clear(int(Options["Threads"]));
        Time.init(lim, pos.side_to_move(), pos.game_ply(), int(Options["Move Overhead"]), popcount(pos.pieces()));
        TT.new_search();

        const auto t = std::chrono::steady_clock::now();
        Threads.start_thinking(pos, etats, lim);
        Threads.wait_for_search_finished();
        duree += std::chrono::steady_clock::now() - t;
        noeuds += Threads.nodes_searched();
    }
    const TimePoint ms = std::max<TimePoint>(1, std::chrono::duration_cast<std::chrono::milliseconds>(duree).count());
    std::cout << "\n===========================\nProfondeur     : " << prof
              << "\nPositions      : " << std::size(Banc) << "\nNoeuds totaux  : " << noeuds
              << "\nTemps          : " << ms << " ms\nNPS            : " << 1000 * noeuds / ms << std::endl;
    Prof::report(noeuds);
}

// « setoption name <nom> value <valeur> » (les noms peuvent contenir des espaces)
void cmd_setoption(std::istringstream& in) {
    std::string mot, nom, valeur;
    in >> mot;  // « name »
    while (in >> mot && mot != "value")
        nom += (nom.empty() ? "" : " ") + mot;
    while (in >> mot)
        valeur += (valeur.empty() ? "" : " ") + mot;
    auto it = Options.find(nom);
    if (it != Options.end())
        it->second = valeur;
    else
        std::cout << "option inconnue: " << nom << std::endl;
}

// Chien de garde (7-11) : une recherche au temps qui depasse son plafond
// d'une seconde est arretee (elle joue alors son meilleur coup). Le bot avait
// perdu des parties au temps apres des « ponderhit ».
void lancer_chien_de_garde() {
    std::thread([] {
        for (;;)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            const bool auTemps = Search::Active && !Search::Ponder && !Search::Limits.infinite
                              && Search::Limits.use_time_management() && !Threads.stop;
            if (auTemps && Time.elapsed() > Time.maximum() + 1000)
            {
                UCI::sortie("info string chien de garde : arret force a " + std::to_string(Time.elapsed())
                            + " ms (maximum " + std::to_string(Time.maximum()) + ")");
                Threads.stop = true;
            }
        }
    }).detach();
}

}  // namespace

// ================================================================ options
void UCI::init(OptionsMap& o) {
    o["Hash"] << Option(64, 1, 33554432, [](const Option& v) {
        TT.resize(std::size_t(int(v)), int(Options["Threads"]));
    });
    o["Threads"] << Option(1, 1, 512, [](const Option& v) {
        Threads.set(std::size_t(int(v)));
        TT.resize(std::size_t(int(Options["Hash"])), int(v));
    });
    // Historiques partages entre fils (adopte le 13-11) ; cache
    o["SharedHistories"] << Option(true, [](const Option& v) {
        SharedHistOn = bool(v);
        Threads.set(std::size_t(int(Options["Threads"])));
    });
    o["SharedHistories"].hidden = true;
    o["Clear Hash"] << Option(OnChange([](const Option&) { TT.clear(int(Options["Threads"])); }));
    o["Move Overhead"] << Option(10, 0, 5000);
    o["EvalFile"] << Option("onechess.nnue", [](const Option& v) {
        const std::string chemin = v;
        if (chemin.empty() || chemin == "<empty>")
            NNUE::unload();
        else if (NNUE::load(chemin))
            std::cout << "info string " << NNUE::info() << std::endl;
        else
            std::cout << "info string reseau introuvable ou invalide : " << chemin << " (reseau precedent conserve)"
                      << std::endl;
    });
    o["Ponder"] << Option(false);
    o["UCI_Chess960"] << Option(false);
    // Tables Syzygy : rien d'obligatoire, chemin vide = sans tables
    o["SyzygyPath"] << Option("<empty>", [](const Option& v) {
        const std::string chemin = v;
        Syzygy::init(chemin);
        if (Syzygy::max_pieces())
            std::cout << "info string Syzygy : jusqu'a " << Syzygy::max_pieces() << " pieces  " << Syzygy::path()
                      << std::endl;
        else if (!chemin.empty() && chemin != "<empty>")
            std::cout << "info string Syzygy introuvable : " << chemin << std::endl;
    });
    o["SyzygyProbeDepth"] << Option(1, 1, 100);  // sonder plus bas coute plus que chercher
    o["SyzygyProbeLimit"] << Option(7, 0, 7);    // brider les tables (mesures)
    o["Syzygy50MoveRule"] << Option(true);       // faux : analyse seulement
    o["MultiPV"] << Option(1, 1, 256);
    o["UCI_ShowWDL"] << Option(false);
    o["Debug Log File"] << Option("", [](const Option& v) { start_logger(std::string(v)); });

    // Constantes de recherche : reglables (tests, SPSA), cachees de la liste
#ifndef ONE_FIGE
#define ONE_OPTION_REGLAGE(nom, defaut, mini, maxi)                \
    o[#nom] << Option(defaut, mini, maxi, [](const Option& v) {    \
        Tune::nom = int(v);                                        \
        Search::init();                                            \
    });                                                            \
    o[#nom].hidden = true;
    TUNABLE_LIST(ONE_OPTION_REGLAGE)
#undef ONE_OPTION_REGLAGE
#endif
}

UCI::Option::Option(OnChange f) : genre(BOUTON), rappel(std::move(f)) {}
UCI::Option::Option(bool v, OnChange f) : genre(CASE), defaut(v ? "true" : "false"), rappel(std::move(f)) {
    valeur = defaut;
}
UCI::Option::Option(const char* v, OnChange f) : genre(TEXTE), defaut(v), rappel(std::move(f)) { valeur = defaut; }
UCI::Option::Option(int v, int lo, int hi, OnChange f) :
    genre(ENTIER), defaut(std::to_string(v)), mini(lo), maxi(hi), rappel(std::move(f)) {
    valeur = defaut;
}

UCI::Option::operator int() const {
    if (genre == CASE)
        return valeur == "true";
    int v = 0;
    return lire_entier(valeur, v) ? v : 0;
}

UCI::Option::operator std::string() const { return valeur; }

void UCI::Option::operator<<(const Option& o) {
    static std::size_t compteur = 0;
    *this                       = o;
    rang                        = compteur++;
}

UCI::Option& UCI::Option::operator=(const std::string& v) {
    int x = 0;
    switch (genre)
    {
    case CASE :
        if (v != "true" && v != "false")
            return *this;
        break;
    case ENTIER :
        if (!lire_entier(v, x) || x < mini || x > maxi)
            return *this;
        break;
    default : break;
    }
    if (genre != BOUTON)
        valeur = v;
    if (rappel)
        rappel(*this);
    return *this;
}

void UCI::Option::decrire(std::ostream& os, const std::string& nom) const {
    static const char* const noms[] = {"button", "check", "string", "spin"};
    os << "\noption name " << nom << " type " << noms[genre];
    if (genre != BOUTON)
        os << " default " << defaut;
    if (genre == ENTIER)
        os << " min " << mini << " max " << maxi;
}

bool UCI::SansCasse::operator()(const std::string& a, const std::string& b) const {
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i)
    {
        const int x = std::tolower((unsigned char) a[i]), y = std::tolower((unsigned char) b[i]);
        if (x != y)
            return x < y;
    }
    return a.size() < b.size();
}

// Les options visibles, dans l'ordre ou elles ont ete declarees
std::ostream& operator<<(std::ostream& os, const UCI::OptionsMap& options) {
    std::vector<std::pair<std::size_t, const std::string*>> ordre;
    for (const auto& [nom, opt] : options)
        if (!opt.hidden)
            ordre.emplace_back(opt.ordre(), &nom);
    std::sort(ordre.begin(), ordre.end());
    for (const auto& [rang, nom] : ordre)
        options.at(*nom).decrire(os, *nom);
    return os;
}

// ================================================================ ecritures
std::string UCI::square(Square s) { return {char('a' + file_of(s)), char('1' + rank_of(s))}; }

void UCI::sortie(const std::string& ligne) {
    static std::mutex verrou;
    const std::lock_guard<std::mutex> garde(verrou);
    std::cout << ligne << std::endl;
}

std::string UCI::move(Move m, bool chess960) {
    if (!m)
        return "(none)";
    if (m == Move::null())
        return "0000";
    const Square de = m.from_sq();
    Square       a  = m.to_sq();
    if (m.type_of() == CASTLING && !chess960)  // en partie normale, case d'arrivee du roi
        a = make_square(a > de ? FILE_G : FILE_C, rank_of(de));
    std::string t = square(de) + square(a);
    if (m.type_of() == PROMOTION)
        t += "nbrq"[m.promotion_type() - KNIGHT];
    return t;
}

Move UCI::to_move(const Position& pos, std::string& texte) {
    if (texte.size() == 5)  // la piece de promotion en minuscule
        texte[4] = char(std::tolower((unsigned char) texte[4]));
    for (const ExtMove& m : MoveList<LEGAL>(pos))
        if (texte == move(m, pos.is_chess960()))
            return m;
    return Move::none();
}

std::string UCI::value(Value v) {
    assert(std::abs(v) < VALUE_INFINITE);
    if (std::abs(v) < VALUE_TB_WIN_IN_MAX_PLY)
        return "cp " + std::to_string(v);
    if (std::abs(v) < VALUE_MATE_IN_MAX_PLY)  // gain ou perte des tables : 20 000 moins la distance
        return "cp " + std::to_string(v > 0 ? 20000 - (VALUE_TB - v) : -20000 + (VALUE_TB + v));
    const int coups = v > 0 ? (VALUE_MATE - v + 1) / 2 : -(VALUE_MATE + v) / 2;  // en coups, pas en demi-coups
    return "mate " + std::to_string(coups);
}

// Probabilites gain / nulle / perte : logistique ajustee sur nos positions
// etiquetees (JOURNAL 4-11), P(gain) = 1 / (1 + exp(-(v - 152) / 101)).
std::string UCI::wdl(Value v) {
    auto      proba = [](double x) { return 1.0 / (1.0 + std::exp(-(x - 152.0) / 101.0)); };
    const int g     = int(std::lround(1000 * proba(v))), p = int(std::lround(1000 * proba(-v)));
    return " wdl " + std::to_string(g) + ' ' + std::to_string(std::max(0, 1000 - g - p)) + ' ' + std::to_string(p);
}

std::string UCI::pv(const Thread& th, Depth depth) {
    std::ostringstream  out;
    const TimePoint     ms     = Time.elapsed() + 1;
    const std::uint64_t noeuds = Threads.nodes_searched();
    const std::size_t   lignes = std::min(std::size_t(int(Options["MultiPV"])), th.rootMoves.size());
    bool                premiere = true;

    for (std::size_t i = 0; i < lignes; ++i)
    {
        const Search::RootMove& rm   = th.rootMoves[i];
        const bool              neuf = rm.score != -VALUE_INFINITE;  // cherchee a cette profondeur ?
        Value                   v    = neuf ? rm.score : rm.previousScore;
        // racine dans les tables : leur verdict, sauf si un mat est deja trouve
        if (Search::RootTbScore != VALUE_NONE && v != -VALUE_INFINITE && std::abs(v) < VALUE_MATE_IN_MAX_PLY)
            v = Search::RootTbScore;
        if (v == -VALUE_INFINITE)
            continue;
        if (!premiere)
            out << '\n';
        premiere = false;
        out << "info depth " << (neuf ? depth : std::max(1, depth - 1)) << " seldepth " << rm.selDepth;
        if (lignes > 1)
            out << " multipv " << i + 1;
        out << " score " << value(v);
        if (bool(Options["UCI_ShowWDL"]) && std::abs(v) < VALUE_TB_WIN_IN_MAX_PLY)
            out << wdl(v);
        out << " nodes " << noeuds << " nps " << noeuds * 1000 / ms << " hashfull " << TT.hashfull() << " tbhits "
            << Threads.tb_hits() << " time " << ms << " pv";
        for (Move m : rm.pv)
            out << ' ' << move(m, th.rootPos.is_chess960());
    }
    return out.str();
}

// ================================================================ boucle
void UCI::loop(int argc, char* argv[]) {
    lancer_chien_de_garde();

    Position     pos;
    StateListPtr etats(new std::deque<StateInfo>(1));
    pos.set(Depart, false, &etats->back());

    // Avec des arguments : une seule commande, puis on sort
    std::string ligne;
    for (int i = 1; i < argc; ++i)
        ligne += std::string(argv[i]) + ' ';

    for (bool continuer = true; continuer;)
    {
        if (argc > 1)
            continuer = false;
        else if (!std::getline(std::cin, ligne))
            ligne = "fin";  // fin de l'entree

        std::istringstream in(ligne);
        std::string        cmd;
        in >> cmd;

        // Commandes qui touchent a ce que la recherche utilise : on attend
        // qu'elle ait fini. Une recherche « ponder » ou infinie ne finit pas
        // seule : on l'arrete d'abord (sinon la boucle se figerait).
        const bool attendre = cmd == "setoption" || cmd == "ucinewgame" || cmd == "bench" || cmd == "perft"
                           || cmd == "perfttest" || cmd == "eval" || cmd == "go";
        if (attendre && (Search::Ponder || Search::Limits.infinite))
            Threads.stop = true;
        // « go » aussi (18-11) : regler Ponder, le temps ou la table pendant
        // qu'une ancienne recherche tourne encore la derangeait (une recherche
        // normale qui voyait Ponder passer a vrai ne s'arretait plus, et le
        // moteur se figeait en attendant sa fin).
        if (attendre)
            Threads.main()->wait_for_search_finished();

        if (cmd == "quit")
        {
            Threads.stop = true;
            Threads.wait_for_search_finished();
            break;
        }
        if (cmd == "stop")
            Threads.stop = true;
        else if (cmd == "fin")
        {
            // un script en tube doit quand meme obtenir son coup : on laisse
            // finir la recherche, sauf si elle est infinie
            if (Search::Limits.infinite || Search::Ponder)
                Threads.stop = true;
            Threads.wait_for_search_finished();
            break;
        }
        else if (cmd == "uci")
            std::cout << "id name " << engine_info() << "\nid author ovnice" << Options << "\nuciok" << std::endl;
        else if (cmd == "isready")
            UCI::sortie("readyok");
        else if (cmd == "setoption")
            cmd_setoption(in);
        else if (cmd == "ucinewgame")
        {
            Threads.clear();
            TT.clear(int(Options["Threads"]));
            Time.new_game();
        }
        else if (cmd == "ponderhit")
        {
            // le coup prevu a ete joue : on passe en temps normal (et on joue
            // tout de suite si la recherche avait deja decide de s'arreter)
            Search::Ponder = false;
            if (Search::StopOnPonderhit)
                Threads.stop = true;
            UCI::sortie("info string ponderhit : " + std::to_string(Time.elapsed()) + " ms ecoulees, optimum "
                        + std::to_string(Time.optimum()) + ", maximum " + std::to_string(Time.maximum()));
        }
        else if (cmd == "position")
            cmd_position(pos, in, etats);
        else if (cmd == "go")
            cmd_go(pos, in, etats);
        else if (cmd == "bench")
            cmd_bench(in);
        else if (cmd == "perft")
        {
            int d = 1;
            in >> d;
            perft_divide(pos, d);
        }
        else if (cmd == "perfttest")
            perft_suite();
        else if (cmd == "eval")
            std::cout << Eval::trace(pos) << std::endl;
        else if (cmd == "d")
        {
            std::cout << pos << "Interface  : historique " << pos.history_length() << " demi-coups, repetition="
                      << pos.repetition() << std::endl;
            const Position& r = Threads.main()->rootPos;  // ce que voit la recherche
            std::cout << "Recherche  : historique " << r.history_length() << " demi-coups, repetition="
                      << r.repetition() << std::endl;
        }
        else if (!cmd.empty())
            std::cout << "commande inconnue: " << cmd << std::endl;
    }
}

}  // namespace one
