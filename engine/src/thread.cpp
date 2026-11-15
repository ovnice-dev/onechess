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

// Fils de recherche (reecrits pour Onechess le 15-11-2026).

#include "thread.h"

#include <algorithm>
#include <cassert>

#include "movegen.h"
#include "prof.h"
#include "search.h"
#include "syzygy.h"
#include "tt.h"
#include "uci.h"

namespace one {

ThreadPool Threads;
bool       SharedHistOn = true;  // SharedHist4 : +16,2 a 4 fils (13-11)

Thread::Thread(std::size_t n, SharedHist* c) :
    pawnHistory(c ? c->pawnHistory : perso.pawnHistory),
    correctionHistory(c ? c->correctionHistory : perso.correctionHistory),
    nonPawnCorrection(c ? c->nonPawnCorrection : perso.nonPawnCorrection),
    minorCorrection(c ? c->minorCorrection : perso.minorCorrection),
    majorCorrection(c ? c->majorCorrection : perso.majorCorrection),
    continuationHistory(c ? c->continuationHistory : perso.continuationHistory),
    numero(n),
    fil([this] { boucle(); }) {
    wait_for_search_finished();  // le fil est pret quand il s'est mis en attente
}

Thread::~Thread() {
    {
        std::lock_guard<std::mutex> g(verrou);
        quitter = travaille = true;
    }
    signal.notify_one();
    fil.join();
}

void Thread::boucle() {
    for (;;)
    {
        std::unique_lock<std::mutex> g(verrou);
        travaille = false;
        signal.notify_one();
        signal.wait(g, [this] { return travaille; });
        if (quitter)
            return;
        g.unlock();
        iterative_deepening();
        Prof::flush();  // seulement avec -DPROFILE
    }
}

void Thread::start_searching() {
    {
        std::lock_guard<std::mutex> g(verrou);
        travaille = true;
    }
    signal.notify_one();
}

void Thread::wait_for_search_finished() {
    std::unique_lock<std::mutex> g(verrou);
    signal.wait(g, [this] { return !travaille; });
}

// Tout reprend de zero. Les historiques de suite partent d'un leger a priori
// negatif (-71) : un coup jamais essaye passe apres un coup deja juste moyen.
void Thread::clear() {
    mainHistory.fill(0);
    captureHistory.fill(0);
    counterMoves.fill(Move::none());
    pawnHistory.fill(0);
    correctionHistory.fill(0);
    nonPawnCorrection[WHITE].fill(0);
    nonPawnCorrection[BLACK].fill(0);
    minorCorrection.fill(0);
    majorCorrection.fill(0);
    for (auto& parEchec : continuationHistory)
        for (auto& parPrise : parEchec)
            for (auto& parPiece : parPrise)
                for (auto& t : parPiece)
                    t->fill(-71);
    for (auto& parPiece : continuationCorrectionHistory)
        for (auto& t : parPiece)
            t->fill(0);
}

// ---------------------------------------------------------------- le groupe
void ThreadPool::set(std::size_t combien) {
    if (!fils.empty())
        main()->wait_for_search_finished();
    fils.clear();  // les destructeurs arretent les fils
    if (!combien)
        return;
    commun.reset(SharedHistOn && combien > 1 ? new SharedHist() : nullptr);
    for (std::size_t i = 0; i < combien; ++i)
        fils.push_back(std::make_unique<Thread>(i, commun.get()));
    clear();
}

void ThreadPool::clear() {
    for (auto& f : fils)
        f->clear();
}

void ThreadPool::start_thinking(Position& pos, StateListPtr& etats, const Search::LimitsType& limites) {
    main()->wait_for_search_finished();
    stop           = false;
    Search::Limits = limites;

    // Coups de la racine (filtres par « go searchmoves »)
    Search::RootMoves racine;
    for (const ExtMove& m : MoveList<LEGAL>(pos))
        if (limites.searchmoves.empty()
            || std::find(limites.searchmoves.begin(), limites.searchmoves.end(), m) != limites.searchmoves.end())
            racine.emplace_back(m);

    // Tables de finales : on ne garde que le coup qu'elles indiquent (pour
    // tous les fils), choisi en respectant la regle des 50 coups. Le score
    // annonce ne sert qu'a l'affichage.
    Search::RootTbScore = VALUE_NONE;
    const int pieces    = popcount(pos.pieces());
    if (Syzygy::max_pieces() && pieces <= std::min(Syzygy::max_pieces(), int(Options["SyzygyProbeLimit"])))
    {
        int        wdl = 0, dtz = 0;
        const Move m   = Syzygy::probe_root(pos, bool(Options["Syzygy50MoveRule"]), &wdl, &dtz);
        const auto it  = std::find(racine.begin(), racine.end(), m);
        if (m && it != racine.end())
        {
            const Search::RootMove garde = *it;
            racine.assign(1, garde);
            const Value d = Value(std::min(dtz, MAX_PLY - 1));
            // Fathom : 4 gain, 3 gain hors delai, 2 nulle, 1 perte hors delai, 0 perte
            Search::RootTbScore = wdl == 4 ? VALUE_TB - d : wdl == 0 ? -VALUE_TB + d : VALUE_DRAW;
        }
    }

    // L'historique de la partie reste vivant pour les repetitions
    if (etats)
        historique = std::move(etats);

    // Chaque fil part de sa copie de la position. set() efface l'etat (et sa
    // chaine vers les coups precedents) : on recopie donc ensuite le dernier
    // etat reel de la partie, ce qui rebranche l'historique.
    for (auto& f : fils)
    {
        f->nodes          = 0;
        f->rootDepth      = 0;
        f->completedDepth = 0;
        f->selDepth       = 0;
        f->nmpMinPly      = 0;
        f->callsCnt       = 0;
        f->rootMoves      = racine;
        f->rootPos.set(pos.fen(), pos.is_chess960(), &f->rootState);
        f->rootState          = historique->back();
        f->rootState.accIndex = 0;  // la racine reprend au debut du reservoir
        f->rootState.accGen   = 0;
    }
    for (auto& f : fils)
        f->start_searching();
}

void ThreadPool::wait_for_search_finished() const {
    for (const auto& f : fils)
        f->wait_for_search_finished();
}

void ThreadPool::wait_for_helpers() const {
    for (std::size_t i = 1; i < fils.size(); ++i)
        fils[i]->wait_for_search_finished();
}

std::uint64_t ThreadPool::nodes_searched() const {
    std::uint64_t n = 0;
    for (const auto& f : fils)
        n += f->nodes.load(std::memory_order_relaxed);
    return n;
}

std::uint64_t ThreadPool::tb_hits() const {
    std::uint64_t n = 0;
    for (const auto& f : fils)
        n += f->tbHits.load(std::memory_order_relaxed);
    return n;
}

// Le fil principal, sauf si un autre a fini une profondeur plus grande avec
// un meilleur score, ou annonce un mat (ou un gain prouve) meilleur.
Thread* ThreadPool::best_thread() const {
    Thread* choix = main();
    for (const auto& f : fils)
    {
        const Value s = f->rootMoves[0].score;
        if (s == -VALUE_INFINITE)  // n'a rien fini
            continue;
        const Value sChoix = choix->rootMoves[0].score;
        const bool  decisif = std::abs(s) >= VALUE_TB_WIN_IN_MAX_PLY;
        if (decisif ? s > sChoix : f->completedDepth > choix->completedDepth && s > sChoix)
            choix = f.get();
    }
    return choix;
}

}  // namespace one
