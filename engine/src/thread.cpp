#include "thread.h"
#include "prof.h"

#include <algorithm>
#include <cassert>

#include "movegen.h"
#include "search.h"
#include "tt.h"
#include "uci.h"

namespace one {

ThreadPool Threads;

Thread::Thread(std::size_t n) :
    idx(n),
    stdThread(&Thread::idle_loop, this) {
    wait_for_search_finished();
}

Thread::~Thread() {
    assert(!searching);

    exit = true;
    start_searching();
    stdThread.join();
}

// Remet a zero tout ce que le thread a appris. Le prior negatif sur
// l'historique de continuation fait qu'un coup jamais essaye est considere
// legerement mauvais plutot que neutre : cela evite qu'il passe devant un
// coup dont on sait deja qu'il marche moyennement.
void Thread::clear() {
    mainHistory.fill(0);
    captureHistory.fill(0);
    pawnHistory.fill(0);
    correctionHistory.fill(0);
    for (auto& h : nonPawnCorrection)
        h.fill(0);
    minorCorrection.fill(0);
    majorCorrection.fill(0);
    counterMoves.fill(Move::none());

    for (bool inCheck : {false, true})
        for (int capture = 0; capture <= 1; ++capture)
            for (auto& to : continuationHistory[inCheck][capture])
                for (auto& h : to)
                    h->fill(-71);

    for (auto& to : continuationCorrectionHistory)
        for (auto& h : to)
            h->fill(0);
}

void Thread::idle_loop() {
    while (true)
    {
        std::unique_lock<std::mutex> lk(mutex);
        searching = false;
        cv.notify_one();  // reveille wait_for_search_finished()
        cv.wait(lk, [&] { return searching; });

        if (exit)
            return;

        lk.unlock();

        iterative_deepening();
        Prof::flush();   // compilation -DPROFILE seulement
    }
}

void Thread::start_searching() {
    mutex.lock();
    searching = true;
    mutex.unlock();
    cv.notify_one();
}

void Thread::wait_for_search_finished() {
    std::unique_lock<std::mutex> lk(mutex);
    cv.wait(lk, [&] { return !searching; });
}

// ---------------------------------------------------------------- le parc

void ThreadPool::set(std::size_t requested) {
    if (!threads.empty())
    {
        main()->wait_for_search_finished();

        while (!threads.empty())
            delete threads.back(), threads.pop_back();
    }

    if (requested == 0)
        return;

    for (std::size_t i = 0; i < requested; ++i)
        threads.push_back(new Thread(i));

    clear();
}

void ThreadPool::clear() {
    for (Thread* th : threads)
        th->clear();
}

// Prepare la racine et lache tous les threads sur la meme position.
void ThreadPool::start_thinking(Position&                 pos,
                                StateListPtr&             states,
                                const Search::LimitsType& limits) {
    main()->wait_for_search_finished();

    stop            = false;
    Search::Limits  = limits;

    Search::RootMoves rootMoves;

    for (const auto& m : MoveList<LEGAL>(pos))
        if (limits.searchmoves.empty()
            || std::count(limits.searchmoves.begin(), limits.searchmoves.end(), m))
            rootMoves.emplace_back(m);

    // On garde la propriete de la liste d'etats : la detection de repetition
    // remonte dans l'historique reel de la partie, pas seulement dans l'arbre.
    if (states.get())
        setupStates = std::move(states);

    // Chaque thread recoit sa propre copie de la position.
    //
    // L'ORDRE compte : set() efface le StateInfo, donc la chaine 'previous'.
    // On recopie donc l'etat de la partie APRES, ce qui rebranche l'historique
    // reel sans deplacer le pointeur (st vise deja &th->rootState).
    //
    // Sans cela la recherche ne voit que les repetitions qu'elle cree
    // elle-meme : le moteur peut repeter trois fois une position gagnante
    // sans jamais s'apercevoir qu'il vient de concder la nulle.
    for (Thread* th : threads)
    {
        th->nodes          = 0;
        th->rootDepth      = 0;
        th->completedDepth = 0;
        th->selDepth       = 0;
        th->nmpMinPly      = 0;
        th->rootMoves      = rootMoves;
        th->rootPos.set(pos.fen(), pos.is_chess960(), &th->rootState);
        th->rootState = setupStates->back();

        // La racine repart au debut du reservoir : la recherche ne descend
        // jamais au-dela de MAX_PLY, l'index ne peut donc pas deborder.
        th->rootState.accIndex = 0;
        th->rootState.accGen   = 0;
    }

    start_searching();
}

void ThreadPool::start_searching() {
    for (Thread* th : threads)
        th->start_searching();
}

void ThreadPool::wait_for_search_finished() const {
    for (Thread* th : threads)
        th->wait_for_search_finished();
}

void ThreadPool::wait_for_helpers() const {
    for (std::size_t i = 1; i < threads.size(); ++i)
        threads[i]->wait_for_search_finished();
}

std::uint64_t ThreadPool::nodes_searched() const {
    std::uint64_t sum = 0;
    for (Thread* th : threads)
        sum += th->nodes.load(std::memory_order_relaxed);
    return sum;
}

// Choisit le thread dont le resultat sera joue. Un helper ne l'emporte que
// s'il a reellement termine une profondeur superieure : comparer sur la
// profondeur selective ferait gagner un thread dont la quiescence a fait une
// longue excursion sans rien prouver.
Thread* ThreadPool::best_thread() const {
    Thread* best = threads.front();

    for (Thread* th : threads)
    {
        const Value bestScore = best->rootMoves[0].score;
        const Value thScore   = th->rootMoves[0].score;

        if (thScore == -VALUE_INFINITE)
            continue;

        // Un mat annonce l'emporte sur toute profondeur
        if (std::abs(thScore) >= VALUE_TB_WIN_IN_MAX_PLY)
        {
            if (thScore > bestScore)
                best = th;
        }
        else if (th->completedDepth > best->completedDepth && thScore > bestScore)
            best = th;
    }

    return best;
}

}  // namespace one
