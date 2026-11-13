#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "history.h"
#include "position.h"
#include "search.h"
#include "types.h"

namespace one {

// Un thread de recherche et tout son etat prive. Seule la table de
// transposition est partagee entre threads : c'est elle qui fait tout le
// travail de synchronisation du Lazy SMP. Les historiques restent locaux,
// et leur divergence est justement ce qui diversifie les recherches.
// Historiques que les fils peuvent partager (option cachee SharedHistories,
// 13-11, d'apres Stockfish : corrections d'evaluation, historique des pions,
// historique de continuation). Partages, les fils apprennent les uns des
// autres ; a un seul fil, rien ne change.
struct SharedHist {
    PawnHistory         pawnHistory;
    CorrectionHistory   correctionHistory;
    CorrectionHistory   nonPawnCorrection[COLOR_NB];
    CorrectionHistory   minorCorrection;
    CorrectionHistory   majorCorrection;
    ContinuationHistory continuationHistory[2][2];
};
extern bool SharedHistOn;   // option SharedHistories
extern bool SmpVote;        // option SmpVote : choix du coup final par vote des fils

class Thread {
   public:
    Thread(std::size_t n, SharedHist* shared = nullptr);
    ~Thread();

    void clear();
    void idle_loop();
    void start_searching();
    void wait_for_search_finished();

    // Approfondissement iteratif complet pour ce thread
    void iterative_deepening();

    std::size_t index() const { return idx; }
    bool        is_main() const { return idx == 0; }

    // ------------------------------------------------------------- racine
    Position          rootPos;
    StateInfo         rootState;
    Search::RootMoves rootMoves;
    Depth             rootDepth      = 0;
    Depth             completedDepth = 0;
    Value             rootDelta      = 0;
    int               selDepth       = 0;
    int               nmpMinPly      = 0;
    int               callsCnt       = 0;   // compte a rebours avant la prochaine lecture de l'horloge
    Color             nmpColor       = WHITE;

    std::atomic<std::uint64_t> nodes{0};
    std::atomic<std::uint64_t> tbHits{0};   // sondes Syzygy reussies
    std::size_t                pvIdx = 0;   // ligne MultiPV en cours de recherche

    // ---------------------------------------------------------- historiques
    ButterflyHistory      mainHistory;
    LowPlyHistory         lowPlyHistory;  // remis a zero a chaque recherche
    CapturePieceToHistory captureHistory;
    CounterMoveHistory    counterMoves;

    // Propres a ce fil, ou partages (SharedHist) : references vers l'un ou l'autre
    SharedHist          own;
    PawnHistory&        pawnHistory;
    CorrectionHistory&  correctionHistory;
    CorrectionHistory (&nonPawnCorrection)[COLOR_NB];  // [couleur des pieces][trait][cle]
    CorrectionHistory&  minorCorrection;  // [trait][cle cavaliers, fous et rois]
    CorrectionHistory&  majorCorrection;  // [trait][cle tours, dames et rois]

    // [le coup precedent etait-il une capture][donnait-il echec]
    ContinuationHistory (&continuationHistory)[2][2];

    // [piece][case] du coup d'avant -> [piece][case] du coup precedent
    ContinuationCorrectionHistory continuationCorrectionHistory;

   private:
    std::mutex              mutex;
    std::condition_variable cv;
    std::size_t             idx;
    bool                    exit = false, searching = true;
    std::thread             stdThread;
};

// Le parc de threads. start_thinking() prepare la racine, puis lache tout le
// monde sur la meme position.
class ThreadPool {
   public:
    void set(std::size_t requested);
    void clear();
    void start_thinking(Position& pos, StateListPtr& states, const Search::LimitsType& limits);
    void start_searching();
    void wait_for_search_finished() const;

    // Attend tous les threads SAUF le principal. Celui-ci ne peut pas
    // s'attendre lui-meme : c'est lui qui appelle, depuis sa propre recherche.
    void wait_for_helpers() const;

    Thread* main() const { return threads.front(); }
    Thread* best_thread() const;

    std::uint64_t nodes_searched() const;
    std::uint64_t tb_hits() const;

    std::size_t size() const { return threads.size(); }

    auto begin() const { return threads.begin(); }
    auto end() const { return threads.end(); }

    std::atomic_bool stop{false};

   private:
    StateListPtr         setupStates;
    std::vector<Thread*> threads;
    SharedHist*          shared = nullptr;
};

extern ThreadPool Threads;

}  // namespace one
