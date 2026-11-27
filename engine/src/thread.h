/*
  Onechess, a UCI chess engine derived from Stockfish.
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)
  Copyright (C) 2026 ovnice (modifications)

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

// Fils de recherche (reecrits pour Onechess le 15-11-2026). Plusieurs fils
// cherchent la meme position en parallele et se transmettent leurs resultats
// par la table de transposition ; ils peuvent aussi partager une partie de
// leurs historiques (option SharedHistories, adoptee le 13-11).

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "history.h"
#include "position.h"
#include "search.h"
#include "types.h"

namespace one {

// Historiques que les fils peuvent mettre en commun
struct SharedHist {
    PawnHistory         pawnHistory;
    CorrectionHistory   correctionHistory;
    CorrectionHistory   nonPawnCorrection[COLOR_NB];
    CorrectionHistory   minorCorrection;
    CorrectionHistory   majorCorrection;
    CorrectionHistory   menaceCorrection;   // CorrMenaceWeight (en test, 18-11)
    ContinuationHistory continuationHistory[2][2];
};
extern bool SharedHistOn;  // option SharedHistories

class Thread {
   public:
    explicit Thread(std::size_t numero, SharedHist* commun = nullptr);
    ~Thread();
    Thread(const Thread&) = delete;

    void clear();                      // oublie tout ce qui a ete appris
    void start_searching();            // lance une recherche (retour immediat)
    void wait_for_search_finished();   // attend la fin de la recherche en cours
    void iterative_deepening();        // la recherche elle-meme (search.cpp)

    std::size_t index() const { return numero; }
    bool        is_main() const { return numero == 0; }

    // --- racine et compteurs (lus et ecrits par la recherche)
    Position          rootPos;
    StateInfo         rootState;
    Search::RootMoves rootMoves;
    Depth             rootDepth = 0, completedDepth = 0;
    int               selDepth = 0, nmpMinPly = 0;
    int               callsCnt = 0;  // noeuds avant de relire l'horloge
    std::size_t       pvIdx    = 0;  // ligne MultiPV en cours

    std::atomic<std::uint64_t> nodes{0};
    std::atomic<std::uint64_t> tbHits{0};

    // --- historiques propres a ce fil
    ButterflyHistory              mainHistory;
    LowPlyHistory                 lowPlyHistory;
    CapturePieceToHistory         captureHistory;
    CounterMoveHistory            counterMoves;
    ContinuationCorrectionHistory continuationCorrectionHistory;

    // --- historiques propres ou communs (references vers « perso » ou vers le commun)
    SharedHist          perso;
    PawnHistory&        pawnHistory;
    CorrectionHistory&  correctionHistory;
    CorrectionHistory (&nonPawnCorrection)[COLOR_NB];
    CorrectionHistory&  minorCorrection;
    CorrectionHistory&  majorCorrection;
    CorrectionHistory&  menaceCorrection;
    ContinuationHistory (&continuationHistory)[2][2];  // [en echec][prise]

   private:
    void boucle();  // attend un ordre, cherche, recommence

    std::size_t             numero;
    std::mutex              verrou;
    std::condition_variable signal;
    bool                    travaille = true, quitter = false;
    std::thread             fil;
};

class ThreadPool {
   public:
    void set(std::size_t combien);   // recree les fils
    void clear();
    void start_thinking(Position& pos, StateListPtr& etats, const Search::LimitsType& limites);
    void wait_for_search_finished() const;
    void wait_for_helpers() const;   // tous sauf le principal (qui appelle)

    Thread*     main() const { return fils.front().get(); }
    Thread*     best_thread() const;  // celui dont on joue le coup
    std::size_t size() const { return fils.size(); }

    std::uint64_t nodes_searched() const;
    std::uint64_t tb_hits() const;

    std::atomic_bool stop{false};

   private:
    std::vector<std::unique_ptr<Thread>> fils;
    std::unique_ptr<SharedHist>          commun;
    StateListPtr                         historique;  // la partie reelle (repetitions)
};

extern ThreadPool Threads;

}  // namespace one
