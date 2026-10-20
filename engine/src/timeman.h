#pragma once

#include "misc.h"
#include "search.h"
#include "types.h"

namespace one {

// Deux budgets : une cible (optimum) qu'une iteration en cours a le droit de
// depasser, et un plafond (maximum) qui, lui, est absolu.
class TimeManagement {
   public:
    void init(Search::LimitsType& limits, Color us, int ply, int moveOverhead);

    TimePoint optimum() const { return optimumTime; }
    TimePoint maximum() const { return maximumTime; }
    TimePoint elapsed() const { return now() - startTime; }

    // Nouvelle partie : le facteur de partie sera recalcule au premier coup.
    void new_game() { originalTimeAdjust = -1.0; }

   private:
    TimePoint startTime   = 0;
    TimePoint optimumTime = 0;
    TimePoint maximumTime = 0;
    double    originalTimeAdjust = -1.0;  // facteur de partie, mode TmSfAlloc
};

extern TimeManagement Time;

}  // namespace one
