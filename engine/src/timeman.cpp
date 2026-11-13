#include "timeman.h"
#include "tune.h"

#include <algorithm>
#include <cmath>

namespace one {

using namespace Tune;

TimeManagement Time;

// Repartit le temps restant sur les coups a venir. La part allouee croit avec
// le numero de coup jusqu'a un plateau : les coups 15 a 40 sont ceux qui
// decident la partie, et une allocation plate les sous-alimente.
void TimeManagement::init(Search::LimitsType& limits, Color us, int ply, int moveOverhead, int pieces) {

    startTime = limits.startTime;

    // Pas de gestion du temps : profondeur fixe, nombre de noeuds fixe, ou
    // temps impose par coup.
    if (limits.movetime)
    {
        optimumTime = maximumTime = std::max<TimePoint>(limits.movetime - moveOverhead, 1);
        return;
    }

    if (!limits.use_time_management())
    {
        optimumTime = maximumTime = 0;
        return;
    }

    // Au-dela de 50 coups a jouer, la difference d'allocation est negligeable
    int mtg = limits.movestogo ? std::min(limits.movestogo, 50) : 50;

    // Temps reellement utilisable, une fois retiree la latence d'echange avec
    // l'interface sur chaque coup restant.
    TimePoint timeLeft = std::max<TimePoint>(
      1, limits.time[us] + limits.inc[us] * (mtg - 1) - moveOverhead * (2 + mtg));

    double optScale, maxScale;

    if (limits.movestogo == 0)
    {
        // Cadence a mort subite : la part monte en racine avec le numero de
        // coup, et reste bornee par une fraction du temps reellement restant.
        optScale = std::min(0.0084 + std::pow(ply + 3.0, 0.5) * 0.0042,
                            0.20 * double(limits.time[us]) / double(timeLeft));
        maxScale = std::min(7.0, 4.0 + ply / 12.0);
    }
    else
    {
        // Nombre de coups impose avant le prochain controle
        optScale = std::min((0.88 + ply / 116.4) / mtg, 0.88 * double(limits.time[us]) / double(timeLeft));
        maxScale = std::min(6.3, 1.5 + 0.11 * mtg);
    }

    // Retard a la pendule : quand on a moins de temps que l'adversaire, on
    // depense moins par coup. Inerte tant que TmClockPct vaut 0.
    if (TmClockPct && limits.movestogo != 1)
    {
        const double avantage = double(limits.time[us] - limits.time[~us])
                              / (1.0 + double(limits.time[us]) + double(limits.time[~us]));
        optScale *= 1.0 + TmClockPct / 1000.0 * std::min(avantage, 0.0);
    }

    (void) pieces;   // TmEarlySave, rejete

    optimumTime = TimePoint(optScale * double(timeLeft));
    maximumTime = TimePoint(std::min(0.84 * double(limits.time[us]) - moveOverhead,
                                     maxScale * double(optimumTime)));

    optimumTime = std::max<TimePoint>(optimumTime, 1);
    maximumTime = std::max<TimePoint>(maximumTime, 1);
}

}  // namespace one
