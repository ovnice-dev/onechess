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
void TimeManagement::init(Search::LimitsType& limits, Color us, int ply, int moveOverhead) {

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

    // Sous la seconde, en mort subite : on suppose beaucoup moins de coups a
    // jouer, ce qui empeche de s'endetter en fin de partie. Stockfish.
    if (TmSfAlloc && limits.movestogo == 0 && limits.time[us] < 1000)
        mtg = std::max(1, int(limits.time[us] * 0.05));

    // Temps reellement utilisable, une fois retiree la latence d'echange avec
    // l'interface sur chaque coup restant.
    TimePoint timeLeft = std::max<TimePoint>(
      1, limits.time[us] + limits.inc[us] * (mtg - 1) - moveOverhead * (2 + mtg));

    double optScale, maxScale;

    if (limits.movestogo == 0)
    {
        // Cadence a mort subite : la part monte en racine avec le numero de
        // coup, et reste bornee par une fraction du temps reellement restant.
        if (TmSfAlloc)
        {
            // Repartition de Stockfish : un facteur fixe pour la partie, tire
            // du temps disponible au premier coup, et des constantes qui
            // dependent du logarithme du temps restant. A 10+0,1, au 20e coup,
            // elle depense 13 % de moins sur le coup ordinaire et s'autorise
            // 21 % de plus sur le coup critique que notre ancienne formule.
            if (originalTimeAdjust < 0)
                originalTimeAdjust = 0.3272 * std::log10(double(timeLeft)) - 0.4141;
            const double logTime = std::log10(std::max(1.0, double(limits.time[us])) / 1000.0);
            const double optC    = std::min(0.0029869 + 0.00033554 * logTime, 0.004905);
            const double maxC    = std::max(3.3744 + 3.0608 * logTime, 3.1441);
            optScale = std::min(0.012112 + std::pow(ply + 3.22713, 0.46866) * optC,
                                0.19404 * double(limits.time[us]) / double(timeLeft))
                     * originalTimeAdjust;
            maxScale = std::min(6.873, maxC + ply / 12.352);
        }
        else
        {
            optScale = std::min(0.0084 + std::pow(ply + 3.0, 0.5) * 0.0042,
                                0.20 * double(limits.time[us]) / double(timeLeft));
            maxScale = std::min(7.0, 4.0 + ply / 12.0);
        }
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

    optimumTime = TimePoint(optScale * double(timeLeft));
    maximumTime = TimePoint(std::min((TmSfAlloc ? 0.8097 : 0.84) * double(limits.time[us]) - moveOverhead,
                                     maxScale * double(optimumTime)));

    optimumTime = std::max<TimePoint>(optimumTime, 1);
    maximumTime = std::max<TimePoint>(maximumTime, 1);
}

}  // namespace one
