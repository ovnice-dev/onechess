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

// Gestion du temps (reecrite pour Onechess le 15-11-2026).

#include "timeman.h"

#include <algorithm>
#include <cmath>

#include "tune.h"

namespace one {

TimeManagement Time;

void TimeManagement::init(Search::LimitsType& lim, Color moi, int ply, int marge, int) {
    debut = lim.startTime;

    if (lim.movetime)  // temps impose
    {
        cible = plafond = std::max<TimePoint>(lim.movetime - marge, 1);
        return;
    }
    if (!lim.use_time_management())  // profondeur ou noeuds imposes, ou analyse
    {
        cible = plafond = 0;
        return;
    }

    const double reste   = double(lim.time[moi]);
    const int    aJouer  = lim.movestogo ? std::min(lim.movestogo, 50) : 50;  // horizon de calcul
    // Temps utilisable sur l'horizon : le reste, les increments a venir, moins
    // la latence de l'interface a chaque coup.
    const TimePoint dispo =
      std::max<TimePoint>(1, lim.time[moi] + lim.inc[moi] * (aJouer - 1) - marge * (2 + aJouer));

    double partCible, multPlafond;
    if (!lim.movestogo)
    {
        // Mort subite : la part grandit avec la racine du numero de demi-coup
        // (le milieu de partie decide), bornee a 20 % du reste.
        // (reglables depuis le 22-11 ; valeurs de depart 0,0084 ; 0,0042 ; 0,20 ; 7 ; 4 ; 12)
        partCible   = std::min(Tune::TmSdBase / 10000.0 + std::pow(ply + 3.0, 0.5) * (Tune::TmSdPente / 10000.0),
                               Tune::TmSdCap / 1000.0 * reste / double(dispo));
        multPlafond = std::min(Tune::TmSdPlafMax / 10.0, Tune::TmSdPlafBase / 10.0 + ply / (Tune::TmSdPlafPly / 10.0));
    }
    else
    {
        // Cadence repetee (CCRL 40/15) ; constantes reglables depuis le 20-11
        // (valeurs de depart : 0,88 ; 116,4 ; 0,88 ; 6,3 ; 1,5 ; 0,11)
        // TmCtrlPly (revue du 22-11, en test) : compter les demi-coups depuis le debut du controle
        // en cours ; avec le demi-coup de la partie, le debut des 2e et 3e controles prenait 2 a 3 fois
        // la moyenne et la fin n'avait presque plus rien (6 puis 2 s par coup en 40/15).
        tailleControle = std::max(tailleControle, lim.movestogo);
        const int plyM = Tune::TmCtrlPly ? 2 * (tailleControle - lim.movestogo) + (moi == BLACK) : ply;
        partCible   = std::min((Tune::TmMtgBase / 1000.0 + plyM / (Tune::TmMtgPly / 10.0)) / aJouer,
                               Tune::TmMtgCap / 1000.0 * reste / double(dispo));
        multPlafond = std::min(Tune::TmMtgPlafMax / 10.0, Tune::TmMtgPlafBase / 10.0 + Tune::TmMtgPlafPente / 1000.0 * aJouer);
    }

    // En retard a la pendule, on joue un peu plus vite (TmClockPct, eteint par defaut)
    if (Tune::TmClockPct && lim.movestogo != 1)
    {
        const double ecart = double(lim.time[moi] - lim.time[~moi]) / (1.0 + reste + double(lim.time[~moi]));
        partCible *= 1.0 + Tune::TmClockPct / 1000.0 * std::min(ecart, 0.0);
    }

    cible   = std::max<TimePoint>(TimePoint(partCible * double(dispo)), 1);
    plafond = std::max<TimePoint>(TimePoint(std::min(Tune::TmPlafReste / 1000.0 * reste - marge, multPlafond * double(TimePoint(partCible * double(dispo))))), 1);
}

}  // namespace one
