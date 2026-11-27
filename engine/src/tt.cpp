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

// Table de transposition (reecrite pour Onechess le 15-11-2026).

#include "tt.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include "prof.h"
#include "tune.h"

namespace one {

TranspositionTable TT;

// Age relatif, en multiples de 8 (bits 3-7 seulement). Le +256 garde la
// difference positive quand le compteur de 8 bits vient de reboucler.
int TTEntry::vieillesse(std::uint8_t ageActuel) const {
    return (256 + 7 + ageActuel - infos) & 0xF8;
}

// Le coup est garde des qu'on en a un (ou que la fiche change de position) :
// meme une fiche sans score utile reste un bon indice de tri. Le reste n'est
// ecrit que si la nouvelle information vaut mieux : borne exacte, autre
// position, fiche ancienne, ou profondeur pas trop inferieure (une fiche PV
// compte pour deux plis de plus).
void TTEntry::save(Key cle, Value v, bool pv, Bound b, Depth d, Move m, Value eb, std::uint8_t ageActuel) {
    PROF_SCOPE(TTSAVE);
    // seule l'evaluation brute entre ici (la corrigee se recorrigerait a chaque lecture)
    assert(eb == VALUE_NONE || std::abs(eb) < 20000);

    const std::uint16_t c16    = std::uint16_t(cle);
    const bool          autre  = c16 != cle16;
    const int           nouvel = d - DEPTH_ENTRY_TT;
    if (m || autre)
        coup = m.raw();
    if (b == BOUND_EXACT || autre || nouvel + 2 * int(pv) > int(prof) - 4 || vieillesse(ageActuel))
    {
        assert(nouvel > 0 && nouvel < 256);
        cle16     = c16;
        prof      = std::uint8_t(nouvel);
        infos     = std::uint8_t(ageActuel | (pv ? 4 : 0) | b);
        score     = std::int16_t(v);
        evalBrute = std::int16_t(eb);
    }
}

void TranspositionTable::resize(std::size_t mega, int fils) {
    aligned_large_pages_free(paquets);
    paquets = nullptr;
    // Si le systeme refuse, on retente avec deux fois moins (et on le dit) :
    // une table absente ferait planter la premiere sonde.
    for (std::size_t mo = mega ? mega : 1; mo && !paquets; mo /= 2)
    {
        nbPaquets = mo * 1024 * 1024 / sizeof(Paquet);
        paquets   = static_cast<Paquet*>(aligned_large_pages_alloc(nbPaquets * sizeof(Paquet)));
        if (paquets && mo != mega)
            std::cout << "info string Hash " << mega << " Mo refuse par le systeme, " << mo << " Mo alloues"
                      << std::endl;
    }
    clear(fils);
}

// Remise a zero partagee entre les fils (une grosse table prend sinon des
// secondes avant le premier coup).
void TranspositionTable::clear(int fils) {
    if (!paquets)
        return;
    const int                n     = std::max(fils, 1);
    const std::size_t        tranche = nbPaquets / n;
    std::vector<std::thread> equipe;
    for (int i = 0; i < n; ++i)
    {
        const std::size_t debut = tranche * i;
        const std::size_t taille = i + 1 < n ? tranche : nbPaquets - debut;
        equipe.emplace_back([=, this]() { std::memset(paquets + debut, 0, taille * sizeof(Paquet)); });
    }
    for (std::thread& t : equipe)
        t.join();
    age = 0;
}

TTEntry* TranspositionTable::probe(Key cle, bool& trouvee) const {
    PROF_SCOPE(TTPROBE);
    TTEntry* const      f   = first_entry(cle)->fiches;
    const std::uint16_t c16 = std::uint16_t(cle);

    // La position, ou une place vide : on la rajeunit au passage
    for (int i = 0; i < 3; ++i)
        if (f[i].cle16 == c16 || f[i].prof == 0)
        {
            // TtRajeunir (en test, 19-11) : 0 = l'age ne change qu'a l'ecriture, comme
            // Stockfish 19 ; plus de ligne de cache salie a chaque sonde
            if (Tune::TtRajeunir)
                f[i].infos = std::uint8_t(age | (f[i].infos & 7));
            trouvee    = f[i].prof != 0;
            return &f[i];
        }

    // Sinon la moins precieuse : profondeur moins deux plis par recherche d'age
    // (une fiche profonde mais perimee finit par ceder sa place).
    auto     valeur = [this](const TTEntry& e) { return int(e.prof) - e.vieillesse(age) * 2; };
    TTEntry* moins  = &f[0];
    for (int i = 1; i < 3; ++i)
        if (valeur(f[i]) < valeur(*moins))
            moins = &f[i];
    trouvee = false;
    return moins;
}

int TranspositionTable::hashfull() const {
    if (!paquets)
        return 0;
    int pleines = 0;
    for (std::size_t i = 0; i < 1000 && i < nbPaquets; ++i)
        for (const TTEntry& e : paquets[i].fiches)
            pleines += e.prof != 0 && (e.infos & 0xF8) == age;
    return pleines / 3;
}

}  // namespace one
