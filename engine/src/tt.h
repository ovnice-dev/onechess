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
//
// Des paquets de 32 octets (une demi-ligne de cache) de trois fiches de
// 10 octets. Une fiche garde 16 bits de la cle, le score, l'evaluation brute,
// le meilleur coup, la profondeur et un octet « infos » :
//   bits 0-1 : borne, bit 2 : noeud PV, bits 3-7 : age (numero de recherche).

#pragma once

#include <cstddef>
#include <cstdint>

#include "misc.h"
#include "types.h"

namespace one {

class TranspositionTable;

struct TTEntry {
    Move  move() const { return Move(coup); }
    Value value() const { return score; }
    Value eval() const { return evalBrute; }
    Depth depth() const { return Depth(prof) + DEPTH_ENTRY_TT; }
    bool  is_pv() const { return (infos & 4) != 0; }
    Bound bound() const { return Bound(infos & 3); }

    // Ecrit la fiche, si elle n'en remplace pas une plus precieuse
    void save(Key cle, Value v, bool pv, Bound b, Depth d, Move m, Value evalBrute, std::uint8_t age);

   private:
    friend class TranspositionTable;
    int vieillesse(std::uint8_t ageActuel) const;  // recherches ecoulees depuis l'ecriture, x 8

    std::uint16_t cle16;
    std::uint8_t  prof;   // profondeur - DEPTH_ENTRY_TT ; 0 = fiche vide
    std::uint8_t  infos;
    std::int16_t  score;
    std::int16_t  evalBrute;
    std::uint16_t coup;
};

struct alignas(32) Paquet {
    TTEntry fiches[3];
    char    bourrage[2];
};
static_assert(sizeof(Paquet) == 32);

class TranspositionTable {
   public:
    ~TranspositionTable() { aligned_large_pages_free(paquets); }

    void resize(std::size_t mega, int fils);  // Mo demandes ; moins si le systeme refuse
    void clear(int fils);
    void new_search() { age = std::uint8_t(age + 8); }  // l'age vit dans les bits 3-7

    // La fiche de cette cle (trouvee = vrai), sinon celle a remplacer
    TTEntry* probe(Key cle, bool& trouvee) const;
    int      hashfull() const;  // pour mille, sur un echantillon
    std::uint8_t generation() const { return age; }

    // Paquet de la cle : partie haute de cle x nombre de paquets (repartition
    // uniforme, taille quelconque). Sert aussi au prechargement.
    Paquet* first_entry(Key cle) const {
        return paquets + std::size_t((unsigned __int128) cle * nbPaquets >> 64);
    }

   private:
    Paquet*      paquets   = nullptr;
    std::size_t  nbPaquets = 0;
    std::uint8_t age       = 0;
};

extern TranspositionTable TT;

}  // namespace one
