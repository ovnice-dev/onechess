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

// Protocole UCI : options, commandes, et ecriture des coups et des scores
// (reecrit pour Onechess le 15-11-2026).

#pragma once

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <map>
#include <string>

#include "types.h"

namespace one {

class Position;
class Thread;

namespace UCI {

class Option;
using OnChange = std::function<void(const Option&)>;  // appele apres chaque changement

// Une option UCI : bouton, case a cocher, texte, ou entier borne.
class Option {
   public:
    enum Genre { BOUTON, CASE, TEXTE, ENTIER };

    Option(OnChange rappel = nullptr);                            // bouton
    Option(bool v, OnChange rappel = nullptr);                    // case a cocher
    Option(const char* v, OnChange rappel = nullptr);             // texte
    Option(int v, int mini, int maxi, OnChange rappel = nullptr);  // entier

    Option& operator=(const std::string& v);  // « setoption » : ignore une valeur invalide
    void    operator<<(const Option& o);      // declaration (garde l'ordre d'affichage)

    operator int() const;          // entier, ou 0/1 pour une case
    operator std::string() const;  // texte

    bool hidden = false;  // reglable, mais absent de la liste « uci » (constantes internes)

    // Ligne « option name ... » de la commande uci
    void        decrire(std::ostream& os, const std::string& nom) const;
    std::size_t ordre() const { return rang; }

   private:
    Genre       genre = BOUTON;
    std::string defaut, valeur;
    int         mini = 0, maxi = 0;
    std::size_t rang = 0;
    OnChange    rappel;
};

// Les noms d'options se comparent sans tenir compte des majuscules
struct SansCasse {
    bool operator()(const std::string& a, const std::string& b) const;
};
using OptionsMap = std::map<std::string, Option, SansCasse>;

void init(OptionsMap& o);
void loop(int argc, char* argv[]);

std::string value(Value v);                // « cp 34 » ou « mate 5 »
std::string wdl(Value v);                  // « wdl g n p » (pour mille)
std::string square(Square s);              // « e4 »
std::string move(Move m, bool chess960);
// Ecrit une ligne entiere sous verrou (22-11) : recherche, interface et chien de garde ecrivent en
// meme temps ; une ligne « bestmove » coupee par une autre serait lue comme un coup illegal.
void sortie(const std::string& ligne);   // « e2e4 », « e7e8q », roque selon la variante
std::string pv(const Thread& th, Depth depth);
Move        to_move(const Position& pos, std::string& texte);

}  // namespace UCI

std::ostream& operator<<(std::ostream& os, const UCI::OptionsMap& options);

extern UCI::OptionsMap Options;

}  // namespace one
