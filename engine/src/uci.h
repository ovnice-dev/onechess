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

// Le callback est appele apres chaque changement : c'est lui qui
// redimensionne la table de transposition ou reconstruit les tables de
// reduction quand une constante reglable bouge.
using OnChange = std::function<void(const Option&)>;

class Option {
   public:
    Option(OnChange = nullptr);
    Option(bool v, OnChange = nullptr);
    Option(const char* v, OnChange = nullptr);
    Option(int v, int minv, int maxv, OnChange = nullptr);

    Option& operator=(const std::string& v);
    void    operator<<(const Option& o);

    operator int() const;
    operator std::string() const;

    std::string defaultValue, currentValue, type;
    int         min = 0, max = 0;
    std::size_t idx = 0;
    OnChange    on_change;
};

struct CaseInsensitiveLess {
    bool operator()(const std::string&, const std::string&) const;
};

using OptionsMap = std::map<std::string, Option, CaseInsensitiveLess>;

void init(OptionsMap&);
void loop(int argc, char* argv[]);

std::string value(Value v);
std::string square(Square s);
std::string move(Move m, bool chess960);
std::string pv(const Thread& th, Depth depth);
Move        to_move(const Position& pos, std::string& str);

}  // namespace UCI

std::ostream& operator<<(std::ostream& os, const UCI::OptionsMap& om);

extern UCI::OptionsMap Options;

}  // namespace one
