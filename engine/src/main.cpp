#include <iostream>

#include "bitboard.h"
#include "evaluate.h"
#include "nnue.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"

using namespace one;

int main(int argc, char* argv[]) {
    std::cout << engine_info() << std::endl;

    Bitboards::init();
    Position::init();
    Eval::init();
    Search::init();

    UCI::init(Options);

    // Le reseau est cherche a cote de l'executable ; son absence n'est pas une
    // erreur, le moteur retombe simplement sur l'evaluation PSQT.
    NNUE::load("onechess.nnue");
    std::cout << NNUE::info() << std::endl;

    Threads.set(std::size_t(int(Options["Threads"])));
    TT.resize(std::size_t(int(Options["Hash"])), int(Options["Threads"]));

    UCI::loop(argc, argv);

    Threads.set(0);
    return 0;
}
