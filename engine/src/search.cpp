#include "search.h"
#include "prof.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <chrono>
#include <iostream>
#include <thread>
#include <sstream>

#include "evaluate.h"
#include "misc.h"
#include "movegen.h"
#include "movepick.h"
#include "position.h"
#include "syzygy.h"
#include "thread.h"
#include "timeman.h"
#include "tt.h"
#include "tune.h"
#include "uci.h"

namespace one {

namespace Tune {
#define TUNABLE_DEFINE(name, def, lo, hi) int name = def;
TUNABLE_LIST(TUNABLE_DEFINE)
#undef TUNABLE_DEFINE
}

namespace Search {
LimitsType Limits;
}

using namespace Search;
using namespace Tune;

namespace {

enum NodeType {
    NonPV,
    PV,
    Root
};

// Table de reduction des coups tardifs, reconstruite a chaque changement des
// constantes pour que le reglage par UCI prenne effet sans recompilation.
int LMRTable[64][64];
int LMRTableF[64][64];  // la meme, en 1/1024 de pli (LmrFrac)

Depth reduction(Depth d, int mn) { return LMRTable[std::min(int(d), 63)][std::min(mn, 63)]; }
int   reduction_f(Depth d, int mn) { return LMRTableF[std::min(int(d), 63)][std::min(mn, 63)]; }

// La marge se resserre quand la position s'ameliore : l'evaluation a plus de
// chances de tenir, donc on ose couper de plus pres. C'est le sens qui compte
// ici -- l'inverser coute des dizaines d'Elo.
Value futility_margin(Depth d, bool improving) {
    return Value(FutilityMargin * (d - improving));
}

int stat_bonus(Depth d) { return std::min(StatBonusMul * d - StatBonusSub, StatBonusMax); }
int stat_malus(Depth d) { return std::min(StatMalusMul * d - StatMalusSub, StatMalusMax); }

// Un score de mat est stocke en distance depuis le noeud courant, pas depuis
// la racine : c'est ce qui rend l'entree reutilisable ailleurs dans l'arbre.
Value value_to_tt(Value v, int ply) {
    assert(v != VALUE_NONE);
    return v >= VALUE_TB_WIN_IN_MAX_PLY    ? Value(v + ply)
         : v <= VALUE_TB_LOSS_IN_MAX_PLY   ? Value(v - ply)
                                           : v;
}

Value value_from_tt(Value v, int ply, int r50c) {
    if (v == VALUE_NONE)
        return VALUE_NONE;

    if (v >= VALUE_TB_WIN_IN_MAX_PLY)
    {
        // Un mat plus lointain que la regle des 50 coups n'en est pas un
        if (VALUE_MATE - v > 99 - r50c)
            return VALUE_TB_WIN_IN_MAX_PLY - 1;
        return Value(v - ply);
    }

    if (v <= VALUE_TB_LOSS_IN_MAX_PLY)
    {
        if (VALUE_MATE + v > 99 - r50c)
            return VALUE_TB_LOSS_IN_MAX_PLY + 1;
        return Value(v + ply);
    }

    return v;
}

void update_pv(Move* pv, Move move, const Move* childPv) {
    for (*pv++ = move; childPv && *childPv != Move::none();)
        *pv++ = *childPv++;
    *pv = Move::none();
}

// Applique l'ecart appris entre evaluation statique et score reellement
// trouve par la recherche, pour les positions de meme structure de pions. La
// forme quadratique laisse les petites corrections petites : le terme ne mord
// que lorsque l'ecart est confirme.
// Cles des pieces mineures et majeures, calculees a la volee a partir des
// bitboards : un melangeur par bitboard, puis un ou exclusif. Les rois sont
// inclus, comme dans la cle des pieces mineures de Stockfish. Le cout est de
// quelques multiplications par appel, negligeable devant le reseau.
inline Key bb_mix(Bitboard b, Key sel) {
    Key x = b ^ sel;
    x ^= x >> 31;
    x *= 0x7fb5d329728ea185ULL;
    x ^= x >> 27;
    x *= 0x81dadef4bc2dd44dULL;
    x ^= x >> 33;
    return x;
}

inline Key minor_key(const Position& pos) {
    return bb_mix(pos.pieces(WHITE, KNIGHT), 0x11) ^ bb_mix(pos.pieces(BLACK, KNIGHT), 0x22)
         ^ bb_mix(pos.pieces(WHITE, BISHOP), 0x33) ^ bb_mix(pos.pieces(BLACK, BISHOP), 0x44)
         ^ bb_mix(pos.pieces(WHITE, KING), 0x55) ^ bb_mix(pos.pieces(BLACK, KING), 0x66);
}

inline Key major_key(const Position& pos) {
    return bb_mix(pos.pieces(WHITE, ROOK), 0x77) ^ bb_mix(pos.pieces(BLACK, ROOK), 0x88)
         ^ bb_mix(pos.pieces(WHITE, QUEEN), 0x99) ^ bb_mix(pos.pieces(BLACK, QUEEN), 0xaa)
         ^ bb_mix(pos.pieces(WHITE, KING), 0xbb) ^ bb_mix(pos.pieces(BLACK, KING), 0xcc);
}

Value corrected_eval(Value v, const Thread* th, const Position& pos, const Stack* ss) {
    PROF_SCOPE(CORR);
    const int cv = th->correctionHistory[pos.side_to_move()][correction_index(pos.pawn_key())];
    v += Value(CorrHistWeight * cv * std::abs(cv) / CorrHistDiv);

    // Composante non-pions, une table par couleur de pieces. Les tables sont
    // toujours tenues a jour ; tant que CorrNpWeight vaut 0 elles ne sont pas
    // lues, et le jeu est strictement celui d'avant.
    if (CorrNpWeight)
    {
        const Color us = pos.side_to_move();
        const int w = th->nonPawnCorrection[WHITE][us][correction_index(pos.non_pawn_key(WHITE))];
        const int b = th->nonPawnCorrection[BLACK][us][correction_index(pos.non_pawn_key(BLACK))];
        v += Value(CorrNpWeight * (w * std::abs(w) + b * std::abs(b)) / CorrHistDiv);
    }

    // Composantes pieces mineures et pieces majeures (Stockfish, Rudim) :
    // une table chacune, lue seulement si son poids est non nul.
    if (CorrMinorWeight)
    {
        const int c = th->minorCorrection[pos.side_to_move()][correction_index(minor_key(pos))];
        v += Value(CorrMinorWeight * c * std::abs(c) / CorrHistDiv);
    }
    if (CorrMajorWeight)
    {
        const int c = th->majorCorrection[pos.side_to_move()][correction_index(major_key(pos))];
        v += Value(CorrMajorWeight * c * std::abs(c) / CorrHistDiv);
    }

    // Composante de continuation : l'ecart constate apres la meme paire de
    // coups -- le precedent, et celui qui l'a precede. Repris de Stockfish.
    // Tenue a jour en permanence, lue seulement si CorrContWeight est non nul.
    if (CorrContWeight)
    {
        const Move m = (ss - 1)->currentMove;
        if (m.is_ok())
        {
            const int c =
              (*(ss - 2)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()];
            v += Value(CorrContWeight * c * std::abs(c) / CorrHistDiv);
        }
    }
    return std::clamp(v, Value(VALUE_TB_LOSS_IN_MAX_PLY + 1), Value(VALUE_TB_WIN_IN_MAX_PLY - 1));
}

// Un coup est juge sur la reponse qu'il apporte au coup precedent (pli 1),
// mais aussi sur la facon dont il prolonge notre propre plan (plis 2, 4, 6).
void update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus) {
    // ContWeightsSf : poids propres a chaque pli et multiplicateur de
    // coherence (plus d'entrees deja positives, plus de poids), Stockfish.
    if (ContWeightsSf)
    {
        static constexpr int plis[]  = {1, 2, 3, 4, 6};
        static constexpr int poids[] = {520, 390, 145, 251, 209};
        static constexpr int coh[]   = {94, 103, 110, 106, 119, 126};
        int                  positifs = 0;
        for (int k = 0; k < 5; ++k)
        {
            const int i = plis[k];
            if (ss->inCheck && i > 2)
                break;
            if ((ss - i)->currentMove.is_ok())
            {
                auto& e = (*(ss - i)->continuationHistory)[pc][to];
                positifs += e > 0;
                e << bonus * poids[k] * coh[positifs] / 65536 + 73 * (i < 2);
            }
        }
        return;
    }

    for (int i : {1, 2, 3, 4, 6})
    {
        if (ss->inCheck && i > 2)
            break;

        if ((ss - i)->currentMove.is_ok())
            (*(ss - i)->continuationHistory)[pc][to] << bonus / (1 + 3 * (i == 3));
    }
}

void update_quiet_stats(const Position& pos, Stack* ss, Thread* th, Move move, int bonus) {
    if (ss->killers[0] != move)
    {
        ss->killers[1] = ss->killers[0];
        ss->killers[0] = move;
    }

    const Color us = pos.side_to_move();
    th->mainHistory[us][hist_index(move, threat_map(pos))] << bonus;
    if (ss->ply < LOW_PLY_HISTORY_SIZE)
        th->lowPlyHistory[ss->ply][move.from_to()] << bonus * 3 / 4;
    update_continuation_histories(ss, pos.moved_piece(move), move.to_sq(), bonus);
    th->pawnHistory[pawn_structure_index(pos.pawn_key())][pos.moved_piece(move)][move.to_sq()]
      << bonus;

    // Le coup precedent est refute par celui-ci
    if (((ss - 1)->currentMove).is_ok())
    {
        const Square prevSq = ((ss - 1)->currentMove).to_sq();
        th->counterMoves[pos.piece_on(prevSq)][prevSq] = move;
    }
}

// Recompense le coup qui a coupe, penalise tous ceux qui ont ete essayes en
// vain avant lui. Sans le malus, un mauvais coup essaye tot garde son rang
// indefiniment.
void update_all_stats(const Position& pos,
                      Stack*          ss,
                      Thread*         th,
                      Move            bestMove,
                      Square          prevSq,
                      Move*           quietsSearched,
                      int             quietCount,
                      Move*           capturesSearched,
                      int             captureCount,
                      Depth           depth,
                      Move            ttMove,
                      bool            pvNode) {
    PROF_SCOPE(HISTUPD);

    const Piece     moved        = pos.moved_piece(bestMove);
    const PieceType capturedType = type_of(pos.piece_on(bestMove.to_sq()));
    int             bonus        = stat_bonus(depth);

    // BonusSf : le bonus grandit si le meilleur coup etait celui de la table,
    // selon le statScore du coup precedent, et -- hors PV -- avec le nombre de
    // coups qu'il a fallu essayer avant de le trouver. Stockfish.
    if (BonusSf)
    {
        bonus += 364 * (bestMove == ttMove) * bonus / 1487 + (ss - 1)->statScore / 28;
        bonus = std::max(bonus, 0);
        if (!pvNode)
            bonus += int(std::int64_t(bonus) * (quietCount + captureCount) / 256);
    }
    const int       malus        = stat_malus(depth);

    if (!pos.capture_stage(bestMove))
    {
        update_quiet_stats(pos, ss, th, bestMove, bonus);

        // Un coup essaye tard avait deja de mauvaises statistiques, sinon il
        // aurait ete essaye plus tot : il merite un malus plus leger. Le malus
        // decroit donc d'un coup a l'autre, repris de Stockfish. Inerte tant
        // que MalusDecay vaut 1024.
        int            qMalus  = malus;
        const Bitboard threats = threat_map(pos);
        for (int i = 0; i < quietCount; ++i)
        {
            const Move m = quietsSearched[i];
            th->mainHistory[pos.side_to_move()][hist_index(m, threats)] << -qMalus;
            if (ss->ply < LOW_PLY_HISTORY_SIZE)
                th->lowPlyHistory[ss->ply][m.from_to()] << -qMalus * 3 / 4;
            th->pawnHistory[pawn_structure_index(pos.pawn_key())][pos.moved_piece(m)][m.to_sq()]
              << -qMalus;
            update_continuation_histories(ss, pos.moved_piece(m), m.to_sq(), -qMalus);
            qMalus = qMalus * MalusDecay / 1024;
        }
    }
    else
        th->captureHistory[moved][bestMove.to_sq()][capturedType] << bonus;

    // Le coup precedent a permis cette refutation : il etait donc mauvais
    if (prevSq != SQ_NONE && ((ss - 1)->moveCount == 1 + (ss - 1)->ttHit)
        && !pos.captured_piece())
        update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -malus);

    for (int i = 0; i < captureCount; ++i)
    {
        const Move m = capturesSearched[i];
        th->captureHistory[pos.moved_piece(m)][m.to_sq()][type_of(pos.piece_on(m.to_sq()))]
          << -malus;
    }
}

// Ecart constate entre l'evaluation statique et le score de la recherche :
// toutes les tables de correction l'apprennent.
void update_correction_history(const Position& pos, Stack* ss, Thread* th, int bonus) {
    const Color us = pos.side_to_move();
    th->correctionHistory[us][correction_index(pos.pawn_key())] << bonus;
    th->nonPawnCorrection[WHITE][us][correction_index(pos.non_pawn_key(WHITE))] << bonus;
    th->nonPawnCorrection[BLACK][us][correction_index(pos.non_pawn_key(BLACK))] << bonus;
    if (CorrMinorWeight)
        th->minorCorrection[us][correction_index(minor_key(pos))] << bonus;
    if (CorrMajorWeight)
        th->majorCorrection[us][correction_index(major_key(pos))] << bonus;

    const Move pm = (ss - 1)->currentMove;
    if (pm.is_ok())
        (*(ss - 2)->continuationCorrectionHistory)[pos.piece_on(pm.to_sq())][pm.to_sq()] << bonus;
}

void check_time() {
    if (!Limits.use_time_management() && !Limits.movetime && !Limits.nodes)
        return;

    if (Limits.nodes && Threads.nodes_searched() >= Limits.nodes)
        Threads.stop = true;

    if ((Limits.use_time_management() || Limits.movetime) && Time.elapsed() >= Time.maximum())
        Threads.stop = true;
}

template<NodeType nodeType>
Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth);

template<NodeType nodeType>
Value search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode);

}  // namespace

// Reconstruit les tables qui dependent des constantes reglables.
void Search::init() {
    for (int d = 0; d < 64; ++d)
        for (int m = 0; m < 64; ++m)
            LMRTable[d][m] = (d == 0 || m == 0)
                             ? 0
                             : int(LmrBase / 100.0
                                   + std::log(double(d)) * std::log(double(m))
                                       / (LmrDivisor / 100.0));
    for (int d = 0; d < 64; ++d)
        for (int m = 0; m < 64; ++m)
            LMRTableF[d][m] = (d == 0 || m == 0)
                              ? 0
                              : int(1024.0 * (LmrBase / 100.0
                                               + std::log(double(d)) * std::log(double(m))
                                                   / (LmrDivisor / 100.0)));
}

// Regle de surprise (Triumviratus) : la position qu'on attendait apres notre
// coup et la reponse prevue. Si l'adversaire joue autre chose, on se donne
// plus de temps (TmSurprisePct). Inerte a 100.
namespace {
Key ExpectedKey = 0;
int ExpectedPly = -1;
}

// ---------------------------------------------------------------------------
// Approfondissement iteratif. Chaque thread execute cette boucle sur sa propre
// copie de la position ; seule la table de transposition les relie.
// ---------------------------------------------------------------------------
void Thread::iterative_deepening() {
    PROF_SCOPE(SEARCH);
    Stack  stack[MAX_PLY + 10] = {};
    Stack* ss                  = stack + 7;
    Move   pv[MAX_PLY + 1];

    Value bestValue = -VALUE_INFINITE;
    Value alpha = -VALUE_INFINITE, beta = VALUE_INFINITE;
    Move  lastBestMove  = Move::none();
    Depth lastBestDepth = 0;
    double bestMoveChanges = 0.0;  // decroit de moitie a chaque iteration
    Value prevIterScore = -VALUE_INFINITE;  // score de l'iteration precedente

    // Les sept cases avant la racine rendent (ss - 6) toujours lisible
    for (int i = 7; i > 0; --i)
    {
        (ss - i)->continuationHistory = &continuationHistory[0][0][NO_PIECE][SQ_A1];
        (ss - i)->continuationCorrectionHistory = &continuationCorrectionHistory[NO_PIECE][SQ_A1];
        (ss - i)->staticEval          = VALUE_NONE;
    }

    for (int i = 0; i <= MAX_PLY + 2; ++i)
        (ss + i)->ply = i;

    ss->pv = pv;

    // L'historique pres de la racine ne vaut que pour CETTE racine
    lowPlyHistory.fill(0);

    rootPos.set_thread(this);

    const bool surprise = is_main() && ExpectedPly == rootPos.game_ply()
                       && rootPos.key() != ExpectedKey;

    if (rootMoves.empty())
    {
        rootMoves.emplace_back(Move::none());
        if (is_main() && !Limits.silent)
            std::cout << "info depth 0 score "
                      << UCI::value(rootPos.checkers() ? -VALUE_MATE : VALUE_DRAW)
                      << std::endl
                      << "bestmove (none)" << std::endl;
        return;
    }

    // Le filtre des tables de finales a la racine est fait une fois pour tous
    // les threads, dans ThreadPool::start_thinking().

    const int multiPV = 1;
    (void) multiPV;

    while (++rootDepth < MAX_PLY && !Threads.stop
           && !(Limits.depth && is_main() && rootDepth > Limits.depth))
    {
        // Les helpers partent avec un decalage de profondeur nul : la
        // diversite vient des historiques, qui divergent naturellement.
        for (RootMove& rm : rootMoves)
            rm.previousScore = rm.score;

        selDepth = 0;

        // ------------------------------------------------- fenetre d'aspiration
        Value delta = VALUE_ZERO;
        Value avg   = rootMoves[0].averageScore;

        if (rootDepth >= AspirationMinDepth && avg != -VALUE_INFINITE)
        {
            delta = Value(AspirationDelta + std::abs(avg) / 10000);
            alpha = std::max(avg - delta, -VALUE_INFINITE);
            beta  = std::min(avg + delta, VALUE_INFINITE);
        }
        else
        {
            alpha = -VALUE_INFINITE;
            beta  = VALUE_INFINITE;
        }

        int failedHighCnt = 0;

        while (true)
        {
            // Un echec haut repete signifie que la recherche se noie : on
            // reduit la profondeur pour rendre l'echec exploitable.
            const Depth adjustedDepth =
              std::max(1, rootDepth - failedHighCnt);

            rootDelta = beta - alpha;
            bestValue = search<Root>(rootPos, ss, alpha, beta, adjustedDepth, false);

            std::stable_sort(rootMoves.begin(), rootMoves.end());

            if (Threads.stop)
                break;

            if (bestValue <= alpha)
            {
                // Echec bas : on n'elargit que le bord qui a cede
                beta  = (alpha + beta) / 2;
                alpha = std::max(bestValue - delta, -VALUE_INFINITE);
                failedHighCnt = 0;
            }
            else if (bestValue >= beta)
            {
                beta = std::min(bestValue + delta, VALUE_INFINITE);
                ++failedHighCnt;
            }
            else
                break;

            delta += delta / 3;

            assert(alpha >= -VALUE_INFINITE && beta <= VALUE_INFINITE);
        }

        bestMoveChanges /= 2.0;

        if (!Threads.stop)
            completedDepth = rootDepth;

        if (rootMoves[0].pv[0] != lastBestMove)
        {
            lastBestMove  = rootMoves[0].pv[0];
            lastBestDepth = rootDepth;
            bestMoveChanges += 1.0;
        }

        // Une iteration interrompue avant qu'un coup ait un score ne produit
        // rien : il ne faut surtout pas emettre une ligne vide, certaines
        // interfaces la traitent comme un message inattendu.
        if (is_main())
        {
            const std::string line = UCI::pv(*this, rootDepth);
            if (!line.empty())
                std::cout << line << std::endl;
        }

        if (Threads.stop)
            break;

        // -------------------------------------------- arret par le temps
        if (is_main() && Limits.use_time_management() && !Limits.movetime)
        {
            // Un meilleur coup qui bouge encore merite qu'on prolonge. Le
            // code d'origine faisait l'inverse : lastBestDepth vaut rootDepth
            // quand le coup VIENT de changer, et c'etait le cas stable qui
            // recevait les 35 % de plus. Les valeurs par defaut le reproduisent
            // a l'identique ; le correctif se teste au temps (RECHERCHES.md).
            const bool changed = rootDepth == lastBestDepth;
            double     factor  = (changed ? TmChangedPct : TmStablePct) / 100.0;

            // Le NOMBRE de changements, et non le seul fait qu'il y en ait eu
            // un : Stockfish calcule 1,077 + 2,229 x changements / nbFils.
            // Inerte tant que TmInstMul vaut 0.
            if (TmInstMul)
                factor *= 1.0 + TmInstMul / 1000.0 * bestMoveChanges;

            // Un score qui baisse d'une iteration a l'autre merite du temps ;
            // un score qui monte, un peu moins.
            if (TmFallPct && prevIterScore != -VALUE_INFINITE)
                factor *= std::clamp(1.0 + TmFallPct / 1000.0 * double(prevIterScore - bestValue),
                                     0.8, 1.5);
            prevIterScore = bestValue;

            if (surprise)
                factor *= TmSurprisePct / 100.0;

            // Un seul coup legal : rien a chercher
            const bool forced = TmSingleMs && rootMoves.size() == 1
                             && Time.elapsed() >= std::min<TimePoint>(TmSingleMs, Time.optimum());


            // L'effort : quand le meilleur coup a absorbe l'essentiel des
            // noeuds, le choix est fait et on peut jouer plus tot. Inerte
            // tant que TmEffortPct vaut 0.
            if (TmEffortPct)
            {
                const std::uint64_t tot =
                  std::max<std::uint64_t>(1, nodes.load(std::memory_order_relaxed));
                const double part = double(rootMoves[0].effort) / double(tot);
                factor *= std::clamp(
                  1.0 - TmEffortPct / 1000.0 * std::max(0.0, part - 0.5), 0.69, 1.0);
            }

            // Ne pas commencer une iteration qu'on ne pourra pas finir : la
            // prochaine coute a peu pres autant que tout ce qui precede.
            // TmNoStartPct = 100 reproduit l'ancien comportement.
            const double budget = double(Time.optimum()) * factor * (TmNoStartPct / 100.0);

            if (forced || Time.elapsed() > TimePoint(budget))
            {
                Threads.stop = true;
                break;
            }
        }

        if (is_main() && Limits.movetime && Time.elapsed() >= Time.optimum())
        {
            Threads.stop = true;
            break;
        }
    }

    if (!is_main())
        return;

    // "go infinite" impose de ne rendre le coup qu'apres un "stop" explicite.
    // C'est pour cela que la boucle UCI ne doit surtout pas attendre ici :
    // sinon elle ne lirait jamais ce "stop".
    while (!Threads.stop && Limits.infinite)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

    Threads.stop = true;
    Threads.wait_for_helpers();

    if (Limits.silent)
        return;

    const Thread* best = Threads.best_thread();
    const bool    c960 = rootPos.is_chess960();

    // Position attendue au prochain coup : notre coup, puis la reponse prevue
    ExpectedPly = -1;
    if (best->rootMoves[0].pv.size() > 1)
    {
        StateInfo s1, s2;
        const Move m1 = best->rootMoves[0].pv[0], m2 = best->rootMoves[0].pv[1];
        rootPos.do_move(m1, s1);
        if (rootPos.pseudo_legal(m2) && rootPos.legal(m2))
        {
            rootPos.do_move(m2, s2);
            ExpectedKey = rootPos.key();
            ExpectedPly = rootPos.game_ply();
            rootPos.undo_move(m2);
        }
        rootPos.undo_move(m1);
    }

    std::cout << "bestmove " << UCI::move(best->rootMoves[0].pv[0], c960);
    if (best->rootMoves[0].pv.size() > 1)
        std::cout << " ponder " << UCI::move(best->rootMoves[0].pv[1], c960);
    std::cout << std::endl;
}

namespace {

// ---------------------------------------------------------------------------
// Quiescence : on ne s'arrete que sur une position calme. Sans elle, le moteur
// evalue au milieu d'un echange et se trompe systematiquement d'une piece.
// ---------------------------------------------------------------------------
template<NodeType nodeType>
Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth) {
    constexpr bool PvNode = nodeType == PV;

    static_assert(nodeType != Root, "pas de quiescence a la racine");
    assert(alpha >= -VALUE_INFINITE && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));
    assert(depth <= 0);

    Move      pv[MAX_PLY + 1];
    StateInfo st;

    Key      posKey;
    Move     move, bestMove;
    Value    bestValue, value, futilityBase;
    bool     pvHit, givesCheck, capture;
    int      moveCount;
    Thread*  thisThread = pos.this_thread();
    TTEntry* tte;
    bool     ttHit;

    if (PvNode)
    {
        (ss + 1)->pv = pv;
        ss->pv[0]    = Move::none();
    }

    bestMove             = Move::none();
    ss->inCheck          = pos.checkers();
    moveCount            = 0;
    const Value oldAlpha = alpha;

    // L'evaluation BRUTE est ce qui part en table de transposition ; la
    // corrigee ne sert qu'ici. Stocker la corrigee ferait reappliquer la
    // correction a chaque relecture, et le terme se composerait jusqu'a
    // saturer la borne.
    Value unadjustedStaticEval = VALUE_NONE;

    // Nulle ou profondeur maximale
    if (pos.is_draw(ss->ply) || ss->ply >= MAX_PLY)
        return (ss->ply >= MAX_PLY && !ss->inCheck) ? Eval::evaluate(pos) : VALUE_DRAW;

    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    // Deux profondeurs seulement en quiescence : 0 quand on donne encore les
    // echecs, -1 ensuite. Cela evite de multiplier les entrees de TT.
    const Depth ttDepth = ss->inCheck || depth >= DEPTH_QS ? DEPTH_QS : DEPTH_QS - 1;

    posKey = pos.key();
    tte    = TT.probe(posKey, ttHit);
    ss->ttHit = ttHit;
    const Value ttValue = ttHit ? value_from_tt(tte->value(), ss->ply, pos.rule50_count())
                                : VALUE_NONE;
    const Move ttMove = ttHit ? tte->move() : Move::none();
    pvHit             = ttHit && tte->is_pv();

    if (!PvNode && ttHit && tte->depth() >= ttDepth && ttValue != VALUE_NONE
        && (tte->bound() & (ttValue >= beta ? BOUND_LOWER : BOUND_UPPER)))
        return ttValue;

    // ------------------------------------------------------------ stand-pat
    if (ss->inCheck)
    {
        ss->staticEval = VALUE_NONE;
        bestValue = futilityBase = -VALUE_INFINITE;
    }
    else
    {
        unadjustedStaticEval =
          (ttHit && tte->eval() != VALUE_NONE) ? tte->eval() : Eval::evaluate(pos);

        ss->staticEval = bestValue = corrected_eval(unadjustedStaticEval, thisThread, pos, ss);

        // Le score de la TT est une information plus fine que l'evaluation
        if (ttValue != VALUE_NONE
            && (tte->bound() & (ttValue > bestValue ? BOUND_LOWER : BOUND_UPPER)))
            bestValue = ttValue;

        if (bestValue >= beta)
        {
            // QsFailFirm : le stand-pat rend la moyenne avec beta plutot que
            // l'evaluation entiere (« fail-firm » de Stockfish).
            if (QsFailFirm && std::abs(bestValue) < VALUE_TB_WIN_IN_MAX_PLY)
                bestValue = (bestValue + beta) / 2;
            if (!ttHit)
                tte->save(posKey, value_to_tt(bestValue, ss->ply), false, BOUND_LOWER,
                          DEPTH_ENTRY_TT + 1, Move::none(), unadjustedStaticEval,
                          TT.generation());
            return bestValue;
        }

        if (bestValue > alpha)
            alpha = bestValue;

        futilityBase = bestValue + QsFutilityMargin;
    }

    const PieceToHistory* contHist[] = {
      (ss - 1)->continuationHistory, (ss - 2)->continuationHistory,
      (ss - 3)->continuationHistory, (ss - 4)->continuationHistory,
      (ss - 5)->continuationHistory, (ss - 6)->continuationHistory};

    const Square prevSq = ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq()
                                                          : SQ_NONE;
    (void) prevSq;

    // En echec, MovePicker genere les parades ; sinon uniquement les captures
    // et la promotion en dame.
    MovePicker mp(pos, ttMove, depth, &thisThread->mainHistory, &thisThread->captureHistory,
                  contHist, &thisThread->pawnHistory, ss->killers, Move::none(),
                  &thisThread->lowPlyHistory, ss->ply);

    while ((move = mp.next_move()) != Move::none())
    {
        assert(move.is_ok());

        if (!pos.legal(move))
            continue;

        givesCheck = pos.gives_check(move);
        capture    = pos.capture_stage(move);

        ++moveCount;

        // ------------------------------------------------------- elagages
        if (bestValue > VALUE_TB_LOSS_IN_MAX_PLY && pos.non_pawn_material(pos.side_to_move()))
        {
            // Delta : meme en gagnant la piece on reste loin d'alpha
            if (!givesCheck && move.to_sq() != prevSq && futilityBase > VALUE_TB_LOSS_IN_MAX_PLY
                && move.type_of() != PROMOTION)
            {
                if (moveCount > 2)
                    continue;

                const Value futilityValue =
                  futilityBase + PieceValue[pos.piece_on(move.to_sq())];

                if (futilityValue <= alpha)
                {
                    bestValue = std::max(bestValue, futilityValue);
                    continue;
                }

                // Rien a gagner et la case est defendue
                if (futilityBase <= alpha && !pos.see_ge(move, 1))
                {
                    bestValue = std::max(bestValue, futilityBase);
                    continue;
                }
            }

            // Toute capture perdante au sens SEE est ecartee
            if (!pos.see_ge(move, -QsDeltaMargin))
                continue;
        }

        ss->currentMove = move;
        ss->continuationHistory =
          &thisThread->continuationHistory[ss->inCheck][capture][pos.moved_piece(move)]
                                          [move.to_sq()];
        ss->continuationCorrectionHistory =
          &thisThread->continuationCorrectionHistory[pos.moved_piece(move)][move.to_sq()];

        prefetch(TT.first_entry(pos.key_after(move)));

        thisThread->nodes.fetch_add(1, std::memory_order_relaxed);
        pos.do_move(move, st, givesCheck);
        value = -qsearch<nodeType>(pos, ss + 1, -beta, -alpha, depth - 1);
        pos.undo_move(move);

        assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

        if (value > bestValue)
        {
            bestValue = value;

            if (value > alpha)
            {
                bestMove = move;

                if (PvNode)
                    update_pv(ss->pv, move, (ss + 1)->pv);

                if (PvNode && value < beta)
                    alpha = value;
                else
                    break;  // coupure beta
            }
        }
    }

    // Zero coup legal alors qu'on est en echec : c'est mat.
    if (ss->inCheck && bestValue == -VALUE_INFINITE)
    {
        assert(!MoveList<LEGAL>(pos).size());
        return mated_in(ss->ply);
    }

    if (QsFailFirm && bestValue >= beta && std::abs(bestValue) < VALUE_TB_WIN_IN_MAX_PLY)
        bestValue = (bestValue + beta) / 2;

    tte->save(posKey, value_to_tt(bestValue, ss->ply), pvHit,
              bestValue >= beta    ? BOUND_LOWER
              : PvNode && bestValue > oldAlpha ? BOUND_EXACT
                                               : BOUND_UPPER,
              ttDepth, bestMove, unadjustedStaticEval, TT.generation());

    assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

    return bestValue;
}

// ---------------------------------------------------------------------------
// Recherche principale : alpha-beta a variation principale.
// ---------------------------------------------------------------------------
template<NodeType nodeType>
Value search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode) {
    constexpr bool PvNode   = nodeType != NonPV;
    constexpr bool rootNode = nodeType == Root;

    if (depth <= 0)
        return qsearch<PvNode ? PV : NonPV>(pos, ss, alpha, beta, 0);

    assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));
    assert(0 < depth && depth < MAX_PLY);
    assert(!(PvNode && cutNode));

    Move      pv[MAX_PLY + 1], capturesSearched[32], quietsSearched[64];
    StateInfo st;

    Key   posKey;
    Move  move, excludedMove, bestMove;
    Depth extension, newDepth;
    Value bestValue, value, eval, maxValue, probCutBeta;
    bool  givesCheck, improving, priorCapture;
    bool  capture, ttCapture;
    int   moveCount, captureCount, quietCount;

    Value unadjustedStaticEval = VALUE_NONE;

    Thread* thisThread = pos.this_thread();
    ss->inCheck        = pos.checkers();
    priorCapture       = pos.captured_piece();
    const Color us     = pos.side_to_move();
    moveCount = captureCount = quietCount = ss->moveCount = 0;
    bestValue                                             = -VALUE_INFINITE;
    maxValue                                              = VALUE_INFINITE;

    // Le thread principal surveille l'horloge pour tout le monde
    if (thisThread->is_main() && (thisThread->nodes.load(std::memory_order_relaxed) & 1023) == 0)
        check_time();

    if (PvNode && thisThread->selDepth < ss->ply + 1)
        thisThread->selDepth = ss->ply + 1;

    if (!rootNode)
    {
        if (Threads.stop.load(std::memory_order_relaxed) || pos.is_draw(ss->ply)
            || ss->ply >= MAX_PLY)
            return (ss->ply >= MAX_PLY && !ss->inCheck) ? Eval::evaluate(pos) : VALUE_DRAW;

        // Elagage par distance au mat : inutile de chercher un mat plus long
        // que celui qu'on tient deja.
        alpha = std::max(mated_in(ss->ply), alpha);
        beta  = std::min(mate_in(ss->ply + 1), beta);
        if (alpha >= beta)
            return alpha;

    }

    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    bestMove              = Move::none();
    (ss + 2)->killers[0] = (ss + 2)->killers[1] = Move::none();
    (ss + 2)->cutoffCnt                          = 0;
    ss->statScore                                = 0;
    const int priorReduction                     = ss->reduction;
    ss->reduction                                = 0;

    const Square prevSq =
      ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq() : SQ_NONE;

    // ------------------------------------------------- table de transposition
    excludedMove = ss->excludedMove;
    posKey       = pos.key();

    TTEntry*    tte   = TT.probe(posKey, ss->ttHit);
    const bool  ttHit = ss->ttHit;
    const Value ttValue =
      ttHit ? value_from_tt(tte->value(), ss->ply, pos.rule50_count()) : VALUE_NONE;
    const Move ttMove = rootNode ? thisThread->rootMoves[0].pv[0]
                      : ttHit    ? tte->move()
                                 : Move::none();
    ttCapture         = ttMove && pos.capture_stage(ttMove);

    if (!excludedMove)
        ss->ttPv = PvNode || (ttHit && tte->is_pv());

    // La coupure de TT ne s'applique ni aux noeuds PV ni pendant une
    // verification singuliere : la sonde y porterait sur la meme cle et
    // rendrait le score obtenu avec le coup qu'on vient justement d'exclure.
    if (!PvNode && !excludedMove && ttHit && tte->depth() > depth - (ttValue <= beta)
        && ttValue != VALUE_NONE
        && (tte->bound() & (ttValue >= beta ? BOUND_LOWER : BOUND_UPPER))
        // TtCutRefine : a faible profondeur, on ne coupe que si le sens de la
        // coupure correspond au type de noeud attendu. Stockfish.
        && (!TtCutRefine || cutNode == (ttValue >= beta) || depth > 4))
    {
        if (ttMove && ttValue >= beta)
        {
            if (!ttCapture)
                update_quiet_stats(pos, ss, thisThread, ttMove,
                                   TtCutHistSf ? 131 * depth : stat_bonus(depth));

            // Le coup precedent a permis cette coupure : il etait mauvais.
            // TtCutHistSf : malus fixe, pour les 4 premiers coups (Stockfish).
            if (prevSq != SQ_NONE && (ss - 1)->moveCount <= (TtCutHistSf ? 4 : 2) && !priorCapture)
                update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq,
                                              TtCutHistSf ? -2210 : -stat_malus(depth + 1));
        }

        // Sous la regle des 50 coups la position n'est plus vraiment la meme
        if (pos.rule50_count() < 90)
        {
            // TtCutRefine, en profondeur : la position atteinte par le coup de
            // table doit confirmer la coupure. Parade partielle au probleme
            // d'interaction entre le graphe et l'historique. Stockfish.
            if (TtCutRefine && depth >= 7 && ttMove && pos.pseudo_legal(ttMove)
                && pos.legal(ttMove) && std::abs(ttValue) < VALUE_TB_WIN_IN_MAX_PLY)
            {
                pos.do_move(ttMove, st, pos.gives_check(ttMove));
                bool           nextHit;
                const TTEntry* next  = TT.probe(pos.key(), nextHit);
                const Value    nextV = nextHit ? next->value() : VALUE_NONE;
                pos.undo_move(ttMove);
                if (nextV == VALUE_NONE || (ttValue >= beta) == (-nextV >= beta))
                    return ttValue;
            }
            else
                return ttValue;
        }
    }

    // ------------------------------------------------ tables de finale
    // Une sonde reussie rend le resultat EXACT : la recherche s'arrete la, et
    // n'explore pas un sous-arbre dont la reponse est deja connue. Le score
    // est ecrit en table de transposition a une profondeur volontairement
    // haute -- il vaut mieux que n'importe quelle recherche.
    if (!rootNode && !excludedMove && Syzygy::max_pieces()
        && popcount(pos.pieces()) <= std::min(Syzygy::max_pieces(),
                                              int(Options["SyzygyProbeLimit"]))
        && depth >= int(Options["SyzygyProbeDepth"]))
    {
        Value tbValue;
        if (Syzygy::probe_wdl(pos, ss->ply, bool(Options["Syzygy50MoveRule"]), &tbValue))
        {
            thisThread->tbHits.fetch_add(1, std::memory_order_relaxed);

            tte->save(posKey, value_to_tt(tbValue, ss->ply), ss->ttPv, BOUND_EXACT,
                      std::min(depth + 6, MAX_PLY - 1), Move::none(), VALUE_NONE,
                      TT.generation());
            return tbValue;
        }
    }

    // -------------------------------------------------- evaluation statique
    if (ss->inCheck)
    {
        ss->staticEval = eval = VALUE_NONE;
        improving             = false;
        goto moves_loop;
    }
    else if (excludedMove)
        eval = ss->staticEval;

    else if (ttHit && tte->eval() != VALUE_NONE)
    {
        unadjustedStaticEval = tte->eval();
        ss->staticEval = eval = corrected_eval(unadjustedStaticEval, thisThread, pos, ss);

        if (ttValue != VALUE_NONE
            && (tte->bound() & (ttValue > eval ? BOUND_LOWER : BOUND_UPPER)))
            eval = ttValue;
    }
    else
    {
        unadjustedStaticEval  = Eval::evaluate(pos);
        ss->staticEval = eval = corrected_eval(unadjustedStaticEval, thisThread, pos, ss);

        // On enregistre l'evaluation meme sans score : la prochaine visite
        // n'aura pas a la recalculer.
        if (!excludedMove)
            tte->save(posKey, VALUE_NONE, ss->ttPv, BOUND_NONE, DEPTH_NONE, Move::none(),
                      unadjustedStaticEval, TT.generation());
    }

    // EvalDiffHist : l'evaluation statique a-t-elle beaucoup bouge depuis le
    // pere ? Le coup tranquille de l'adversaire qui y a mene est recompense ou
    // puni d'autant, dans l'historique principal et celui des structures de
    // pions. Stockfish. Inerte tant que EvalDiffHist vaut 0.
    if (EvalDiffHist && ((ss - 1)->currentMove).is_ok() && !(ss - 1)->inCheck && !priorCapture
        && (ss - 1)->staticEval != VALUE_NONE && ss->staticEval != VALUE_NONE)
    {
        const int evalDiff =
          std::clamp(-int((ss - 1)->staticEval + ss->staticEval), -189, 194) + 60;
        thisThread->mainHistory[~us][hist_index((ss - 1)->currentMove, (ss - 1)->threats)]
          << evalDiff * 11;
        if (!ttHit && type_of(pos.piece_on(prevSq)) != PAWN
            && ((ss - 1)->currentMove).type_of() != PROMOTION)
            thisThread->pawnHistory[pawn_structure_index(pos.pawn_key())][pos.piece_on(prevSq)]
                                   [prevSq]
              << evalDiff * 13;
    }

    // La position s'ameliore-t-elle par rapport a deux plis plus tot, quand
    // c'etait deja notre trait ? Un noeud en echec n'a pas d'evaluation
    // exploitable : on remonte alors de deux plis de plus.
    improving = (ss - 2)->staticEval != VALUE_NONE   ? ss->staticEval > (ss - 2)->staticEval
              : (ss - 4)->staticEval != VALUE_NONE   ? ss->staticEval > (ss - 4)->staticEval
                                                     : true;

    // Reduction a posteriori : le pere a reduit ce noeud sans savoir ce qu'on
    // y trouverait. Maintenant qu'on a l'evaluation, on corrige son pari --
    // un pli rendu s'il avait beaucoup reduit sans que l'adversaire empire, un
    // pli retire s'il avait peu reduit et que la position est deja confortable.
    // Repris de Stockfish. Inerte tant que Hindsight vaut 0.
    if (Hindsight && (ss - 1)->staticEval != VALUE_NONE)
    {
        const bool opponentWorsening = ss->staticEval > -(ss - 1)->staticEval;

        if (!PvNode && priorReduction >= 3 && !opponentWorsening)
            depth++;
        if (priorReduction >= 2 && depth >= 2 && ss->staticEval + (ss - 1)->staticEval > 166)
            depth--;
    }

    // --------------------------------------------------------- razoring
    if (!PvNode && depth <= RazorMaxDepth
        && eval + RazorMargin * depth < alpha)
    {
        // Contrairement au RFP, ce n'est pas une coupure seche : on verifie
        // par une quiescence avant de rendre.
        value = qsearch<NonPV>(pos, ss, alpha - 1, alpha, 0);
        if (value < alpha)
            return value;
    }

    // ---------------------------------------- reverse futility (RFP)
    // RfpOppW : marge reduite quand l'evaluation adverse s'est degradee au
    // pli precedent (opponentWorsening de Stockfish). Inerte a 0.
    if (!ss->ttPv && depth <= RfpMaxDepth
        && eval
               - (RfpOwn ? Value((std::min(45 + 4 * int(depth), 85) - 20 * !ttHit) * depth
                                 - 2789 * improving * (std::min(45 + 4 * int(depth), 85) - 20 * !ttHit)
                                     / 1024)
                         : futility_margin(depth, improving))
               + RfpOppW * ((ss - 1)->staticEval != VALUE_NONE
                                     && ss->staticEval > -(ss - 1)->staticEval)
               - (ss - 1)->statScore / 300 >= beta
        && eval >= beta && eval < VALUE_TB_WIN_IN_MAX_PLY && (!ttMove || ttCapture))
        return beta > VALUE_TB_LOSS_IN_MAX_PLY ? (eval + beta) / 2 : eval;

    // ------------------------------------------------------ coup nul (NMP)
    if (!PvNode && (ss - 1)->currentMove != Move::null() && !excludedMove
        && depth >= NmpMinDepth && eval >= beta && ss->staticEval >= beta
        && pos.non_pawn_material(us) && ss->ply >= thisThread->nmpMinPly
        && beta > VALUE_TB_LOSS_IN_MAX_PLY)
    {
        const Depth R = std::min(std::max(0, int(eval - beta)) / NmpEvalDiv, NmpMaxEvalR)
                      + depth / NmpDepthDiv + NmpBaseReduction;

        ss->currentMove         = Move::null();
        ss->continuationHistory = &thisThread->continuationHistory[0][0][NO_PIECE][SQ_A1];
        ss->continuationCorrectionHistory =
          &thisThread->continuationCorrectionHistory[NO_PIECE][SQ_A1];

        pos.do_null_move(st);
        Value nullValue = -search<NonPV>(pos, ss + 1, -beta, -beta + 1, depth - R, !cutNode);
        pos.undo_null_move();

        if (nullValue >= beta && nullValue < VALUE_TB_WIN_IN_MAX_PLY)
        {
            // Une recherche de verification, sans coup nul, evite les
            // catastrophes de zugzwang en finale.
            if (thisThread->nmpMinPly || depth < 14)
                return nullValue;

            assert(!thisThread->nmpMinPly);

            thisThread->nmpMinPly = ss->ply + 3 * (depth - R) / 4;

            const Value v = search<NonPV>(pos, ss, beta - 1, beta, depth - R, false);

            thisThread->nmpMinPly = 0;

            if (v >= beta)
                return nullValue;
        }
    }

    // ImprovingBeta : une position dont l'eval statique bat deja beta est
    // traitee comme en progres pour la suite (Stockfish).
    if (ImprovingBeta && ss->staticEval >= beta)
        improving = true;

    // ----------------------------------------------------------- ProbCut
    // ProbcutSf : marge reduite quand la position progresse, et profondeur
    // de verification qui en depend (Stockfish).
    probCutBeta = beta + ProbcutMargin - (ProbcutSf ? 64 * improving : 0);

    // !excludedMove est indispensable : pendant une verification singuliere,
    // ProbCut ecrirait sous la VRAIE cle un score obtenu en excluant un coup.
    if (!PvNode && !excludedMove && depth >= ProbcutMinDepth
        && std::abs(beta) < VALUE_TB_WIN_IN_MAX_PLY
        && !(ttHit && tte->depth() >= depth - 3 && ttValue != VALUE_NONE
             && ttValue < probCutBeta))
    {
        assert(probCutBeta < VALUE_INFINITE && probCutBeta > beta);

        MovePicker mp(pos, ttMove, probCutBeta - ss->staticEval, &thisThread->captureHistory);

        while ((move = mp.next_move()) != Move::none())
        {
            if (move == excludedMove || !pos.legal(move))
                continue;

            assert(pos.capture_stage(move));

            ss->currentMove = move;
            ss->continuationHistory =
              &thisThread->continuationHistory[ss->inCheck][true][pos.moved_piece(move)]
                                              [move.to_sq()];
            ss->continuationCorrectionHistory =
              &thisThread->continuationCorrectionHistory[pos.moved_piece(move)][move.to_sq()];

            thisThread->nodes.fetch_add(1, std::memory_order_relaxed);
            pos.do_move(move, st);

            // Quiescence d'abord : si elle ne confirme pas, inutile de
            // lancer la recherche courte.
            value = -qsearch<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1, 0);

            const Depth pcDepth = ProbcutSf ? depth - (improving ? 5 : 3) : depth - 4;
            if (value >= probCutBeta && pcDepth > 0)
                value = -search<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1,
                                       pcDepth, !cutNode);

            pos.undo_move(move);

            if (value >= probCutBeta)
            {
                tte->save(posKey, value_to_tt(value, ss->ply), ss->ttPv, BOUND_LOWER,
                          ProbcutSf ? pcDepth + 1 : depth - 3, move, unadjustedStaticEval,
                          TT.generation());
                return std::abs(value) < VALUE_TB_WIN_IN_MAX_PLY ? value - (probCutBeta - beta)
                                                                 : value;
            }
        }
    }

    // ------------------------------- IIR : pas de coup de TT, donc mauvais tri
    if (IirSf)
    {
        // IirSf : un seul pli, aux noeuds PV et CUT (Stockfish)
        if ((PvNode || cutNode) && depth >= 6 && !ttMove)
            depth -= 1;
    }
    else
    {
        if (PvNode && !ttMove && depth >= IirMinDepth)
            depth -= 2;

        if (cutNode && depth >= IirMinDepth + 3 && !ttMove)
            depth -= 2;
    }

    if (depth <= 0)
        return qsearch<PvNode ? PV : NonPV>(pos, ss, alpha, beta, 0);

moves_loop:

    // SmallProbcut : une borne basse de table, presque assez profonde et tres
    // au-dessus de beta, suffit a couper (Stockfish, « small ProbCut »).
    if (SmallProbcut && !rootNode && !excludedMove && ttHit && ttValue != VALUE_NONE
        && (tte->bound() & BOUND_LOWER) && tte->depth() >= depth - 4
        && ttValue >= beta + SmallProbcut && std::abs(beta) < VALUE_TB_WIN_IN_MAX_PLY
        && std::abs(ttValue) < VALUE_TB_WIN_IN_MAX_PLY)
        return beta + SmallProbcut;

    const PieceToHistory* contHist[] = {
      (ss - 1)->continuationHistory, (ss - 2)->continuationHistory,
      (ss - 3)->continuationHistory, (ss - 4)->continuationHistory,
      (ss - 5)->continuationHistory, (ss - 6)->continuationHistory};

    const Move counterMove =
      prevSq != SQ_NONE ? Move(thisThread->counterMoves[pos.piece_on(prevSq)][prevSq])
                        : Move::none();

    MovePicker mp(pos, ttMove, depth, &thisThread->mainHistory, &thisThread->captureHistory,
                  contHist, &thisThread->pawnHistory, ss->killers, counterMove,
                  &thisThread->lowPlyHistory, ss->ply);

    value          = bestValue;
    bool moveCountPruning = false;

    // Menaces de la position du noeud, pour l'historique principal : calculees
    // une fois, avant que do_move() ne change la position.
    const Bitboard nodeThreats = threat_map(pos);
    ss->threats                = nodeThreats;  // relu par les fils (coup du pere)

    while ((move = mp.next_move(moveCountPruning)) != Move::none())
    {
        assert(move.is_ok());

        if (move == excludedMove)
            continue;

        if (!pos.legal(move))
            continue;

        // A la racine, on ne joue que les coups demandes par l'interface
        if (rootNode
            && !std::count(thisThread->rootMoves.begin(), thisThread->rootMoves.end(), move))
            continue;

        ss->moveCount = ++moveCount;

        if (PvNode)
            (ss + 1)->pv = nullptr;

        extension  = 0;
        capture    = pos.capture_stage(move);
        givesCheck = pos.gives_check(move);
        newDepth   = depth - 1;

        const Piece movedPiece = pos.moved_piece(move);

        // ------------------------------------------------- elagages en boucle
        if (!rootNode && pos.non_pawn_material(us) && bestValue > VALUE_TB_LOSS_IN_MAX_PLY)
        {
            // LMP : au-dela d'un certain rang on ne regarde plus les
            // tranquilles. La table doit rester serree : trop permissive, le
            // mecanisme ne coupe simplement jamais.
            if (!moveCountPruning)
                moveCountPruning =
                  moveCount >= (LmpBase + LmpDepthFactor * depth * depth) / (2 - improving);

            const Depth lmrDepth = std::max(0, newDepth - reduction(depth, moveCount));

            if (capture || givesCheck)
            {
                // CapFutility : meme une prise ne remonte pas jusqu'a alpha
                // (Stockfish ; nos valeurs de pieces sont moitie des siennes).
                if (CapFutility && capture && !givesCheck && !ss->inCheck && lmrDepth < 8)
                {
                    const Piece cp = pos.piece_on(move.to_sq());
                    const int   ch = thisThread->captureHistory[movedPiece][move.to_sq()][type_of(cp)];
                    if (ss->staticEval + 234 + 247 * lmrDepth + 2 * PieceValue[cp] + 134 * ch / 1024
                        <= alpha)
                        continue;
                }

                // SEE sur les coups tactiques : la marge est negative, on
                // tolere de perdre d'autant plus qu'on est profond.
                if (!pos.see_ge(move, -SeeCaptureMargin * depth))
                    continue;
            }
            else
            {
                const int history = thisThread->mainHistory[us][hist_index(move, nodeThreats)]
                                  + (*contHist[0])[movedPiece][move.to_sq()]
                                  + (*contHist[1])[movedPiece][move.to_sq()];

                // Un coup dont l'historique est tres negatif a echoue partout
                // ailleurs dans l'arbre.
                if (history < -HistPruneMargin * depth && depth <= HistPruneMaxDepth)
                    continue;

                // Futility : meme en gagnant la marge on reste sous alpha
                if (!ss->inCheck && lmrDepth <= FutilityMaxDepth)
                {
                    const Value futVal =
                      Value(ss->staticEval + FutilityMargin * (lmrDepth + 1) + history / 200);
                    if (futVal <= alpha)
                    {
                        // FutValue : le score rendu tient compte du coup elague
                        if (FutValue && bestValue <= futVal
                            && std::abs(bestValue) < VALUE_TB_WIN_IN_MAX_PLY
                            && futVal < VALUE_TB_WIN_IN_MAX_PLY)
                            bestValue = futVal;
                        continue;
                    }
                }

                if (!pos.see_ge(move, -SeeQuietMargin * lmrDepth * lmrDepth))
                    continue;
            }
        }

        // ------------------------------------------------------- extensions
        //
        // Extension singuliere : si le coup de la table est le seul a tenir,
        // il merite d'etre creuse. On relance a profondeur reduite en
        // excluant ce coup ; si tous les autres echouent, extension.
        if (!rootNode && depth >= SingularMinDepth && move == ttMove && !excludedMove
            && std::abs(ttValue) < VALUE_TB_WIN_IN_MAX_PLY && (tte->bound() & BOUND_LOWER)
            && tte->depth() >= depth - 3)
        {
            const Value singularBeta  = ttValue - (SingularBetaFactor * depth) / 16;
            const Depth singularDepth = (depth - 1) / 2;

            ss->excludedMove = move;
            value = search<NonPV>(pos, ss, singularBeta - 1, singularBeta, singularDepth,
                                  cutNode);
            ss->excludedMove = Move::none();

            if (value < singularBeta)
            {
                extension = 1;

                // Nettement singulier : on creuse encore
                if (!PvNode && value < singularBeta - SingularExtraMargin)
                    extension = 2;
            }

            // Le coup de TT n'est pas singulier et il bat deja beta : la
            // position entiere coupe, inutile de la fouiller. MultiCutSf : on
            // rend le score trouve des que celui-ci bat beta (Stockfish).
            else if (MultiCutSf ? value >= beta && std::abs(value) < VALUE_TB_WIN_IN_MAX_PLY
                                : singularBeta >= beta)
            {
                if (MultiCutCorr && !ss->inCheck && value > ss->staticEval)
                    update_correction_history(
                      pos, ss, thisThread,
                      std::clamp(int(value - ss->staticEval) * 664 / 1024,
                                 -CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4));
                return MultiCutSf ? value : singularBeta;
            }

            else if (ttValue >= beta)
                extension = NegExtSf ? -3 : -2;

            else if (cutNode)
                extension = NegExtSf ? -3 : -1;
        }

        // Extension d'echec, sous condition : un echec qui perd du materiel
        // ne merite pas un pli de plus. Sans ce filtre, la profondeur ne
        // decroit jamais dans une sequence d'echecs.
        else if (givesCheck && depth > 8 && pos.see_ge(move))
            extension = 1;

        newDepth += extension;

        ss->currentMove = move;
        ss->continuationHistory =
          &thisThread->continuationHistory[ss->inCheck][capture][movedPiece][move.to_sq()];
        ss->continuationCorrectionHistory =
          &thisThread->continuationCorrectionHistory[movedPiece][move.to_sq()];

        // La ligne de cache de la future sonde est demandee maintenant : elle
        // arrivera pendant que do_move() travaille.
        prefetch(TT.first_entry(pos.key_after(move)));

        const std::uint64_t nodesBefore =
          rootNode ? thisThread->nodes.load(std::memory_order_relaxed) : 0;
        thisThread->nodes.fetch_add(1, std::memory_order_relaxed);
        pos.do_move(move, st, givesCheck);

        // --------------------------------------------------------- LMR
        Depth r = reduction(depth, moveCount);
        // LmrFrac : la meme reduction, comptee en 1/1024 de pli ; les
        // retouches de Stockfish (fractions de pli) n'existent qu'ainsi.
        int rf = LmrFrac ? reduction_f(depth, moveCount) : 0;

        if (ss->ttPv)
            r -= 1;
        if (PvNode)
            r -= 1;
        if (cutNode)
            r += 2;
        if (ttCapture && !capture)
            r += 1;
        if (!improving)
            r += 1;
        if (LmrFrac && LmrCutoffSf)
        {
            if ((ss + 1)->cutoffCnt > 1)
                rf += 264 + 1095 * ((ss + 1)->cutoffCnt > 2) + 1138 * (!PvNode && !cutNode);
        }
        else if ((ss + 1)->cutoffCnt > 3)
            r += 1;

        if (!capture)
        {
            ss->statScore = 2 * thisThread->mainHistory[us][hist_index(move, nodeThreats)]
                          + (*contHist[0])[movedPiece][move.to_sq()]
                          + (*contHist[1])[movedPiece][move.to_sq()]
                          + (*contHist[3])[movedPiece][move.to_sq()] - 3800;

            if (LmrFrac)
                rf -= int(std::int64_t(ss->statScore) * 1024 / LmrHistDiv);
            else
                r -= ss->statScore / LmrHistDiv;
        }
        // Une prise a elle aussi un historique : sans lui, bonnes et mauvaises
        // prises sont reduites pareil. Valeur de la piece prise plus historique
        // des prises, repris de Stockfish. Inerte tant que CapStatMul vaut 0.
        else if (CapStatMul)
        {
            const Piece captured = pos.captured_piece();
            ss->statScore = CapStatMul * int(PieceValue[captured]) / 128
                          + thisThread->captureHistory[movedPiece][move.to_sq()][type_of(captured)];

            if (LmrFrac)
                rf -= int(std::int64_t(ss->statScore) * 1024 / LmrHistDiv);
            else
                r -= ss->statScore / LmrHistDiv;
        }
        else
            ss->statScore = 0;

        if (LmrFrac)
        {
            // Les retouches entieres faites plus haut sur r passent dans rf
            rf += 1024 * (r - reduction(depth, moveCount));

            // LmrAlphaEval : un tranquille joue loin sous alpha est reduit plus
            if (LmrAlphaEval && !capture && !ss->inCheck && std::abs(alpha) < VALUE_TB_WIN_IN_MAX_PLY)
                rf += 3 * std::clamp(int(alpha - ss->staticEval), -64, 96);

            // LmrAllScale : reductions grossies aux noeuds ALL attendus
            if (LmrAllScale && !PvNode && !cutNode)
                rf += rf * 276 / (256 * depth + 268);

            r = rf / 1024;
        }

        if (depth >= 2 && moveCount > 1 + rootNode)
        {
            // Fenetre nulle a profondeur reduite. Si le coup bat alpha malgre
            // la reduction, il faut le rechercher pour de bon.
            const Depth d = std::clamp(newDepth - r, 1, newDepth + 1);

            (ss + 1)->reduction = newDepth - d;
            value               = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, d, true);
            (ss + 1)->reduction = 0;

            if (value > alpha && d < newDepth)
            {
                // La recherche reduite a battu alpha : de combien ? Largement,
                // et le coup merite meme plus que sa profondeur pleine ; de
                // justesse, et une profondeur de moins suffit. Repris de
                // Stockfish. Inerte tant que les deux marges valent 0.
                const bool deeper =
                  LmrDeeperMargin && value > bestValue + LmrDeeperMargin + 2 * newDepth;
                const bool shallower = LmrShallowerMargin && value < bestValue + LmrShallowerMargin;
                newDepth += deeper - shallower;

                if (newDepth > d)
                    value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);

                // PostLmrSf : bonus fixe pour tout coup dont la LMR a battu
                // alpha, prises comprises (Stockfish).
                if (PostLmrSf)
                    update_continuation_histories(ss, movedPiece, move.to_sq(), 1334);
                else if (!capture)
                {
                    const int bonus = value <= alpha   ? -stat_malus(newDepth)
                                    : value >= beta    ? stat_bonus(newDepth)
                                                       : 0;
                    update_continuation_histories(ss, movedPiece, move.to_sq(), bonus);
                }
            }
        }
        else if (!PvNode || moveCount > 1)
        {
            value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);
        }

        // Le premier coup d'un noeud PV, ou un coup qui a battu alpha en
        // fenetre nulle, se cherche en fenetre pleine.
        if (PvNode && (moveCount == 1 || value > alpha))
        {
            (ss + 1)->pv    = pv;
            (ss + 1)->pv[0] = Move::none();

            value = -search<PV>(pos, ss + 1, -beta, -alpha, newDepth, false);
        }

        pos.undo_move(move);

        assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

        if (Threads.stop.load(std::memory_order_relaxed))
            return VALUE_ZERO;

        // ------------------------------------------------ mise a jour du best
        if (rootNode)
        {
            RootMove& rm =
              *std::find(thisThread->rootMoves.begin(), thisThread->rootMoves.end(), move);

            // Noeuds depenses sous ce coup : quand le meilleur en a absorbe
            // l'essentiel, le choix est fait et on peut jouer plus tot.
            rm.effort += thisThread->nodes.load(std::memory_order_relaxed) - nodesBefore;

            rm.averageScore = rm.averageScore == -VALUE_INFINITE
                              ? value
                              : Value((2 * value + rm.averageScore) / 3);

            if (moveCount == 1 || value > alpha)
            {
                rm.score    = value;
                rm.selDepth = thisThread->selDepth;

                rm.pv.resize(1);
                assert((ss + 1)->pv);
                for (Move* m = (ss + 1)->pv; *m != Move::none(); ++m)
                    rm.pv.push_back(*m);
            }
            else
                // Un coup qui n'a pas battu alpha ne doit pas garder le score
                // de l'iteration precedente : il serait trie trop haut.
                rm.score = -VALUE_INFINITE;
        }

        if (value > bestValue)
        {
            bestValue = value;

            if (value > alpha)
            {
                bestMove = move;

                if (PvNode && !rootNode)
                    update_pv(ss->pv, move, (ss + 1)->pv);

                if (value >= beta)
                {
                    ss->cutoffCnt += 1 + !ttMove;
                    PROF_COUNT(FH_TOTAL, 1);
                    PROF_COUNT(FH_FIRST, moveCount == 1);
                    PROF_COUNT(FH_SUMIDX, moveCount);
                    PROF_COUNT(FH_TT, bool(ttMove));
                    PROF_COUNT(FH_TT_FIRST, ttMove && moveCount == 1);
                    PROF_COUNT(FH_QUIET, !capture);
                    PROF_COUNT(FH_QUIET_SUMIDX, capture ? 0 : moveCount);
                    break;
                }

                alpha = value;
            }
        }

        // On garde la trace des coups essayes en vain, pour les penaliser
        if (move != bestMove)
        {
            if (capture && captureCount < 32)
                capturesSearched[captureCount++] = move;
            else if (!capture && quietCount < 64)
                quietsSearched[quietCount++] = move;
        }
    }

    assert(moveCount || !ss->inCheck || excludedMove || !MoveList<LEGAL>(pos).size());

    // Aucun coup legal : mat si on est en echec, pat sinon.
    if (!moveCount)
        bestValue = excludedMove ? alpha : ss->inCheck ? mated_in(ss->ply) : VALUE_DRAW;

    else if (bestMove)
        update_all_stats(pos, ss, thisThread, bestMove, prevSq, quietsSearched, quietCount,
                         capturesSearched, captureCount, depth, ttMove, PvNode);

    // Aucun coup n'a battu alpha alors que le precedent etait tranquille :
    // ce precedent etait donc bon, on le recompense.
    else if (!priorCapture && prevSq != SQ_NONE)
    {
        update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq,
                                      stat_bonus(depth) * (1 + (bestValue > beta)));
    }

    if (PvNode)
        bestValue = std::min(bestValue, maxValue);

    if (!excludedMove && !(rootNode && thisThread->rootMoves.size() > 1))
        tte->save(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv,
                  bestValue >= beta          ? BOUND_LOWER
                  : PvNode && bestMove       ? BOUND_EXACT
                                             : BOUND_UPPER,
                  depth, bestMove, unadjustedStaticEval, TT.generation());

    // ------------------------------------------- historique de correction
    if (!ss->inCheck && !(bestMove && pos.capture(bestMove))
        && !(bestValue >= beta && bestValue <= ss->staticEval)
        && !(!bestMove && bestValue >= ss->staticEval))
    {
        const int bonus = std::clamp(int(bestValue - ss->staticEval) * depth / 8,
                                     -CORRECTION_HISTORY_LIMIT / 4,
                                     CORRECTION_HISTORY_LIMIT / 4);

        update_correction_history(pos, ss, thisThread, bonus);
    }

    assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

    return bestValue;
}

}  // namespace
}  // namespace one
