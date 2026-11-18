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

// Recherche alpha-beta (reecrite pour Onechess le 15-11-2026).
//
// Les idees viennent de nombreux moteurs libres (Stockfish d'abord) et de la
// litterature ; chacune a ete testee chez nous au temps avant d'etre gardee
// (voir RECHERCHES et JOURNAL). Le code est le notre.

#include "search.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <thread>

#include "evaluate.h"
#include "misc.h"
#include "movegen.h"
#include "movepick.h"
#include "position.h"
#include "prof.h"
#include "syzygy.h"
#include "thread.h"
#include "timeman.h"
#include "tt.h"
#include "tune.h"
#include "uci.h"

namespace one {

#ifndef ONE_FIGE
namespace Tune {
#define ONE_DEFINIR(nom, defaut, mini, maxi) int nom = defaut;
TUNABLE_LIST(ONE_DEFINIR)
#undef ONE_DEFINIR
}  // namespace Tune
#endif

namespace Search {
LimitsType        Limits;
Value             RootTbScore = VALUE_NONE;
std::atomic<bool> Ponder{false}, StopOnPonderhit{false}, Active{false};
}  // namespace Search

using namespace Search;
using namespace Tune;

namespace {

enum Noeud { NON_PV, PV, RACINE };

// =================================================================== outils

// Reductions des coups tardifs : base + ln(profondeur) * ln(rang) / diviseur
// (reconstruite a chaque changement de constante, voir Search::init).
int Reductions[64][64];
Depth reduction(Depth d, int rang) { return Reductions[std::min(int(d), 63)][std::min(rang, 63)]; }

// Marge de l'elagage par l'evaluation : plus serree si la position progresse
Value marge_futilite(Depth d, bool progresse) { return Value(FutilityMargin * (d - progresse)); }

int bonus_stat(Depth d) { return std::min(StatBonusMul * d - StatBonusSub, StatBonusMax); }
int malus_stat(Depth d) { return std::min(StatMalusMul * d - StatMalusSub, StatMalusMax); }

// Les mats et gains prouves se rangent en table en distance depuis le noeud
// (et non depuis la racine), pour rester valables ailleurs dans l'arbre.
Value vers_table(Value v, int ply) {
    assert(v != VALUE_NONE);
    if (v >= VALUE_TB_WIN_IN_MAX_PLY)
        return v + ply;
    if (v <= VALUE_TB_LOSS_IN_MAX_PLY)
        return v - ply;
    return v;
}

// Relecture ; un mat ou un gain trop lointain pour la regle des 50 coups
// (compteur r50) redevient un simple « presque gain ».
Value depuis_table(Value v, int ply, int r50) {
    if (v == VALUE_NONE)
        return v;
    const int reste = 99 - r50;
    if (v >= VALUE_TB_WIN_IN_MAX_PLY)
    {
        const int distance = v >= VALUE_MATE_IN_MAX_PLY ? VALUE_MATE - v : VALUE_TB - v;
        return distance > reste ? VALUE_TB_WIN_IN_MAX_PLY - 1 : v - ply;
    }
    if (v <= VALUE_TB_LOSS_IN_MAX_PLY)
    {
        const int distance = v <= VALUE_MATED_IN_MAX_PLY ? VALUE_MATE + v : VALUE_TB + v;
        return distance > reste ? VALUE_TB_LOSS_IN_MAX_PLY + 1 : v + ply;
    }
    return v;
}

// Variante principale : le coup, puis celle du fils
// Tension tactique (idee d'Onechess, 17-11) : pieces hors pions attaquees,
// des deux cotes. Une position calme en a 0 ; une position de combat plusieurs.
int tension(const Position& pos) {
    int t = 0;
    for (Color c : {WHITE, BLACK})
    {
        Bitboard att = 0;
        att |= pos.attacks_by<PAWN>(~c) | pos.attacks_by<KNIGHT>(~c) | pos.attacks_by<BISHOP>(~c)
             | pos.attacks_by<ROOK>(~c) | pos.attacks_by<QUEEN>(~c) | pos.attacks_by<KING>(~c);
        t += popcount(att & pos.pieces(c) & ~pos.pieces(PAWN, KING));
    }
    return t;
}

void nouvelle_pv(Move* pv, Move m, const Move* fils) {
    *pv++ = m;
    if (fils)
        while (*fils)
            *pv++ = *fils++;
    *pv = Move::none();
}

// --------------------------------------------------- corrections d'evaluation
// Cles des pieces mineures (cavaliers, fous, rois) et majeures (tours, dames,
// rois), calculees a la volee : un melange par ensemble de cases.
Key melange(Bitboard b, Key graine) {
    Key x = b ^ graine;
    x     = (x ^ (x >> 31)) * 0x7fb5d329728ea185ULL;
    x     = (x ^ (x >> 27)) * 0x81dadef4bc2dd44dULL;
    return x ^ (x >> 33);
}
Key cle_mineures(const Position& p) {
    return melange(p.pieces(WHITE, KNIGHT), 0x11) ^ melange(p.pieces(BLACK, KNIGHT), 0x22)
         ^ melange(p.pieces(WHITE, BISHOP), 0x33) ^ melange(p.pieces(BLACK, BISHOP), 0x44)
         ^ melange(p.pieces(WHITE, KING), 0x55) ^ melange(p.pieces(BLACK, KING), 0x66);
}
Key cle_majeures(const Position& p) {
    return melange(p.pieces(WHITE, ROOK), 0x77) ^ melange(p.pieces(BLACK, ROOK), 0x88)
         ^ melange(p.pieces(WHITE, QUEEN), 0x99) ^ melange(p.pieces(BLACK, QUEEN), 0xaa)
         ^ melange(p.pieces(WHITE, KING), 0xbb) ^ melange(p.pieces(BLACK, KING), 0xcc);
}

// Terme quadratique : c * |c| (les petites corrections restent petites)
int carre_signe(int c) { return c * std::abs(c); }

// Evaluation corrigee par ce que la recherche a appris de positions voisines
// (meme structure de pions, memes pieces, meme paire de coups).
Value evaluation_corrigee(Value v, const Thread* f, const Position& pos, const Stack* ss) {
    PROF_SCOPE(CORR);
    const Color moi = pos.side_to_move();
    v += Value(CorrHistWeight * carre_signe(f->correctionHistory[moi][correction_index(pos.pawn_key())]) / CorrHistDiv);
    if (CorrNpWeight)
    {
        const int b = f->nonPawnCorrection[WHITE][moi][correction_index(pos.non_pawn_key(WHITE))];
        const int n = f->nonPawnCorrection[BLACK][moi][correction_index(pos.non_pawn_key(BLACK))];
        v += Value(CorrNpWeight * (carre_signe(b) + carre_signe(n)) / CorrHistDiv);
    }
    if (CorrMinorWeight)
        v += Value(CorrMinorWeight * carre_signe(f->minorCorrection[moi][correction_index(cle_mineures(pos))])
                   / CorrHistDiv);
    if (CorrMajorWeight)
        v += Value(CorrMajorWeight * carre_signe(f->majorCorrection[moi][correction_index(cle_majeures(pos))])
                   / CorrHistDiv);
    if (CorrContWeight)
        if (const Move m = (ss - 1)->coup; m.is_ok())
            v += Value(CorrContWeight * carre_signe((*(ss - 2)->suiteCorr)[pos.piece_on(m.to_sq())][m.to_sq()])
                       / CorrHistDiv);
    // R50Div (en test) : plus la regle des 50 coups approche, plus l'evaluation
    // se rapproche de la nulle (un avantage qu'on ne concretise pas s'evapore)
    if (R50Div)
        v = Value(int(v) * (R50Div - pos.rule50_count()) / R50Div);
    return std::clamp(v, Value(VALUE_TB_LOSS_IN_MAX_PLY + 1), Value(VALUE_TB_WIN_IN_MAX_PLY - 1));
}

// Ces tables sont lues juste apres le reseau : on les demande des l'entree du noeud
void precharger_corrections(const Position& pos, const Thread* f, const Stack* ss) {
    const Color moi = pos.side_to_move();
    prefetch(&f->correctionHistory[moi][correction_index(pos.pawn_key())]);
    prefetch(&f->nonPawnCorrection[WHITE][moi][correction_index(pos.non_pawn_key(WHITE))]);
    prefetch(&f->nonPawnCorrection[BLACK][moi][correction_index(pos.non_pawn_key(BLACK))]);
    if (const Move m = (ss - 1)->coup; m.is_ok())
        prefetch(&(*(ss - 2)->suiteCorr)[pos.piece_on(m.to_sq())][m.to_sq()]);
}

void apprendre_correction(const Position& pos, Stack* ss, Thread* f, int bonus) {
    const Color moi = pos.side_to_move();
    f->correctionHistory[moi][correction_index(pos.pawn_key())] << bonus;
    f->nonPawnCorrection[WHITE][moi][correction_index(pos.non_pawn_key(WHITE))] << bonus;
    f->nonPawnCorrection[BLACK][moi][correction_index(pos.non_pawn_key(BLACK))] << bonus;
    if (CorrMinorWeight)
        f->minorCorrection[moi][correction_index(cle_mineures(pos))] << bonus;
    if (CorrMajorWeight)
        f->majorCorrection[moi][correction_index(cle_majeures(pos))] << bonus;
    if (const Move m = (ss - 1)->coup; m.is_ok())
        (*(ss - 2)->suiteCorr)[pos.piece_on(m.to_sq())][m.to_sq()] << bonus;
}

// ------------------------------------------------------------- historiques
// Historiques de suite : le coup juge par rapport aux coups precedents (1, 2,
// 3, 4 et 6 demi-coups plus tot ; seulement 1 et 2 en echec). Le 3e compte
// pour un quart.
void suites(Stack* ss, Piece p, Square a, int bonus) {
    for (int recul : {1, 2, 3, 4, 6})
    {
        if (ss->enEchec && recul > 2)
            return;
        if ((ss - recul)->coup.is_ok())
            (*(ss - recul)->suite)[p][a] << (recul == 3 ? bonus / 4 : bonus);
    }
}

// Un coup tranquille a coupe : coups meurtriers, historiques, reponse
void recompenser_tranquille(const Position& pos, Stack* ss, Thread* f, Move m, int bonus, Bitboard menaces) {
    if (ss->meurtriers[0] != m)
    {
        ss->meurtriers[1] = ss->meurtriers[0];
        ss->meurtriers[0] = m;
    }
    const Piece p = pos.moved_piece(m);
    f->mainHistory[pos.side_to_move()][hist_index(m, menaces)] << bonus;
    if (ss->ply < LOW_PLY_HISTORY_SIZE)
        f->lowPlyHistory[ss->ply][m.from_to()] << bonus * 3 / 4;
    suites(ss, p, m.to_sq(), bonus);
    f->pawnHistory[pawn_structure_index(pos.pawn_key())][p][m.to_sq()] << bonus;
    if (const Move avant = (ss - 1)->coup; avant.is_ok())
        f->counterMoves[pos.piece_on(avant.to_sq())][avant.to_sq()] = m;
}

// Fin d'un noeud avec un meilleur coup : recompense pour lui, malus pour les
// coups essayes en vain avant lui (decroissant, MalusDecay), et pour le coup
// precedent s'il a ete refute d'emblee.
void apprendre_coupure(const Position& pos, Stack* ss, Thread* f, Move meilleur, Square avantCase,
                       const Move* tranquilles, int nbTranq, const Move* prises, int nbPrises, Depth d) {
    PROF_SCOPE(HISTUPD);
    int bonus = bonus_stat(d);
    const int malus = malus_stat(d);

    if (pos.capture_stage(meilleur))
        f->captureHistory[pos.moved_piece(meilleur)][meilleur.to_sq()][type_of(pos.piece_on(meilleur.to_sq()))] << bonus;
    else
    {
        recompenser_tranquille(pos, ss, f, meilleur, bonus, ss->menaces);
        int m = malus;
        for (int i = 0; i < nbTranq; ++i)
        {
            const Move  q = tranquilles[i];
            const Piece p = pos.moved_piece(q);
            f->mainHistory[pos.side_to_move()][hist_index(q, ss->menaces)] << -m;
            if (ss->ply < LOW_PLY_HISTORY_SIZE)
                f->lowPlyHistory[ss->ply][q.from_to()] << -m * 3 / 4;
            f->pawnHistory[pawn_structure_index(pos.pawn_key())][p][q.to_sq()] << -m;
            suites(ss, p, q.to_sq(), -m);
            m = m * MalusDecay / 1024;
        }
    }

    if (avantCase != SQ_NONE && (ss - 1)->nbCoups == 1 + (ss - 1)->ttTrouve && !pos.captured_piece())
        suites(ss - 1, pos.piece_on(avantCase), avantCase, -malus);

    for (int i = 0; i < nbPrises; ++i)
    {
        const Move c = prises[i];
        f->captureHistory[pos.moved_piece(c)][c.to_sq()][type_of(pos.piece_on(c.to_sq()))] << -malus;
    }
}

// ------------------------------------------------------------------ horloge
void verifier_temps() {
    if (Ponder)  // c'est le temps de l'adversaire
        return;
    const bool auTemps = Limits.use_time_management() || Limits.movetime;
    if (Limits.nodes && Threads.nodes_searched() >= Limits.nodes)
        Threads.stop = true;
    if (auTemps && Time.elapsed() >= Time.maximum())
        Threads.stop = true;
}

// Le fil principal relit l'horloge tous les 512 noeuds (recherche et quiescence)
void sonder_horloge(Thread* f) {
    if (!f->is_main() || --f->callsCnt > 0)
        return;
    f->callsCnt = Limits.nodes ? int(std::min<std::uint64_t>(512, Limits.nodes / 1024)) : 512;
    verifier_temps();
}

template<Noeud N>
Value quiescence(Position& pos, Stack* ss, Value alpha, Value beta, Depth d);
template<Noeud N>
Value chercher(Position& pos, Stack* ss, Value alpha, Value beta, Depth d, bool noeudCut);

// Position attendue apres notre coup et la reponse prevue (regle de surprise :
// si l'adversaire joue autre chose, on se donne plus de temps).
Key CleAttendue  = 0;
int DemiCoupAttendu = -1;

}  // namespace

void Search::init() {
    for (int d = 0; d < 64; ++d)
        for (int r = 0; r < 64; ++r)
            Reductions[d][r] =
              d && r ? int(LmrBase / 100.0 + std::log(double(d)) * std::log(double(r)) / (LmrDivisor / 100.0)) : 0;
}

// =================================================== approfondissement iteratif
void Thread::iterative_deepening() {
    PROF_SCOPE(SEARCH);
    Stack  pile[MAX_PLY + 10] = {};
    Stack* ss                 = pile + 7;  // 7 cases avant la racine : (ss - 7) lisible
    Move   pv[MAX_PLY + 1];

    for (int i = 1; i <= 7; ++i)
    {
        (ss - i)->suite        = &continuationHistory[0][0][NO_PIECE][SQ_A1];
        (ss - i)->suiteCorr    = &continuationCorrectionHistory[NO_PIECE][SQ_A1];
        (ss - i)->evalStatique = VALUE_NONE;
    }
    for (int i = 0; i <= MAX_PLY + 2; ++i)
        (ss + i)->ply = i;
    ss->pv = pv;

    lowPlyHistory.fill(0);  // il ne vaut que pour cette racine
    rootPos.set_thread(this);
    const bool surprise = is_main() && DemiCoupAttendu == rootPos.game_ply() && rootPos.key() != CleAttendue;

    if (rootMoves.empty())  // mat ou pat
    {
        rootMoves.emplace_back(Move::none());
        if (is_main())
        {
            if (!Limits.silent)
                std::cout << "info depth 0 score " << UCI::value(rootPos.checkers() ? -VALUE_MATE : VALUE_DRAW)
                          << "\nbestmove (none)" << std::endl;
            Active = false;
        }
        return;
    }

    Value       meilleur = -VALUE_INFINITE, alpha = -VALUE_INFINITE, beta = VALUE_INFINITE;
    Move        dernierCoup = Move::none();
    Depth       dernierChangement = 0;
    double      changements = 0.0;               // amortis de moitie a chaque iteration
    Value       scorePrecedent = -VALUE_INFINITE;
    const auto  lignes = std::min(std::size_t(int(Options["MultiPV"])), rootMoves.size());

    while (++rootDepth < MAX_PLY && !Threads.stop && !(Limits.depth && is_main() && rootDepth > Limits.depth))
    {
        for (RootMove& rm : rootMoves)
            rm.previousScore = rm.score;

        // MultiPV : chaque ligne exclut les coups des lignes precedentes
        for (pvIdx = 0; pvIdx < lignes && !Threads.stop; ++pvIdx)
        {
            selDepth = 0;

            // Fenetre d'aspiration autour du score moyen du coup
            const Value moyen = rootMoves[pvIdx].averageScore;
            Value       delta = 0;
            if (rootDepth >= AspirationMinDepth && moyen != -VALUE_INFINITE)
            {
                delta = Value(AspirationDelta + std::abs(moyen) / 10000);
                alpha = std::max(moyen - delta, -VALUE_INFINITE);
                beta  = std::min(moyen + delta, VALUE_INFINITE);
            }
            else
                alpha = -VALUE_INFINITE, beta = VALUE_INFINITE;

            for (int echecsHauts = 0;;)
            {
                // apres des echecs hauts repetes, un peu moins profond
                rootDelta = beta - alpha;
                meilleur  = chercher<RACINE>(rootPos, ss, alpha, beta, std::max(1, rootDepth - echecsHauts), false);
                std::stable_sort(rootMoves.begin() + pvIdx, rootMoves.end());
                if (Threads.stop)
                    break;
                if (meilleur <= alpha)  // on n'elargit que le bord qui a cede
                {
                    beta        = (alpha + beta) / 2;
                    alpha       = std::max(meilleur - delta, -VALUE_INFINITE);
                    echecsHauts = 0;
                }
                else if (meilleur >= beta)
                {
                    beta = std::min(meilleur + delta, VALUE_INFINITE);
                    ++echecsHauts;
                }
                else
                    break;
                delta += delta / 3;
            }
            std::stable_sort(rootMoves.begin(), rootMoves.begin() + pvIdx + 1);
        }
        pvIdx = 0;
        if (lignes > 1)
            meilleur = rootMoves[0].score;

        changements /= 2.0;
        if (!Threads.stop)
            completedDepth = rootDepth;

        // « go mate N » : mat en N coups au plus prouve
        if (Limits.mate && is_main() && !Threads.stop && rootMoves[0].score >= VALUE_MATE_IN_MAX_PLY
            && VALUE_MATE - rootMoves[0].score <= 2 * Limits.mate)
            Threads.stop = true;

        if (rootMoves[0].pv[0] != dernierCoup)
        {
            dernierCoup       = rootMoves[0].pv[0];
            dernierChangement = rootDepth;
            changements += 1.0;
        }

        if (is_main())  // pas de ligne vide (certaines interfaces la refusent)
            if (const std::string l = UCI::pv(*this, rootDepth); !l.empty())
                std::cout << l << std::endl;
        if (Threads.stop)
            break;

        // ------------------------------------------- faut-il s'arreter ?
        if (is_main() && Limits.use_time_management() && !Limits.movetime)
        {
            double f = (rootDepth == dernierChangement ? TmChangedPct : TmStablePct) / 100.0;
            if (TmInstMul)  // le nombre de changements de meilleur coup
                f *= 1.0 + TmInstMul / 1000.0 * changements;
            if (TmFallPct && scorePrecedent != -VALUE_INFINITE)  // un score qui baisse merite du temps
                f *= std::clamp(1.0 + TmFallPct / 1000.0 * double(scorePrecedent - meilleur), 0.8, 1.5);
            scorePrecedent = meilleur;
            if (surprise)
                f *= TmSurprisePct / 100.0;
            if (TmTension)  // option en test : plus de temps quand la position est tendue, moins quand elle est calme
                f *= std::clamp(1.0 + TmTension / 1000.0 * (tension(rootPos) - 1), 0.85, 1.25);
            const bool force = TmSingleMs && rootMoves.size() == 1
                            && Time.elapsed() >= std::min<TimePoint>(TmSingleMs, Time.optimum());
            if (TmEffortPct)  // le meilleur coup a absorbe l'essentiel des noeuds
            {
                const double part = double(rootMoves[0].effort)
                                  / double(std::max<std::uint64_t>(1, nodes.load(std::memory_order_relaxed)));
                f *= std::clamp(1.0 - TmEffortPct / 1000.0 * std::max(0.0, part - 0.5), 0.69, 1.0);
            }
            // pas de nouvelle iteration qu'on ne pourrait pas finir
            const double budget = double(Time.optimum()) * f * (TmNoStartPct / 100.0);
            if (force || Time.elapsed() > TimePoint(budget))
            {
                if (Ponder)
                    StopOnPonderhit = true;
                else
                {
                    Threads.stop = true;
                    break;
                }
            }
        }
        if (is_main() && Limits.movetime && Time.elapsed() >= Time.optimum())
        {
            if (Ponder)
                StopOnPonderhit = true;
            else
            {
                Threads.stop = true;
                break;
            }
        }
    }

    if (!is_main())
        return;

    // « go infinite » et le ponder ne rendent le coup qu'apres « stop » ou « ponderhit »
    while (!Threads.stop && (Limits.infinite || Ponder))
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Threads.stop = true;
    Threads.wait_for_helpers();

    if (Limits.silent)
    {
        Active = false;
        return;
    }

    const Thread*     choix = Threads.best_thread();
    const auto&       ligne = choix->rootMoves[0].pv;
    const bool        c960  = rootPos.is_chess960();

    // Ce qu'on attend au prochain coup (notre coup puis la reponse prevue)
    DemiCoupAttendu = -1;
    if (ligne.size() > 1)
    {
        StateInfo e1, e2;
        rootPos.do_move(ligne[0], e1);
        if (rootPos.pseudo_legal(ligne[1]) && rootPos.legal(ligne[1]))
        {
            rootPos.do_move(ligne[1], e2);
            CleAttendue     = rootPos.key();
            DemiCoupAttendu = rootPos.game_ply();
            rootPos.undo_move(ligne[1]);
        }
        rootPos.undo_move(ligne[0]);
    }

    std::cout << "bestmove " << UCI::move(ligne[0], c960);
    if (ligne.size() > 1)
        std::cout << " ponder " << UCI::move(ligne[1], c960);
    std::cout << std::endl;
    Active = false;
}

namespace {

// =================================================================== quiescence
// Seulement les prises (et les parades en echec), jusqu'a une position calme.
template<Noeud N>
Value quiescence(Position& pos, Stack* ss, Value alpha, Value beta, Depth d) {
    static_assert(N != RACINE);
    constexpr bool estPv = N == PV;
    assert(alpha >= -VALUE_INFINITE && alpha < beta && beta <= VALUE_INFINITE && d <= 0);

    Move      pv[MAX_PLY + 1];
    StateInfo etat;
    Thread*   f = pos.this_thread();

    if (estPv)
    {
        (ss + 1)->pv = pv;
        ss->pv[0]    = Move::none();
    }
    ss->enEchec = pos.checkers();
    sonder_horloge(f);
    const Value alphaDepart = alpha;

    if (pos.is_draw(ss->ply) || ss->ply >= MAX_PLY)
        return ss->ply >= MAX_PLY && !ss->enEchec ? Eval::evaluate(pos) : VALUE_DRAW;
    // L'arret doit etre entendu ici aussi (finales gagnees : arbres enormes)
    if (Threads.stop.load(std::memory_order_relaxed))
        return VALUE_ZERO;

    if (QsFin & 1)  // distance au mat
    {
        alpha = std::max(mated_in(ss->ply), alpha);
        beta  = std::min(mate_in(ss->ply + 1), beta);
        if (alpha >= beta)
            return alpha;
    }

    // Deux profondeurs en table : 0 en echec (ou au premier niveau), -1 ensuite
    const Depth profTable = ss->enEchec || d >= DEPTH_QS ? DEPTH_QS : DEPTH_QS - 1;

    const Key   cle = pos.key();
    bool        trouve;
    TTEntry*    fiche = TT.probe(cle, trouve);
    if (!ss->enEchec)
        precharger_corrections(pos, f, ss);
    ss->ttTrouve        = trouve;
    const Value vTable  = trouve ? depuis_table(fiche->value(), ss->ply, pos.rule50_count()) : VALUE_NONE;
    const Move  cTable  = trouve ? fiche->move() : Move::none();
    const bool  pvTable = trouve && fiche->is_pv();

    if (!estPv && trouve && fiche->depth() >= profTable && vTable != VALUE_NONE
        && (fiche->bound() & (vTable >= beta ? BOUND_LOWER : BOUND_UPPER)))
        return vTable;

    // Evaluation au repos (« stand pat ») ; c'est l'evaluation BRUTE qui va en
    // table (la corrigee se recorrigerait a chaque relecture).
    Value brute = VALUE_NONE, meilleure, baseFutilite;
    if (ss->enEchec)
    {
        ss->evalStatique = VALUE_NONE;
        meilleure = baseFutilite = -VALUE_INFINITE;
    }
    else
    {
        brute = trouve && fiche->eval() != VALUE_NONE ? fiche->eval() : Eval::evaluate(pos);
        ss->evalStatique = meilleure = evaluation_corrigee(brute, f, pos, ss);
        if (vTable != VALUE_NONE && (fiche->bound() & (vTable > meilleure ? BOUND_LOWER : BOUND_UPPER)))
            meilleure = vTable;
        if (meilleure >= beta)
        {
            if (!trouve)
                fiche->save(cle, vers_table(meilleure, ss->ply), false, BOUND_LOWER, DEPTH_ENTRY_TT + 1, Move::none(),
                            brute, TT.generation());
            return meilleure;
        }
        alpha        = std::max(alpha, meilleure);
        baseFutilite = meilleure + QsFutilityMargin;
    }

    const PieceToHistory* hs[] = {(ss - 1)->suite, (ss - 2)->suite, (ss - 3)->suite,
                                  (ss - 4)->suite, (ss - 5)->suite, (ss - 6)->suite};
    const Square avantCase = (ss - 1)->coup.is_ok() ? (ss - 1)->coup.to_sq() : SQ_NONE;

    MovePicker tri(pos, cTable, d, &f->mainHistory, &f->captureHistory, hs, &f->pawnHistory, ss->meurtriers,
                   Move::none(), &f->lowPlyHistory, ss->ply);
    Move meilleurCoup = Move::none();
    int  rang         = 0;

    for (Move m; (m = tri.next_move());)
    {
        if (!pos.legal(m))
            continue;
        const bool echec = pos.gives_check(m);
        const bool prise = pos.capture_stage(m);
        ++rang;

        if (meilleure > VALUE_TB_LOSS_IN_MAX_PLY && pos.non_pawn_material(pos.side_to_move()))
        {
            // Futilite : meme en gagnant la piece, loin d'alpha
            if (!echec && m.to_sq() != avantCase && baseFutilite > VALUE_TB_LOSS_IN_MAX_PLY
                && m.type_of() != PROMOTION)
            {
                if (rang > 2)
                    continue;
                const Value vFut = baseFutilite + PieceValue[pos.piece_on(m.to_sq())];
                if (vFut <= alpha)
                {
                    meilleure = std::max(meilleure, vFut);
                    continue;
                }
                if (baseFutilite <= alpha && !pos.see_ge(m, 1))  // rien a gagner, case defendue
                {
                    meilleure = std::max(meilleure, baseFutilite);
                    continue;
                }
            }
            if (!pos.see_ge(m, -QsDeltaMargin))  // echange perdant
                continue;
        }

        const Piece p = pos.moved_piece(m);
        ss->coup      = m;
        ss->suite     = &f->continuationHistory[ss->enEchec][prise][p][m.to_sq()];
        ss->suiteCorr = &f->continuationCorrectionHistory[p][m.to_sq()];
        prefetch(TT.first_entry(pos.key_after(m)));
        f->nodes.fetch_add(1, std::memory_order_relaxed);
        pos.do_move(m, etat, echec);
        const Value v = -quiescence<N>(pos, ss + 1, -beta, -alpha, d - 1);
        pos.undo_move(m);

        if (v <= meilleure)
            continue;
        meilleure = v;
        if (v <= alpha)
            continue;
        meilleurCoup = m;
        if (estPv)
            nouvelle_pv(ss->pv, m, (ss + 1)->pv);
        if (!estPv || v >= beta)
            break;  // coupure
        alpha = v;
    }

    if (ss->enEchec && meilleure == -VALUE_INFINITE)  // en echec sans parade : mat
        return mated_in(ss->ply);

    const Bound borne = meilleure >= beta ? BOUND_LOWER : estPv && meilleure > alphaDepart ? BOUND_EXACT : BOUND_UPPER;
    fiche->save(cle, vers_table(meilleure, ss->ply), pvTable, borne, profTable, meilleurCoup, brute, TT.generation());
    return meilleure;
}

// ============================================================ recherche principale
template<Noeud N>
Value chercher(Position& pos, Stack* ss, Value alpha, Value beta, Depth d, bool noeudCut) {
    constexpr bool estPv   = N != NON_PV;
    constexpr bool racine  = N == RACINE;

    if (d <= 0)
        return quiescence<estPv ? PV : NON_PV>(pos, ss, alpha, beta, 0);
    assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE && d < MAX_PLY);

    Move      pv[MAX_PLY + 1], prises[32], tranquilles[64];
    StateInfo etat;
    Thread*   f           = pos.this_thread();
    const Color moi       = pos.side_to_move();
    const bool prisePrec  = pos.captured_piece();
    int        nbPrises = 0, nbTranq = 0, rang = 0;

    ss->enEchec    = pos.checkers();
    ss->nbCoups    = 0;
    ss->extDoubles = (ss - 1)->extDoubles;
    Value meilleure = -VALUE_INFINITE;

    sonder_horloge(f);
    if (estPv && f->selDepth < ss->ply + 1)
        f->selDepth = ss->ply + 1;

    if (!racine)
    {
        if (Threads.stop.load(std::memory_order_relaxed) || pos.is_draw(ss->ply) || ss->ply >= MAX_PLY)
            return ss->ply >= MAX_PLY && !ss->enEchec ? Eval::evaluate(pos) : VALUE_DRAW;
        // inutile de chercher un mat plus long que celui qu'on tient
        alpha = std::max(mated_in(ss->ply), alpha);
        beta  = std::min(mate_in(ss->ply + 1), beta);
        if (alpha >= beta)
            return alpha;
    }

    Move meilleurCoup                     = Move::none();
    (ss + 2)->meurtriers[0]               = (ss + 2)->meurtriers[1] = Move::none();
    (ss + 2)->coupures                    = 0;
    ss->noteStat                          = 0;
    const int    reductionRecue           = ss->reduction;
    ss->reduction                         = 0;
    const Square avantCase                = (ss - 1)->coup.is_ok() ? (ss - 1)->coup.to_sq() : SQ_NONE;

    // ------------------------------------------------- table de transposition
    const Move  exclu   = ss->exclu;
    const Key   cle     = pos.key();
    TTEntry*    fiche   = TT.probe(cle, ss->ttTrouve);
    if (!ss->enEchec)
        precharger_corrections(pos, f, ss);
    const bool  trouve  = ss->ttTrouve;
    const Value vTable  = trouve ? depuis_table(fiche->value(), ss->ply, pos.rule50_count()) : VALUE_NONE;
    Move        cTable  = racine ? f->rootMoves[f->pvIdx].pv[0] : trouve ? fiche->move() : Move::none();
    bool        priseTable = cTable && pos.capture_stage(cTable);
    if (!exclu)
        ss->ttPv = estPv || (trouve && fiche->is_pv());

    // Coupure par la table (jamais en PV ni pendant une verification singuliere)
    if (!estPv && !exclu && trouve && fiche->depth() > d - (vTable <= beta) && vTable != VALUE_NONE
        && (fiche->bound() & (vTable >= beta ? BOUND_LOWER : BOUND_UPPER)))
    {
        if (cTable && vTable >= beta)
        {
            if (!priseTable)
                recompenser_tranquille(pos, ss, f, cTable, bonus_stat(d), threat_map(pos));
            // le coup precedent a ete refute tout de suite
            if (avantCase != SQ_NONE && (ss - 1)->nbCoups <= 2 && !prisePrec)
                suites(ss - 1, pos.piece_on(avantCase), avantCase, -malus_stat(d + 1));
        }
        if (pos.rule50_count() < 90)  // pres de la regle des 50 coups, on ne se fie plus
            return vTable;
    }

    // ------------------------------------------------ tables de finales
    if (!racine && !exclu && Syzygy::max_pieces()
        && popcount(pos.pieces()) <= std::min(Syzygy::max_pieces(), int(Options["SyzygyProbeLimit"]))
        && d >= int(Options["SyzygyProbeDepth"]))
    {
        Value vTb;
        if (Syzygy::probe_wdl(pos, ss->ply, bool(Options["Syzygy50MoveRule"]), &vTb))
        {
            f->tbHits.fetch_add(1, std::memory_order_relaxed);
            fiche->save(cle, vers_table(vTb, ss->ply), ss->ttPv, BOUND_EXACT, std::min(d + 6, MAX_PLY - 1),
                        Move::none(), VALUE_NONE, TT.generation());
            return vTb;  // resultat exact
        }
    }

    // ------------------------------------------------ evaluation statique
    Value brute = VALUE_NONE, eval = VALUE_NONE;
    bool  progresse = false;
    Value v;

    if (ss->enEchec)
    {
        ss->evalStatique = VALUE_NONE;
        ss->ecartCorr    = 0;
    }
    else
    {
        if (exclu)
            eval = ss->evalStatique;
        else
        {
            const bool dejaEvalue = trouve && fiche->eval() != VALUE_NONE;
            brute                 = dejaEvalue ? fiche->eval() : Eval::evaluate(pos);
            ss->evalStatique = eval = evaluation_corrigee(brute, f, pos, ss);
            ss->ecartCorr           = std::abs(int(ss->evalStatique - brute));
            if (dejaEvalue)
            {
                if (vTable != VALUE_NONE && (fiche->bound() & (vTable > eval ? BOUND_LOWER : BOUND_UPPER)))
                    eval = vTable;
            }
            else  // on garde l'evaluation pour la prochaine visite
                fiche->save(cle, VALUE_NONE, ss->ttPv, BOUND_NONE, DEPTH_NONE, Move::none(), brute, TT.generation());
        }

        // Le coup tranquille adverse a-t-il fait bouger l'evaluation ? Il est
        // recompense ou puni d'autant.
        if ((ss - 1)->coup.is_ok() && !(ss - 1)->enEchec && !prisePrec && (ss - 1)->evalStatique != VALUE_NONE
            && ss->evalStatique != VALUE_NONE)
        {
            const int ecart = std::clamp(-int((ss - 1)->evalStatique + ss->evalStatique), -EcartMin, EcartMax) + EcartDecal;
            f->mainHistory[~moi][hist_index((ss - 1)->coup, (ss - 1)->menaces)] << ecart * EcartPrincipal;
            if (!trouve && type_of(pos.piece_on(avantCase)) != PAWN && (ss - 1)->coup.type_of() != PROMOTION)
                f->pawnHistory[pawn_structure_index(pos.pawn_key())][pos.piece_on(avantCase)][avantCase]
                  << ecart * EcartPions;
        }

        // Progres par rapport a notre dernier trait (ou celui d'avant, s'il etait en echec)
        progresse = (ss - 2)->evalStatique != VALUE_NONE ? ss->evalStatique > (ss - 2)->evalStatique
                  : (ss - 4)->evalStatique != VALUE_NONE ? ss->evalStatique > (ss - 4)->evalStatique
                                                         : true;

        // Reduction a posteriori : on corrige le pari de reduction du pere
        if ((ss - 1)->evalStatique != VALUE_NONE)
        {
            const bool adversaireEmpire = ss->evalStatique > -(ss - 1)->evalStatique;
            if (!estPv && reductionRecue >= 3 && !adversaireEmpire)
                ++d;
            if (reductionRecue >= 2 && d >= 2 && ss->evalStatique + (ss - 1)->evalStatique > ReculMarge)
                --d;
        }

        // NmpTension / RfpTension (en test, idees d'Onechess) : nos pieces hors pions
        // attaquees ; une position ou l'on est menace se juge mal sans chercher
        const int attaquees = (NmpTension || RfpTension)
                              ? popcount(threat_map_complete(pos) & pos.pieces(moi) & ~pos.pieces(PAWN, KING))
                              : 0;

        // Rasage : tres loin sous alpha, une quiescence decide
        if (!estPv && d <= RazorMaxDepth && eval + RazorMargin * d < alpha)
        {
            v = quiescence<NON_PV>(pos, ss, alpha - 1, alpha, 0);
            if (v < alpha)
                return v;
        }

        // Elagage par l'evaluation (futilite inverse)
        if (!ss->ttPv && d <= RfpMaxDepth
            && eval - marge_futilite(d, progresse) - (ss - 1)->noteStat / RfpStatDiv - RfpTension * attaquees >= beta
            && eval >= beta && eval < VALUE_TB_WIN_IN_MAX_PLY && (!cTable || priseTable))
            return beta > VALUE_TB_LOSS_IN_MAX_PLY ? (eval + beta) / 2 : eval;

        // Coup nul : meme en passant son tour, on reste au-dessus de beta
        if (!estPv && (ss - 1)->coup != Move::null() && !exclu && d >= NmpMinDepth && eval >= beta
            && ss->evalStatique >= beta && pos.non_pawn_material(moi) && !(NmpTension && attaquees >= NmpTension) && ss->ply >= f->nmpMinPly
            && beta > VALUE_TB_LOSS_IN_MAX_PLY)
        {
            const Depth r = std::min(std::max(0, int(eval - beta)) / NmpEvalDiv, NmpMaxEvalR) + d / NmpDepthDiv
                          + NmpBaseReduction;
            ss->coup      = Move::null();
            ss->suite     = &f->continuationHistory[0][0][NO_PIECE][SQ_A1];
            ss->suiteCorr = &f->continuationCorrectionHistory[NO_PIECE][SQ_A1];
            pos.do_null_move(etat);
            const Value vNul = -chercher<NON_PV>(pos, ss + 1, -beta, -beta + 1, d - r, !noeudCut);
            pos.undo_null_move();

            if (vNul >= beta && vNul < VALUE_TB_WIN_IN_MAX_PLY)
            {
                if (f->nmpMinPly || d < NmpVerifProf)
                    return vNul;
                // verification sans coup nul (zugzwangs de finale)
                f->nmpMinPly  = ss->ply + 3 * (d - r) / 4;
                const Value w = chercher<NON_PV>(pos, ss, beta - 1, beta, d - r, false);
                f->nmpMinPly  = 0;
                if (w >= beta)
                    return vNul;
            }
        }

        // ProbCut : une bonne prise qui depasse nettement beta a profondeur
        // reduite suffit (marge plus serree si la position progresse)
        const Value betaPc = beta + ProbcutMargin - ProbcutProgres * progresse;
        if (!estPv && !exclu && d >= ProbcutMinDepth && std::abs(beta) < VALUE_TB_WIN_IN_MAX_PLY
            && !(trouve && fiche->depth() >= d - 3 && vTable != VALUE_NONE && vTable < betaPc))
        {
            MovePicker tri(pos, cTable, betaPc - ss->evalStatique, &f->captureHistory);
            for (Move m; (m = tri.next_move());)
            {
                if (m == exclu || !pos.legal(m))
                    continue;
                const Piece p = pos.moved_piece(m);
                ss->coup      = m;
                ss->suite     = &f->continuationHistory[ss->enEchec][true][p][m.to_sq()];
                ss->suiteCorr = &f->continuationCorrectionHistory[p][m.to_sq()];
                f->nodes.fetch_add(1, std::memory_order_relaxed);
                pos.do_move(m, etat);
                // quiescence d'abord ; recherche courte seulement si elle confirme
                v                 = -quiescence<NON_PV>(pos, ss + 1, -betaPc, -betaPc + 1, 0);
                const Depth dPc   = d - (progresse ? 5 : 3);
                if (v >= betaPc && dPc > 0)
                    v = -chercher<NON_PV>(pos, ss + 1, -betaPc, -betaPc + 1, dPc, !noeudCut);
                pos.undo_move(m);
                if (Threads.stop.load(std::memory_order_relaxed))  // valeur d'une recherche interrompue
                    return VALUE_ZERO;
                if (v >= betaPc)
                {
                    fiche->save(cle, vers_table(v, ss->ply), ss->ttPv, BOUND_LOWER, dPc + 1, m, brute,
                                TT.generation());
                    return std::abs(v) < VALUE_TB_WIN_IN_MAX_PLY ? v - (betaPc - beta) : v;
                }
            }
        }

        // Sans coup de table, le tri sera mauvais. IidMin (en test) : une
        // recherche reduite au meme noeud fournit d'abord un premier coup ;
        // sinon (par defaut), on se contente d'un pli de moins.
        if (IidMin && !cTable && d >= IidMin && !exclu)
        {
            chercher<N>(pos, ss, alpha, beta, d - IidRed, noeudCut);
            bool            t2 = false;
            const TTEntry*  f2 = TT.probe(cle, t2);
            if (t2 && f2->move() && pos.pseudo_legal(f2->move()))
            {
                cTable     = f2->move();
                priseTable = pos.capture_stage(cTable);
            }
        }
        else if ((estPv || noeudCut) && d >= 6 && !cTable)
            d -= 1;
        if (d <= 0)
            return quiescence<estPv ? PV : NON_PV>(pos, ss, alpha, beta, 0);
    }

    // =========================================================== les coups
    const PieceToHistory* hs[] = {(ss - 1)->suite, (ss - 2)->suite, (ss - 3)->suite,
                                  (ss - 4)->suite, (ss - 5)->suite, (ss - 6)->suite};
    const Move reponse = avantCase != SQ_NONE ? Move(f->counterMoves[pos.piece_on(avantCase)][avantCase]) : Move::none();
    MovePicker tri(pos, cTable, d, &f->mainHistory, &f->captureHistory, hs, &f->pawnHistory, ss->meurtriers, reponse,
                   &f->lowPlyHistory, ss->ply);

    const Bitboard menaces = threat_map(pos);  // avant que do_move ne change la position
    // option en test : nos pieces attaquees ; a partir de LmrTension, on reduit un pli de moins
    const bool tendu = LmrTension && popcount(menaces & pos.pieces(moi) & ~pos.pieces(PAWN, KING)) >= LmrTension;
    ss->menaces            = menaces;
    bool sauterTranquilles = false;
    v                      = meilleure;
#ifdef PROFILE
    bool premierPrise = false;
#endif

    for (Move m; (m = tri.next_move(sauterTranquilles));)
    {
        if (m == exclu || !pos.legal(m))
            continue;
        // a la racine, seulement les coups demandes (et pas ceux des lignes MultiPV deja faites)
        if (racine && std::find(f->rootMoves.begin() + f->pvIdx, f->rootMoves.end(), m) == f->rootMoves.end())
            continue;

        ss->nbCoups = ++rang;
#ifdef PROFILE
        if (rang == 1)
            premierPrise = pos.capture_stage(m);
#endif
        if (estPv)
            (ss + 1)->pv = nullptr;

        const bool  prise = pos.capture_stage(m);
        const bool  echec = pos.gives_check(m);
        const Piece p     = pos.moved_piece(m);
        const Square a    = m.to_sq();
        Depth       nouvelle = d - 1;
        int         extension = 0;

        // ------------------------------------------------- elagages en boucle
        if (!racine && pos.non_pawn_material(moi) && meilleure > VALUE_TB_LOSS_IN_MAX_PLY)
        {
            // au-dela d'un certain rang, plus de coups tranquilles
            if (!sauterTranquilles)
                sauterTranquilles = rang >= (LmpBase + LmpDepthFactor * d * d) / (2 - progresse);
            const Depth dLmr = std::max(0, nouvelle - reduction(d, rang));

            if (prise || echec)
            {
                if (prise && !echec && !ss->enEchec && dLmr < PriseFutProf)  // meme la prise ne remonte pas a alpha
                {
                    const Piece prisePiece = pos.piece_on(a);
                    const int   hp         = f->captureHistory[p][a][type_of(prisePiece)];
                    if (ss->evalStatique + PriseFutBase + PriseFutPente * dLmr + 2 * PieceValue[prisePiece] + PriseFutHist * hp / 1024 <= alpha)
                        continue;
                }
                if (!pos.see_ge(m, -SeeCaptureMargin * d))
                    continue;
            }
            else
            {
                const int h = f->mainHistory[moi][hist_index(m, menaces)] + (*hs[0])[p][a] + (*hs[1])[p][a];
                if (h < -HistPruneMargin * d && d <= HistPruneMaxDepth)  // a echoue partout ailleurs
                    continue;
                if (!ss->enEchec && dLmr <= FutilityMaxDepth
                    && Value(ss->evalStatique + FutilityMargin * (dLmr + 1) + h / FutHistDiv) <= alpha)
                    continue;
                if (!pos.see_ge(m, -SeeQuietMargin * dLmr * dLmr))
                    continue;
            }
        }

        // ------------------------------------------------------- extensions
        // Singularite : le coup de table est-il le seul a tenir ? On cherche
        // sans lui, plus court ; si rien n'atteint la barre, il est prolonge.
        if (!racine && d >= SingularMinDepth && m == cTable && !exclu && std::abs(vTable) < VALUE_TB_WIN_IN_MAX_PLY
            && (fiche->bound() & BOUND_LOWER) && fiche->depth() >= d - 3 && ss->ply < f->rootDepth * 2)
        {
            const Value barre = vTable - SingularBetaFactor * d / 16;
            ss->exclu         = m;
            v                 = chercher<NON_PV>(pos, ss, barre - 1, barre, (d - 1) / 2, noeudCut);
            ss->exclu         = Move::none();

            if (v < barre)
            {
                extension = 1;
                if (K1Ext && reductionRecue > 0)  // on rend au fils une partie de la reduction du pere
                    extension += std::min(reductionRecue, int(K1Ext));
                if (!estPv && v < barre - SingularExtraMargin)
                    extension = 2;
            }
            else if (barre >= beta)  // meme sans lui on coupe : tout le noeud coupe
                return barre;
            else if (vTable >= beta)
                extension = -2;
            else if (noeudCut)
                extension = -1;
        }
        else if (echec && d > EchecExtProf && pos.see_ge(m))  // echec qui ne perd rien
            extension = 1;

        nouvelle += extension;
        ss->extDoubles = (ss - 1)->extDoubles + (extension >= 2);

        ss->coup      = m;
        ss->suite     = &f->continuationHistory[ss->enEchec][prise][p][a];
        ss->suiteCorr = &f->continuationCorrectionHistory[p][a];
        prefetch(TT.first_entry(pos.key_after(m)));

        const std::uint64_t noeudsAvant = racine ? f->nodes.load(std::memory_order_relaxed) : 0;
        f->nodes.fetch_add(1, std::memory_order_relaxed);
        pos.do_move(m, etat, echec);

        // ------------------------------------------------------------- LMR
        Depth r = reduction(d, rang) - ss->ttPv - estPv + 2 * noeudCut + (priseTable && !prise) + !progresse
                + ((ss + 1)->coupures > 3) - tendu
                - (LmrRefut && (m == ss->meurtriers[0] || m == ss->meurtriers[1] || m == reponse))
                - (LmrEchec && echec) - (LmrPrise && prise);
        if (prise)
            ss->noteStat = 0;
        else
        {
            ss->noteStat = 2 * f->mainHistory[moi][hist_index(m, menaces)] + (*hs[0])[p][a] + (*hs[1])[p][a]
                         + (*hs[3])[p][a] - NoteStatDecal;
            r -= ss->noteStat / LmrHistDiv;
        }

        if (d >= 2 && rang > 1 + racine)
        {
            // fenetre nulle, profondeur reduite ; si ca passe, on approfondit
            const Depth dr      = std::clamp(nouvelle - r, 1, nouvelle + 1);
            (ss + 1)->reduction = nouvelle - dr;
            v                   = -chercher<NON_PV>(pos, ss + 1, -(alpha + 1), -alpha, dr, true);
            (ss + 1)->reduction = 0;

            if (v > alpha && dr < nouvelle)
            {
                // de combien alpha est-il battu ? largement : plus profond ;
                // de justesse : un pli de moins suffit
                nouvelle += (LmrDeeperMargin && v > meilleure + LmrDeeperMargin + 2 * nouvelle)
                          - (LmrShallowerMargin && v < meilleure + LmrShallowerMargin);
                if (nouvelle > dr)
                    v = -chercher<NON_PV>(pos, ss + 1, -(alpha + 1), -alpha, nouvelle, !noeudCut);
                suites(ss, p, a, BonusApresLmr);
            }
        }
        else if (!estPv || rang > 1)
            v = -chercher<NON_PV>(pos, ss + 1, -(alpha + 1), -alpha, nouvelle, !noeudCut);

        // en PV : le premier coup, ou celui qui a battu alpha, en fenetre pleine
        if (estPv && (rang == 1 || v > alpha))
        {
            (ss + 1)->pv    = pv;
            (ss + 1)->pv[0] = Move::none();
            v               = -chercher<PV>(pos, ss + 1, -beta, -alpha, nouvelle, false);
        }
        pos.undo_move(m);

        if (Threads.stop.load(std::memory_order_relaxed))
            return VALUE_ZERO;

        if (racine)
        {
            RootMove& rm = *std::find(f->rootMoves.begin(), f->rootMoves.end(), m);
            rm.effort += f->nodes.load(std::memory_order_relaxed) - noeudsAvant;
            rm.averageScore = rm.averageScore == -VALUE_INFINITE ? v : Value((2 * v + rm.averageScore) / 3);
            if (rang == 1 || v > alpha)
            {
                rm.score    = v;
                rm.selDepth = f->selDepth;
                rm.pv.resize(1);
                for (const Move* q = (ss + 1)->pv; *q; ++q)
                    rm.pv.push_back(*q);
            }
            else  // pas de score ancien qui le trierait trop haut
                rm.score = -VALUE_INFINITE;
        }

        if (v > meilleure)
        {
            meilleure = v;
            if (v > alpha)
            {
                meilleurCoup = m;
                if (estPv && !racine)
                    nouvelle_pv(ss->pv, m, (ss + 1)->pv);
                if (v >= beta)
                {
                    ss->coupures += 1 + !cTable;
                    PROF_COUNT(FH_TOTAL, 1);
                    PROF_COUNT(FH_FIRST, rang == 1);
                    PROF_COUNT(FH_SUMIDX, rang);
                    PROF_COUNT(FH_TT, bool(cTable));
                    PROF_COUNT(FH_TT_FIRST, cTable && rang == 1);
                    PROF_COUNT(FH_QUIET, !prise);
                    PROF_COUNT(FH_QUIET_SUMIDX, prise ? 0 : rang);
#ifdef PROFILE
                    if (!cTable && rang > 1)
                    {
                        const bool refu = m == ss->meurtriers[0] || m == ss->meurtriers[1] || m == reponse;
                        PROF_COUNT(FH_NT_LATE, 1);
                        PROF_COUNT(FH_NT_LATE_SUMIDX, rang);
                        PROF_COUNT(FH_NT_LATE_CAP, prise);
                        PROF_COUNT(FH_NT_LATE_REF, !prise && refu);
                        PROF_COUNT(FH_NT_LATE_QUIET, !prise && !refu);
                        PROF_COUNT(FH_NT_LATE_FCAP, premierPrise);
                    }
#endif
                    break;
                }
                alpha = v;
            }
        }
        if (m != meilleurCoup)  // essaye en vain : on le note pour le malus
        {
            if (prise && nbPrises < 32)
                prises[nbPrises++] = m;
            else if (!prise && nbTranq < 64)
                tranquilles[nbTranq++] = m;
        }
    }

    // ------------------------------------------------------------ bilan
    if (!rang)  // aucun coup : mat ou pat (ou tout etait exclu)
        meilleure = exclu ? alpha : ss->enEchec ? mated_in(ss->ply) : VALUE_DRAW;
    else if (meilleurCoup)
        apprendre_coupure(pos, ss, f, meilleurCoup, avantCase, tranquilles, nbTranq, prises, nbPrises, d);
    else if (!prisePrec && avantCase != SQ_NONE)  // rien n'a battu alpha : le coup d'avant etait bon
        suites(ss - 1, pos.piece_on(avantCase), avantCase, bonus_stat(d) * (1 + (meilleure > beta)));

    // CoupureAdoucie (en test) : hors PV, une coupure loin au-dessus de beta est
    // ramenee vers beta d'autant plus que la recherche etait courte
    if (CoupureAdoucie && !estPv && meilleure >= beta && std::abs(meilleure) < VALUE_TB_WIN_IN_MAX_PLY
        && std::abs(beta) < VALUE_TB_WIN_IN_MAX_PLY)
        meilleure = Value((int(meilleure) * d + int(beta)) / (d + 1));

    if (!exclu && !(racine && f->rootMoves.size() > 1))
        fiche->save(cle, vers_table(meilleure, ss->ply), ss->ttPv,
                    meilleure >= beta ? BOUND_LOWER : estPv && meilleurCoup ? BOUND_EXACT : BOUND_UPPER, d,
                    meilleurCoup, brute, TT.generation());

    // Ecart entre l'evaluation et le resultat : les corrections l'apprennent
    // (sauf si le resultat n'en dit rien : prise, ou borne du mauvais cote)
    if (!ss->enEchec && !(meilleurCoup && pos.capture(meilleurCoup))
        && !(meilleure >= beta && meilleure <= ss->evalStatique) && !(!meilleurCoup && meilleure >= ss->evalStatique))
        apprendre_correction(pos, ss, f,
                             std::clamp(int(meilleure - ss->evalStatique) * d / 8, -CORRECTION_HISTORY_LIMIT / 4,
                                        CORRECTION_HISTORY_LIMIT / 4));
    return meilleure;
}

}  // namespace
}  // namespace one
