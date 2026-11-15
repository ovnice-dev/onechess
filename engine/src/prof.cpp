#include "prof.h"

#ifdef PROFILE

    #include <atomic>
    #include <cstdio>

namespace one {
namespace Prof {

thread_local Counters Local{};

namespace {

std::atomic<std::uint64_t> TotalCycles[NCAT];
std::atomic<std::uint64_t> TotalCalls[NCAT];

const char* const Names[NCAT] = {
  "recherche totale",   "evaluation",      "  dont sortie NNUE", "  dont mise a jour acc.",
  "selection de coups", "SEE (tous)",      "sonde de table",     "do_move",
  "undo_move",          "legal",           "gives_check",        "key_after",
  "historiques (MAJ)",  "correction eval", "nulle / repetition", "ecriture en table",
  "coupures beta",      "  dont 1er coup", "somme des rangs",    "avec coup de table",
  "  dont 1er coup",    "par coup tranquille", "somme de leurs rangs"};

// Postes disjoints qui, ensemble, recouvrent une partie de la recherche. Le
// "reste" est ce qu'ils laissent de cote.
constexpr int Top[] = {EVAL,       MOVEPICK, TTPROBE, DOMOVE, UNDOMOVE, LEGAL,
                       GIVESCHECK, KEYAFTER, HISTUPD, CORR,   DRAW,     TTSAVE};

// Details imbriques dans un poste ci-dessus : affiches, jamais additionnes.
// SEE est appele a la fois depuis la selection de coups et depuis la
// recherche : l'additionner compterait deux fois sa part interne au tri.
constexpr int Detail[] = {NNUE_APPLY, NNUE_OUT, SEE};

// Cout d'une portee vide, vu depuis une portee englobante. Lire deux fois le
// compteur n'est pas gratuit : une fonction appelee des millions de fois
// porterait ce cout comme s'il etait le sien, et la recherche qui l'englobe
// aussi. On le mesure pour le retrancher.
double scope_overhead() {
    const Counters saved = Local;
    constexpr int  N     = 2'000'000;

    const std::uint64_t t0 = __rdtsc();
    for (int i = 0; i < N; ++i)
    {
        Scope s(NNUE_OUT);
        asm volatile("" ::: "memory");
    }
    const double per = double(__rdtsc() - t0) / N;

    Local = saved;
    return per;
}

}  // namespace

void reset() {
    for (int i = 0; i < NCAT; ++i)
    {
        TotalCycles[i] = 0;
        TotalCalls[i]  = 0;
    }
    Local = Counters{};
}

void flush() {
    for (int i = 0; i < NCAT; ++i)
    {
        TotalCycles[i] += Local.cycles[i];
        TotalCalls[i] += Local.calls[i];
    }
    Local = Counters{};
}

void report(std::uint64_t nodes) {
    const double ovh = scope_overhead();

    double cyc[NCAT], calls[NCAT];
    for (int i = 0; i < NCAT; ++i)
    {
        cyc[i]   = double(TotalCycles[i].load());
        calls[i] = double(TotalCalls[i].load());
    }

    // Chaque portee imbriquee ajoute son propre cout de mesure a celles qui
    // l'englobent. Sans correction, le "reste" -- justement ce qu'on cherche --
    // serait gonfle par l'instrument lui-meme. Les simples comptes (au-dela de
    // NCYC) n'ouvrent pas de portee et n'entrent pas dans la correction.
    double nested = 0;
    for (int i = EVAL; i < NCYC; ++i)
        nested += calls[i];
    const double search = cyc[SEARCH] - ovh * nested;
    cyc[EVAL] -= ovh * (calls[NNUE_OUT] + calls[NNUE_APPLY]);

    const double n = double(nodes);
    std::printf("\nProfil par compteurs de cycles  (%.0f noeuds ; portee vide = %.0f cycles, retranche)\n",
                n, ovh);
    std::printf("  %-26s %12s %8s %11s %8s\n", "poste", "appels", "/noeud", "cycles/app", "part");

    auto row = [&](int i) {
        std::printf("  %-26s %12.0f %8.2f %11.0f %7.1f %%\n", Names[i], calls[i], calls[i] / n,
                    calls[i] > 0 ? cyc[i] / calls[i] : 0.0, 100.0 * cyc[i] / search);
    };

    double known = 0;
    for (int i : Top)
    {
        row(i);
        known += cyc[i];
    }
    std::printf("  %-26s %12s %8s %11s %7.1f %%\n", "reste de la recherche", "", "", "",
                100.0 * (search - known) / search);

    std::printf("  -- detail, deja compte ci-dessus --\n");
    for (int i : Detail)
        row(i);

    std::printf("  cycles de reference par noeud : %.0f\n", search / n);

    // Qualite du tri des coups : quand une coupure beta a lieu, le coup qui la
    // produit est-il le premier essaye ? Reference courante : 80 a 90 %, plus
    // de 90 % pour un tri excellent. Ce sont des comptes exacts, independants
    // de la charge de la machine.
    //
    // La separation avec / sans coup de table dit OU le tri pece : avec un
    // coup de table, le premier coup devrait couper presque toujours ; sans,
    // tout repose sur les historiques et la generation des coups.
    const double tot = calls[FH_TOTAL];
    if (tot > 0)
    {
        const double tt = calls[FH_TT], nott = tot - tt, q = calls[FH_QUIET];
        std::printf("\nQualite du tri (recherche principale)\n"
                    "  coupures beta           : %.0f  (%.3f par noeud)\n"
                    "  au premier coup essaye  : %.1f %%\n"
                    "  rang moyen du coup qui coupe : %.2f\n",
                    tot, tot / n, 100.0 * calls[FH_FIRST] / tot, calls[FH_SUMIDX] / tot);
        std::printf("  noeuds AVEC coup de table : %5.1f %% des coupures, dont %5.1f %% au 1er coup\n",
                    100.0 * tt / tot, tt > 0 ? 100.0 * calls[FH_TT_FIRST] / tt : 0.0);
        std::printf("  noeuds SANS coup de table : %5.1f %% des coupures, dont %5.1f %% au 1er coup\n",
                    100.0 * nott / tot,
                    nott > 0 ? 100.0 * (calls[FH_FIRST] - calls[FH_TT_FIRST]) / nott : 0.0);
        std::printf("  coupures par coup tranquille : %5.1f %%, rang moyen %.2f\n", 100.0 * q / tot,
                    q > 0 ? calls[FH_QUIET_SUMIDX] / q : 0.0);
    }
    std::fflush(stdout);
}

}  // namespace Prof
}  // namespace one

#endif
