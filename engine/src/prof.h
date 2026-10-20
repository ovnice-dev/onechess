#pragma once

// Profilage par compteurs de cycles, actif uniquement avec -DPROFILE.
//
// gprof ne voit rien de la recherche : sous MinGW il n'echantillonne que le fil
// principal, qui attend pendant que la recherche tourne dans un fil secondaire
// (verifie le 11-09 : profil vide). Le profil par difference (tools/profil.sh)
// isolait l'evaluation (30 %) et la generation de coups (0,7 %), mais laissait
// 69 % du temps dans un bloc indistinct. Ces compteurs le decoupent.
//
// Chaque fonction instrumentee ouvre une portee qui lit le compteur de temps du
// processeur a l'entree et a la sortie. Les totaux sont tenus par fil, puis
// verses dans un total global a la fin de chaque recherche. Hors -DPROFILE tout
// disparait a la compilation : le binaire livre n'en porte aucune trace.
//
// Les compteurs de cycles mesurent du temps ECOULE : un fil interrompu par un
// autre processus au milieu d'une portee la gonfle. D'ou le lancement en
// priorite haute. Les simples COMPTES (coupures, rangs) sont en revanche exacts
// quelle que soit la charge : ils ne dependent que de la recherche.

#include <cstdint>

#ifdef PROFILE
    #include <x86intrin.h>
#endif

namespace one {
namespace Prof {

enum Cat : int {
    SEARCH,      // toute la recherche d'un fil : la reference des pourcentages

    EVAL,        // Eval::evaluate, NNUE comprise
    NNUE_OUT,    //   couche de sortie (incluse dans EVAL)
    NNUE_APPLY,  //   mise a jour de l'accumulateur, par demi-coup (incluse dans EVAL)

    MOVEPICK,    // MovePicker::next_move : generation et tri des coups
    SEE,         // Position::see_ge, tous appels confondus (tri ET elagage)

    TTPROBE,     // sonde de la table de transposition
    DOMOVE,
    UNDOMOVE,
    LEGAL,       // Position::legal
    GIVESCHECK,  // Position::gives_check
    KEYAFTER,    // Position::key_after, pour le prechargement de la table
    HISTUPD,     // update_all_stats : bonus et malus apres une coupure
    CORR,        // corrected_eval : historique de correction
    DRAW,        // Position::is_draw : repetition et regle des 50 coups
    TTSAVE,      // TTEntry::save : ecriture en table

    NCYC,              // fin des postes chronometres ; ce qui suit ne fait que compter
    FH_TOTAL = NCYC,   // coupures beta de la recherche principale
    FH_FIRST,          //   dont produites par le premier coup essaye
    FH_SUMIDX,         //   somme des rangs du coup qui coupe
    FH_TT,             // coupures dans un noeud qui disposait d'un coup de table
    FH_TT_FIRST,       //   dont au premier coup (en principe le coup de table)
    FH_QUIET,          // coupures produites par un coup tranquille
    FH_QUIET_SUMIDX,   //   somme de leurs rangs

    NCAT
};

#ifdef PROFILE

struct Counters {
    std::uint64_t cycles[NCAT];
    std::uint64_t calls[NCAT];
};

extern thread_local Counters Local;

struct Scope {
    Cat           cat;
    std::uint64_t t0;

    explicit Scope(Cat c) :
        cat(c),
        t0(__rdtsc()) {}

    ~Scope() {
        Local.cycles[cat] += __rdtsc() - t0;
        ++Local.calls[cat];
    }
};

inline void count(Cat c, std::uint64_t n) { Local.calls[c] += n; }

void reset();
void flush();
void report(std::uint64_t nodes);

    #define PROF_SCOPE(c) ::one::Prof::Scope prof_scope_(::one::Prof::c)
    #define PROF_COUNT(c, n) ::one::Prof::count(::one::Prof::c, std::uint64_t(n))

#else

inline void reset() {}
inline void flush() {}
inline void report(std::uint64_t) {}

    #define PROF_SCOPE(c) ((void) 0)
    #define PROF_COUNT(c, n) ((void) 0)

#endif

}  // namespace Prof
}  // namespace one
