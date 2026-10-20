#pragma once

// Toutes les constantes reglables de la recherche, declarees une seule fois.
//
// La liste sert a la fois a definir les variables et a les exposer en options
// UCI : un test SPRT sur une valeur ne demande donc pas de recompilation, il
// suffit d'un "setoption". C'est ce qui rend praticable le balayage des
// constantes qui n'ont jamais ete touchees.
//
//        nom                     defaut   min    max
#define TUNABLE_LIST(X) \
    X(AspirationDelta,               11,     4,    50) \
    X(AspirationMinDepth,             4,     1,    10) \
                                                       \
    X(RfpMaxDepth,                    8,     3,    15) \
    X(RfpMargin,                     76,    40,   200) \
    X(RfpImprovingBonus,             23,     0,    80) \
                                                       \
    X(RazorMaxDepth,                  3,     1,     8) \
    X(RazorMargin,                  295,   100,   700) \
                                                       \
    X(NmpMinDepth,                    3,     2,     8) \
    X(NmpBaseReduction,               4,     2,     6) \
    X(NmpDepthDiv,                    3,     2,     8) \
    X(NmpEvalDiv,                   213,    50,   600) \
    X(NmpMaxEvalR,                    3,     0,     6) \
                                                       \
    X(ProbcutMinDepth,                5,     3,    10) \
    X(ProbcutMargin,                187,    80,   350) \
                                                       \
    X(IirMinDepth,                    4,     2,    10) \
                                                       \
    X(LmpBase,                        3,     1,     8) \
    X(LmpDepthFactor,                 1,     1,     4) \
                                                       \
    X(FutilityMaxDepth,               7,     3,    15) \
    X(FutilityMargin,               114,    50,   250) \
                                                       \
    X(HistPruneMaxDepth,              5,     2,    10) \
    X(HistPruneMargin,             3351,  1000, 12000) \
                                                       \
    X(SeeCaptureMargin,             151,    50,   400) \
    X(SeeQuietMargin,                12,     5,   120) \
                                                       \
    X(LmrBase,                       62,    20,   150) \
    X(LmrDivisor,                   259,   120,   400) \
    X(LmrHistDiv,                  14088,  2000, 30000) \
                                                       \
    X(SingularMinDepth,               6,     4,    12) \
    X(SingularBetaFactor,            23,     8,    64) \
    X(SingularExtraMargin,           20,     0,   100) \
                                                       \
    X(StatBonusMul,                 253,    50,   400) \
    X(StatBonusSub,                  81,     0,   400) \
    X(StatBonusMax,                1655,   500,  4000) \
    X(StatMalusMul,                 478,   100,   900) \
    X(StatMalusSub,                 111,     0,   400) \
    X(StatMalusMax,                1864,   500,  4000) \
                                                       \
    X(QsDeltaMargin,                171,    50,   500) \
    X(QsFutilityMargin,             104,    50,   400) \
                                                       \
    X(CorrHistDiv,                 11565,  1024, 65536) \
    X(CorrHistWeight,                 1,     1,     8) \
    X(CorrNpWeight,                   1,     0,     8) \
    X(CorrContWeight,                 1,     0,     8) \
    X(CorrMinorWeight,                0,     0,     8) \
    X(CorrMajorWeight,                0,     0,     8) \
    X(PriorCmBonus,                   0,     0,     1) \
    X(NmpCutOnly,                     0,     0,     1) \
    X(RazorAllOnly,                   0,     0,     1) \
    X(TtCutRefine,                    0,     0,     1) \
    X(ThreatHist,                     1,     0,     1) \
    X(FhRecovery,                     0,     0,     1) \
    X(NmpPriorFh,                     0,     0,   200) \
    X(FutValue,                       0,     0,     1) \
    X(CapFutility,                    0,     0,     1) \
    X(ProbcutSf,                      0,     0,     1) \
    X(SmallProbcut,                   0,     0,   800) \
    X(IirSf,                          0,     0,     1) \
    X(ExtSf,                          0,     0,     1) \
    X(NegExtSf,                       0,     0,     1) \
    X(MultiCutSf,                     0,     0,     1) \
    X(ImprovingBeta,                  0,     0,     1) \
    X(RfpOppW,                        0,     0,   200) \
    X(PostLmrSf,                      0,     0,     1) \
    X(NoKillers,                      0,     0,     1) \
    X(NoCounter,                      0,     0,     1) \
    X(LmrFrac,                        0,     0,     1) \
    X(LmrAllScale,                    0,     0,     1) \
    X(LmrAlphaEval,                   0,     0,     1) \
    X(LmrCutoffSf,                    0,     0,     1) \
    X(BadQuiets,                      0,     0, 30000) \
    X(MultiCutCorr,                   0,     0,     1) \
    X(TtCutHistSf,                    0,     0,     1) \
    X(RfpOwn,                         0,     0,     1) \
    X(EvalDiffHist,                   0,     0,     1) \
    X(HistDecay,                      0,     0,     1) \
    X(ContWeightsSf,                  0,     0,     1) \
    X(BonusSf,                        0,     0,     1) \
    X(ThreatOrder,                    48,     0,    96) \
    X(CheckSeeMargin,                 0,     0,   400) \
    X(LowPlyHist,                     0,     0,    16) \
    X(CapStatMul,                     0,     0,  4000) \
    X(MalusDecay,                   921,   512,  1024) \
    X(LmrDeeperMargin,                0,     0,   200) \
    X(LmrShallowerMargin,             0,     0,    60) \
    X(Hindsight,                      0,     0,     1) \
    X(UpRep,                          0,     0,     2) \
                                                       \
    X(TmStablePct,                  110,    50,   300) \
    X(TmChangedPct,                 170,    50,   300) \
    X(TmFallPct,                      10,     0,    50) \
    X(TmSingleMs,                     300,     0,  1000) \
    X(TmEffortPct,                600,     0,  1000) \
    X(TmNoStartPct,              60,    30,   100) \
    X(TmInstMul,                  250,     0,  1000) \
    X(TmClockPct,                 900,     0,  1000) \
    X(TmSfAlloc,                      0,     0,     1) \
                                                       \
    X(Rule50Damp,                     0,     0,   512)

namespace one {
namespace Tune {

#define TUNABLE_DECLARE(name, def, lo, hi) extern int name;
TUNABLE_LIST(TUNABLE_DECLARE)
#undef TUNABLE_DECLARE

}  // namespace Tune
}  // namespace one
