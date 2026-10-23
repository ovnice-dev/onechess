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
    X(AspirationDelta,               10,     4,    50) \
    X(AspirationMinDepth,             4,     1,    10) \
                                                       \
    X(RfpMaxDepth,                    8,     3,    15) \
    X(RfpMargin,                     76,    40,   200) \
    X(RfpImprovingBonus,             21,     0,    80) \
                                                       \
    X(RazorMaxDepth,                  3,     1,     8) \
    X(RazorMargin,                  293,   100,   700) \
                                                       \
    X(NmpMinDepth,                    3,     2,     8) \
    X(NmpBaseReduction,               4,     2,     6) \
    X(NmpDepthDiv,                    3,     2,     8) \
    X(NmpEvalDiv,                   228,    50,   600) \
    X(NmpMaxEvalR,                    3,     0,     6) \
                                                       \
    X(ProbcutMinDepth,                5,     3,    10) \
    X(ProbcutMargin,                179,    80,   350) \
                                                       \
    X(IirMinDepth,                    4,     2,    10) \
                                                       \
    X(LmpBase,                        3,     1,     8) \
    X(LmpDepthFactor,                 1,     1,     4) \
                                                       \
    X(FutilityMaxDepth,               7,     3,    15) \
    X(FutilityMargin,               103,    50,   250) \
                                                       \
    X(HistPruneMaxDepth,              5,     2,    10) \
    X(HistPruneMargin,             3487,  1000, 12000) \
                                                       \
    X(SeeCaptureMargin,             129,    50,   400) \
    X(SeeQuietMargin,                14,     5,   120) \
                                                       \
    X(LmrBase,                       56,    20,   150) \
    X(LmrDivisor,                   266,   120,   400) \
    X(LmrHistDiv,                  13190,  2000, 30000) \
                                                       \
    X(SingularMinDepth,               6,     4,    12) \
    X(SingularBetaFactor,            24,     8,    64) \
    X(SingularExtraMargin,           18,     0,   100) \
                                                       \
    X(StatBonusMul,                 257,    50,   400) \
    X(StatBonusSub,                  81,     0,   400) \
    X(StatBonusMax,                1826,   500,  4000) \
    X(StatMalusMul,                 485,   100,   900) \
    X(StatMalusSub,                 111,     0,   400) \
    X(StatMalusMax,                1897,   500,  4000) \
                                                       \
    X(QsDeltaMargin,                145,    50,   500) \
    X(QsFutilityMargin,             111,    50,   400) \
                                                       \
    X(CorrHistDiv,                 11136,  1024, 65536) \
    X(CorrHistWeight,                 1,     1,     8) \
    X(CorrNpWeight,                   1,     0,     8) \
    X(CorrContWeight,                 1,     0,     8) \
    X(CorrMinorWeight,                1,     0,     8) \
    X(CorrMajorWeight,                1,     0,     8) \
    X(TtCutRefine,                    0,     0,     1) \
    X(ThreatHist,                     1,     0,     1) \
    X(FutValue,                       0,     0,     1) \
    X(CapFutility,                    1,     0,     1) \
    X(ProbcutSf,                      1,     0,     1) \
    X(SmallProbcut,                   0,     0,   800) \
    X(IirSf,                          0,     0,     1) \
    X(NegExtSf,                       0,     0,     1) \
    X(MultiCutSf,                     0,     0,     1) \
    X(ImprovingBeta,                  0,     0,     1) \
    X(RfpOppW,                        0,     0,   200) \
    X(PostLmrSf,                      1,     0,     1) \
    X(NoKillers,                      0,     0,     1) \
    X(NoCounter,                      0,     0,     1) \
    X(LmrFrac,                        0,     0,     1) \
    X(LmrAllScale,                    0,     0,     1) \
    X(LmrAlphaEval,                   0,     0,     1) \
    X(LmrCutoffSf,                    0,     0,     1) \
    X(BadQuiets,                      14000,     0, 30000) \
    X(MultiCutCorr,                   0,     0,     1) \
    X(TtCutHistSf,                    0,     0,     1) \
    X(RfpOwn,                         0,     0,     1) \
    X(QsFailFirm,                     0,     0,     1) \
    X(ContHist6,                      0,     0,     1) \
    X(EvalDiffHist,                   1,     0,     1) \
    X(ContWeightsSf,                  0,     0,     1) \
    X(BonusSf,                        0,     0,     1) \
    X(ThreatOrder,                    48,     0,    96) \
    X(CheckSeeMargin,                 0,     0,   400) \
    X(LowPlyHist,                     0,     0,    16) \
    X(CapStatMul,                     0,     0,  4000) \
    X(MalusDecay,                   924,   512,  1024) \
    X(LmrDeeperMargin,                53,     0,   200) \
    X(LmrShallowerMargin,             8,     0,    60) \
    X(Hindsight,                      1,     0,     1) \
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
    X(TmSurprisePct,                100,   100,   250) \
    X(TmLowMtg,                       0,     0,     1) \
                                                       \
    X(Rule50Damp,                     0,     0,   512)

namespace one {
namespace Tune {

#define TUNABLE_DECLARE(name, def, lo, hi) extern int name;
TUNABLE_LIST(TUNABLE_DECLARE)
#undef TUNABLE_DECLARE

}  // namespace Tune
}  // namespace one
