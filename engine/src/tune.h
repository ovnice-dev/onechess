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
    X(AspirationDelta,               9,     4,    50) \
    X(AspirationMinDepth,             4,     1,    10) \
                                                       \
    X(RfpMaxDepth,                    8,     3,    15) \
                                                       \
    X(RazorMaxDepth,                  3,     1,     8) \
    X(RazorMargin,                  292,   100,   700) \
                                                       \
    X(NmpMinDepth,                    3,     2,     8) \
    X(NmpBaseReduction,               4,     2,     6) \
    X(NmpDepthDiv,                    3,     2,     8) \
    X(NmpEvalDiv,                   220,    50,   600) \
    X(NmpMaxEvalR,                    3,     0,     6) \
                                                       \
    X(ProbcutMinDepth,                5,     3,    10) \
    X(ProbcutMargin,                175,    80,   350) \
                                                       \
    X(IirMinDepth,                    4,     2,    10) \
                                                       \
    X(LmpBase,                        3,     1,     8) \
    X(LmpDepthFactor,                 1,     1,     4) \
                                                       \
    X(FutilityMaxDepth,               7,     3,    15) \
    X(FutilityMargin,               96,    50,   250) \
                                                       \
    X(HistPruneMaxDepth,              5,     2,    10) \
    X(HistPruneMargin,             3302,  1000, 12000) \
                                                       \
    X(SeeCaptureMargin,             130,    50,   400) \
    X(SeeQuietMargin,                15,     5,   120) \
                                                       \
    X(LmrBase,                       55,    20,   150) \
    X(LmrDivisor,                   264,   120,   400) \
    X(LmrHistDiv,                  13091,  2000, 30000) \
                                                       \
    X(SingularMinDepth,               6,     4,    12) \
    X(SingularBetaFactor,            24,     8,    64) \
    X(SingularExtraMargin,           16,     0,   100) \
                                                       \
    X(StatBonusMul,                 248,    50,   400) \
    X(StatBonusSub,                  83,     0,   400) \
    X(StatBonusMax,                1808,   500,  4000) \
    X(StatMalusMul,                 509,   100,   900) \
    X(StatMalusSub,                 108,     0,   400) \
    X(StatMalusMax,                1908,   500,  4000) \
                                                       \
    X(QsDeltaMargin,                140,    50,   500) \
    X(QsFutilityMargin,             108,    50,   400) \
                                                       \
    X(CorrHistDiv,                 10803,  1024, 65536) \
    X(CorrHistWeight,                 1,     1,     8) \
    X(CorrNpWeight,                   1,     0,     8) \
    X(CorrContWeight,                 1,     0,     8) \
    X(CorrMinorWeight,                1,     0,     8) \
    X(CorrMajorWeight,                1,     0,     8) \
    X(ThreatHist,                     1,     0,     1) \
    X(CapFutility,                    1,     0,     1) \
    X(ProbcutSf,                      1,     0,     1) \
    X(IirSf,                          1,     0,     1) \
    X(PostLmrSf,                      1,     0,     1) \
    X(BadQuiets,                      13845,     0, 30000) \
    X(ContHist6,                      1,     0,     1) \
    X(EvalDiffHist,                   1,     0,     1) \
    X(ThreatOrder,                    48,     0,    96) \
    X(MalusDecay,                   942,   512,  1024) \
    X(LmrDeeperMargin,                53,     0,   200) \
    X(LmrShallowerMargin,             8,     0,    60) \
    X(Hindsight,                      1,     0,     1) \
    X(ExtLimit,                       1,     0,     1) \
    X(RfpCorr,                        0,     0,   256) \
    X(InsufMat,                       0,     0,     1) \
    X(K1Ext,                          1,     0,     3) \
    X(QsFin,                          0,     0,     3) \
                                                       \
    X(TmStablePct,                  110,    50,   300) \
    X(TmChangedPct,                 170,    50,   300) \
    X(TmFallPct,                      10,     0,    50) \
    X(TmSingleMs,                     300,     0,  1000) \
    X(TmEffortPct,                600,     0,  1000) \
    X(TmNoStartPct,              60,    30,   100) \
    X(TmInstMul,                  250,     0,  1000) \
    X(TmClockPct,                 900,     0,  1000) \
    X(TmSurprisePct,                130,   100,   250) \
                                                       \

namespace one {
namespace Tune {

#define TUNABLE_DECLARE(name, def, lo, hi) extern int name;
TUNABLE_LIST(TUNABLE_DECLARE)
#undef TUNABLE_DECLARE

}  // namespace Tune
}  // namespace one
