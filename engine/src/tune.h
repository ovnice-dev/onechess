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
    X(RazorMargin,                  293,   100,   700) \
                                                       \
    X(NmpMinDepth,                    3,     2,     8) \
    X(NmpBaseReduction,               4,     2,     6) \
    X(NmpDepthDiv,                    3,     2,     8) \
    X(NmpEvalDiv,                   222,    50,   600) \
    X(NmpMaxEvalR,                    3,     0,     6) \
                                                       \
    X(ProbcutMinDepth,                5,     3,    10) \
    X(ProbcutMargin,                173,    80,   350) \
                                                       \
                                                       \
    X(LmpBase,                        3,     1,     8) \
    X(LmpDepthFactor,                 1,     1,     4) \
                                                       \
    X(FutilityMaxDepth,               7,     3,    15) \
    X(FutilityMargin,               90,    50,   250) \
                                                       \
    X(HistPruneMaxDepth,              5,     2,    10) \
    X(HistPruneMargin,             3306,  1000, 12000) \
                                                       \
    X(SeeCaptureMargin,             128,    50,   400) \
    X(SeeQuietMargin,                17,     5,   120) \
                                                       \
    X(LmrBase,                       54,    20,   150) \
    X(LmrDivisor,                   262,   120,   400) \
    X(LmrHistDiv,                  13107,  2000, 30000) \
                                                       \
    X(SingularMinDepth,               6,     4,    12) \
    X(SingularBetaFactor,            23,     8,    64) \
    X(SingularExtraMargin,           13,     0,   100) \
                                                       \
    X(StatBonusMul,                 245,    50,   400) \
    X(StatBonusSub,                  79,     0,   400) \
    X(StatBonusMax,                1828,   500,  4000) \
    X(StatMalusMul,                 500,   100,   900) \
    X(StatMalusSub,                 108,     0,   400) \
    X(StatMalusMax,                1980,   500,  4000) \
                                                       \
    X(QsDeltaMargin,                145,    50,   500) \
    X(QsFutilityMargin,             109,    50,   400) \
                                                       \
    X(CorrHistDiv,                  9281,  1024, 65536) \
    X(CorrHistWeight,                 1,     1,     8) \
    X(CorrNpWeight,                   1,     0,     8) \
    X(CorrContWeight,                 1,     0,     8) \
    X(CorrMinorWeight,                1,     0,     8) \
    X(CorrMajorWeight,                1,     0,     8) \
    X(CorrMenaceWeight,               0,     0,     8) \
    X(ThreatHist,                     1,     0,     1) \
    X(BadQuiets,                      13491,     0, 30000) \
    X(ContHist6,                      1,     0,     1) \
    X(ThreatOrder,                    48,     0,    96) \
    X(MalusDecay,                   953,   512,  1024) \
    X(LmrDeeperMargin,                51,     0,   200) \
    X(LmrShallowerMargin,             8,     0,    60) \
    X(K1Ext,                          1,     0,     3) \
    X(QsFin,                          3,     0,     3) \
    X(R50Div,                         0,     0,   400) \
    X(CoupureAdoucie,                 0,     0,     1) \
    X(LmrRefut,                       0,     0,     1) \
    X(LmrEchec,                       0,     0,     1) \
    X(LmrPrise,                       0,     0,     1) \
    X(NmpTension,                     0,     0,     8) \
    X(RfpTension,                     0,     0,   100) \
    X(IidMin,                         0,     0,    20) \
    X(IidRed,                         3,     1,     8) \
    X(TtRajeunir,                     1,     0,     1) \
    X(TmMtgBase,                    880,   400,  1600) \
    X(TmMtgPly,                    1164,   300,  4000) \
    X(TmMtgCap,                     880,   400,   990) \
    X(TmMtgPlafMax,                  63,    20,   120) \
    X(TmMtgPlafBase,                 15,     5,    40) \
    X(TmMtgPlafPente,               110,    30,   300) \
    X(TmPlafReste,                  840,   500,   950) \
    X(TmSdBase,                      84,    20,   300) \
    X(TmSdPente,                     42,    10,   150) \
    X(TmSdCap,                      200,    80,   400) \
    X(TmSdPlafMax,                   70,    30,   120) \
    X(TmSdPlafBase,                  40,    15,    80) \
    X(TmSdPlafPly,                  120,    40,   400) \
    X(LmrTension,                     0,     0,     8) \
    X(TmTension,                      0,     0,   300) \
    X(OrdMain,                       16,     0,    48) \
    X(OrdSuite1,                     16,     0,    48) \
    X(OrdSuite2,                      8,     0,    32) \
    X(OrdSuite3,                      8,     0,    32) \
    X(OrdSuite4,                      8,     0,    32) \
    X(OrdSuite6,                      7,     0,    32) \
    X(OrdPions,                       8,     0,    32) \
    X(OrdEchec,                   16578,     0, 32768) \
    X(EcartMin,                     186,    50,   400) \
    X(EcartMax,                     197,    50,   400) \
    X(EcartDecal,                    53,     0,   200) \
    X(EcartPrincipal,                11,     0,    30) \
    X(EcartPions,                    13,     0,    30) \
    X(ReculMarge,                   165,    50,   400) \
    X(RfpStatDiv,                   303,   100,  1000) \
    X(NmpVerifProf,                  14,     8,    20) \
    X(ProbcutProgres,                65,     0,   150) \
    X(PriseFutProf,                   8,     4,    12) \
    X(PriseFutBase,                 239,   100,   400) \
    X(PriseFutPente,                249,   100,   400) \
    X(PriseFutHist,                 132,     0,   300) \
    X(FutHistDiv,                   197,    50,   600) \
    X(NoteStatDecal,               3755,     0,  8000) \
    X(BonusApresLmr,               1377,   400,  3000) \
    X(EchecExtProf,                   8,     4,    14) \
    X(PriseHistDiv,                  16,     4,    48) \
    X(TriPlancher,                 3037,  1000,  6000) \
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

// ONE_FIGE (compilations de diffusion, build.sh) : les constantes deviennent
// constexpr, le compilateur les integre au code (+2 a 3 % de vitesse mesures
// le 17-11) ; elles ne sont alors plus reglables par setoption. Les tests
// (CI, SPSA) compilent sans ONE_FIGE.
#ifdef ONE_FIGE
#define TUNABLE_DECLARE(name, def, lo, hi) constexpr int name = def;
#else
#define TUNABLE_DECLARE(name, def, lo, hi) extern int name;
#endif
TUNABLE_LIST(TUNABLE_DECLARE)
#undef TUNABLE_DECLARE

}  // namespace Tune
}  // namespace one
