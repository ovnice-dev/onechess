#pragma once

#include <cstdint>
#include <string>

#include "nnue_types.h"
#include "types.h"

namespace one {

class Position;
struct StateInfo;

namespace NNUE {

// ---------------------------------------------------------------------------
// Reseau a perspectives : (768 x buckets -> HL) x2 -> 1
//
// 768 entrees de base = 12 types de pieces x 64 cases, vues depuis le camp
// considere. Chaque camp a son accumulateur ; la couche de sortie lit d'abord
// celui du camp au trait, ce qui donne au reseau la notion de trait sans avoir
// a l'apprendre deux fois.
//
// Le decoupage en buckets duplique ces entrees selon la case du roi du camp
// considere. C'est ce qui exprime la position d'une piece RELATIVEMENT au roi,
// information absente d'un jeu de 768 entrees seul. Le miroir horizontal
// replie l'echiquier sur sa moitie dame quand le roi est colonnes e-h.
//
// Consequence : un coup de roi qui change de bucket ou de sens de miroir
// invalide TOUTES les caracteristiques de sa perspective. La mise a jour
// incrementale bascule alors sur un recalcul, pour cette perspective seule.
// ---------------------------------------------------------------------------

constexpr int QA    = 255;   // echelle des poids d'entree
constexpr int QB    = 64;    // echelle des poids de sortie
constexpr int SCALE = 400;   // conversion sortie -> centipions

constexpr int BASE_INPUTS = 768;

// Paires de pions (PP_3Wide de Stockfish 19) : 96 identifiants de pion, paires
// non ordonnees. Annoncees par le bit PAIRS_FLAG du champ activation.
constexpr int           PAIR_INPUTS = 96 * 95 / 2;
constexpr std::uint32_t PAIRS_FLAG  = 256;

// Menaces : une entree par (piece relative, case relative) d'une piece, hors
// rois, attaquee par le camp adverse. Bit THREAT_FLAG du champ activation.
constexpr int           THREAT_INPUTS = 768;
constexpr std::uint32_t THREAT_FLAG   = 512;

// Couches cachees : 2 * HIDDEN_SIZE -> l1 -> l2 -> 1 par bucket de sortie, en
// entiers (activations 0..QH). Bit HIDDEN_FLAG ; l1 et l2 suivent les menaces.
constexpr std::uint32_t HIDDEN_FLAG = 1024;
// Bit 12 : entrees "menaces completes" (Full Threats de Stockfish 18-19, phase 10) ;
// table int8 de FULL_THREAT_DIM x HIDDEN_SIZE apres les biais de la 1re couche.
constexpr std::uint32_t FULLTHREAT_FLAG = 4096;
constexpr int           FULL_THREAT_DIM = 59808;
// Produit par paires avant l1 (Stockfish) : p = (a[i] * a[i + H/2]) >> 9 dans
// [0, QH], H/2 entrees par perspective, poids l1 en int8.
constexpr std::uint32_t PAIRMUL_FLAG = 2048;
constexpr int           QH          = 127;
constexpr int           MAX_L1      = 64;
constexpr int           MAX_L2      = 64;

constexpr std::uint32_t MAGIC  = 0x334E4E4F;  // "ONN3" : sans champ activation
constexpr std::uint32_t MAGIC4 = 0x344E4E4F;  // "ONN4" : avec champ activation

constexpr int MAX_OUT_BUCKETS = 8;

// Fonction d'activation de la couche cachee.
//
//   CReLU  : clamp(a, 0, QA)          sortie en unites QA
//   SCReLU : clamp(a, 0, QA)^2        sortie en unites QA^2
//
// Le carre donne au reseau une non-linearite qui ne sature pas brutalement :
// entre 0 et QA la pente croit au lieu d'etre constante, ce qui laisse la
// couche de sortie distinguer une caracteristique faiblement active d'une
// caracteristique fortement active. Le prix est une multiplication de plus par
// neurone et un facteur QA a retirer avant d'ajouter le biais de sortie.
//
// Un fichier "ONN3" n'a pas ce champ : il est lu comme CReLU.
enum Activation : std::uint32_t {
    ACT_CRELU  = 0,
    ACT_SCRELU = 1,
};

// Bucket de SORTIE : plusieurs jeux de poids de sortie, choisis selon le
// materiel restant. Une finale et un milieu de partie ne convertissent pas une
// meme configuration en la meme evaluation ; leur imposer les memes poids force
// le reseau a un compromis. Contrairement aux buckets de roi, ceux-ci ne
// multiplient que la DERNIERE couche, donc quelques milliers de poids : ils
// paient des le volume de donnees dont on dispose.
//
// Formule identique a celle de tools/features.py.
inline int output_bucket(int pieceCount, int nBuckets) {
    return (pieceCount - 2) / (32 / nBuckets);
}

// Ce qu'une perspective doit savoir de son propre roi.
struct PerspInfo {
    int bucket;
    int flip;      // 0 ou 7, applique en XOR sur la case
};

bool          load(const std::string& path);
bool          loaded();
void          unload();
std::string   info();
std::uint32_t generation();

PerspInfo persp_info(Color p, Square ksq);
int       feature_index(Color p, Piece pc, Square s, const PerspInfo& pi);

void  refresh_side(Accumulator& acc, const Position& pos, Color p);
void  refresh(Accumulator& acc, const Position& pos);
Value output(const Accumulator& acc, Color stm, int bucket);

// Met l'accumulateur a jour de la facon la moins couteuse possible, puis lit.
Value evaluate(const Position& pos);

// Menaces completes (phase 10) : vrai si le reseau charge en a ; ft_on_move() est
// appele par do_move() et do_null_move() pour noter les menaces que le coup change.
extern bool FullThreatsOn;
void        ft_on_move(const Position& pos);

}  // namespace NNUE
}  // namespace one
