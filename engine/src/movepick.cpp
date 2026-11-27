/*
  Onechess, a UCI chess engine derived from Stockfish.
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)
  Copyright (C) 2026 ovnice (modifications)

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

// Tri des coups (reecrit pour Onechess le 15-11-2026).
//
// Recherche principale : coup de la table, bonnes prises (echange >= 0),
// refutations (coups meurtriers, reponse), coups tranquilles, mauvaises prises,
// puis (BadQuiets) les tranquilles les plus mal notes.
// En echec : coup de la table, puis toutes les parades triees.
// Quiescence : coup de la table, puis les prises triees.
// ProbCut : coup de la table, puis les prises qui passent le seuil.

#include "movepick.h"

#include <cassert>
#include <climits>

#include "prof.h"
#include "tune.h"

namespace one {

enum MovePicker::Etape : int {
    P_TABLE, P_PRISES_GEN, P_BONNES_PRISES, P_REFUT, P_TRANQ_GEN, P_TRANQ, P_MAUVAISES, P_TRANQ_BAS,
    E_TABLE, E_GEN, E_PARADES,
    C_TABLE, C_GEN, C_PRISES,
    Q_TABLE, Q_GEN, Q_PRISES
};

namespace {

// Remonte en tete, tries par note decroissante, le premier coup et tous ceux
// dont la note atteint « plancher » ; le reste garde un ordre quelconque mais
// fixe (les egalites de note gardent leur ordre d'arrivee). Moins cher qu'un
// tri complet : la plupart des coups mal notes ne seront jamais joues.
void remonter(ExtMove* debut, ExtMove* fin, int plancher) {
    ExtMove* bord = debut;  // dernier element de la partie triee
    for (ExtMove* p = debut + 1; p < fin; ++p)
    {
        if (p->value < plancher)
            continue;
        const ExtMove x = *p;
        *p              = *++bord;  // le premier non trie prend la place liberee
        ExtMove* q      = bord;
        for (; q > debut && (q - 1)->value < x.value; --q)
            *q = *(q - 1);
        *q = x;
    }
}

}  // namespace

MovePicker::MovePicker(const Position&              p,
                       Move                         tt,
                       Depth                        d,
                       const ButterflyHistory*      principal,
                       const CapturePieceToHistory* prises,
                       const PieceToHistory**       suites,
                       const PawnHistory*           pions,
                       const Move*                  meurtriers,
                       Move                         reponse,
                       const LowPlyHistory*         bas,
                       int                          ply) :
    pos(p), hPrincipal(principal), hPrises(prises), hSuites(suites), hPions(pions), hBas(bas), plyBas(ply),
    coupTable(tt), prof(d) {

    refut[0] = meurtriers[0];
    refut[1] = meurtriers[1];
    // la reponse n'est gardee que si elle n'est pas deja un coup meurtrier
    refut[2] = reponse == meurtriers[0] || reponse == meurtriers[1] ? Move::none() : reponse;

    bool tableOk = tt && pos.pseudo_legal(tt);
    if (pos.checkers())
        etape = E_TABLE;
    else if (d > 0)
        etape = P_TABLE;
    else
    {
        etape = Q_TABLE;
        // QsFin bit 2 : hors echec, la quiescence ne reprend de la table
        // qu'une prise ou une promotion (sinon des echecs tranquilles sans fin)
        if (Tune::QsFin & 2)
            tableOk = tableOk && pos.capture_stage(tt);
    }
    if (!tableOk)
        ++etape;
}

MovePicker::MovePicker(const Position& p, Move tt, int s, const CapturePieceToHistory* prises) :
    pos(p), hPrises(prises), coupTable(tt), seuil(s) {
    assert(!pos.checkers());
    refut[0] = refut[1] = refut[2] = Move::none();
    const bool tableOk = tt && pos.capture_stage(tt) && pos.pseudo_legal(tt) && pos.see_ge(tt, seuil);
    etape              = C_TABLE + (tableOk ? 0 : 1);
}

// Notes de tri d'une tranche
template<GenType T>
void MovePicker::noter() {
    const Color moi = pos.side_to_move(), eux = ~moi;

    if constexpr (T == CAPTURES)
    {
        // La valeur de la piece prise decide ; l'historique des prises ne fait
        // que departager (s'il dominait, l'arbre gonflait sans rien gagner).
        for (ExtMove* m = cur; m < fin; ++m)
        {
            const Piece prise = pos.piece_on(m->to_sq());
            int         note  = 7 * PieceValue[prise] + (*hPrises)[pos.moved_piece(*m)][m->to_sq()][type_of(prise)] / Tune::PriseHistDiv;
            // En passant et promotion par poussee arrivent sur une case vide
            if (m->type_of() == EN_PASSANT)
                note += 7 * PawnValue;
            else if (m->type_of() == PROMOTION)
                note += 7 * (piece_value(make_piece(WHITE, m->promotion_type())) - PawnValue);
            m->value = note;
        }
        return;
    }

    // Menaces par une piece de moindre valeur (ThreatOrder) : pion sur mineure,
    // mineure sur tour, tour sur dame.
    Bitboard moindre[PIECE_TYPE_NB] = {};
    const bool ordreMenaces = T == QUIETS && Tune::ThreatOrder;
    if (ordreMenaces)
    {
        const Bitboard parPions = pos.attacks_by<PAWN>(eux);
        moindre[KNIGHT] = moindre[BISHOP] = parPions;
        moindre[ROOK]                     = parPions | pos.attacks_by<KNIGHT>(eux) | pos.attacks_by<BISHOP>(eux);
        moindre[QUEEN]                    = moindre[ROOK] | pos.attacks_by<ROOK>(eux);
    }
    // Cases menacees pour l'index de l'historique principal (reutilise le calcul ci-dessus)
    const Bitboard menaces = ordreMenaces && Tune::ThreatHist
                           ? moindre[QUEEN] | pos.attacks_by<QUEEN>(eux) | pos.attacks_by<KING>(eux)
                           : threat_map(pos);

    for (ExtMove* m = cur; m < fin; ++m)
    {
        const Piece  p = pos.moved_piece(*m);
        const Square a = m->to_sq();

        if constexpr (T == QUIETS)
        {
            // Poids des historiques en huitiemes (options OrdXxx, reglees par SPSA ;
            // defaut = 2, 2, 1, 1, 1, 1, 1 : la somme se divise exactement par 8)
            int note = Tune::OrdMain * (*hPrincipal)[moi][hist_index(*m, menaces)]
                     + Tune::OrdSuite1 * (*hSuites[0])[p][a] + Tune::OrdSuite2 * (*hSuites[1])[p][a]
                     + Tune::OrdSuite3 * (*hSuites[2])[p][a] + Tune::OrdSuite4 * (*hSuites[3])[p][a];
            if (Tune::ContHist6)
                note += Tune::OrdSuite6 * (*hSuites[5])[p][a];
            note += Tune::OrdPions * (*hPions)[pawn_structure_index(pos.pawn_key())][p][a];
            note /= 8;
            // (26-11, revue externe) l'historique des premiers plis etait tenu mais jamais lu ;
            // LowPly > 0 l'ajoute au tri des tranquilles pres de la racine (idee de Stockfish)
            if (Tune::LowPly && hBas && plyBas < LOW_PLY_HISTORY_SIZE)
                note += Tune::LowPly * (*hBas)[plyBas][m->from_to()] / (1 + plyBas);
            if (pos.check_squares(type_of(p)) & a)  // un echec se regarde tot
                note += Tune::OrdEchec;
            if (ordreMenaces)  // fuir une menace : bien ; s'y exposer : mal
            {
                const PieceType t = type_of(p);
                const int       f = int((moindre[t] & m->from_sq()) != 0) - int((moindre[t] & a) != 0);
                note += PieceValue[p] * Tune::ThreatOrder * f;
            }
            m->value = note;
        }
        else  // EVASIONS : prendre la piece qui donne echec d'abord
        {
            if (pos.capture(*m))
                m->value = (1 << 28) + PieceValue[pos.piece_on(a)] - int(type_of(p));
            else
                m->value = (*hPrincipal)[moi][hist_index(*m, menaces)] + (*hSuites[0])[p][a];
        }
    }
}

template<typename Test>
Move MovePicker::premier(Test accepte) {
    while (cur < fin)
    {
        ExtMove* m = cur++;
        if (*m != coupTable && accepte(m))
            return *m;
    }
    return Move::none();
}

Move MovePicker::next_move(bool sansTranquilles) {
    PROF_SCOPE(MOVEPICK);
    for (;;)
        switch (etape)
        {
        case P_TABLE :
        case E_TABLE :
        case C_TABLE :
        case Q_TABLE :
            ++etape;
            return coupTable;

        case P_PRISES_GEN :
        case C_GEN :
        case Q_GEN :
            cur = finMauvaises = tampon;
            fin                = generate<CAPTURES>(pos, tampon);
            noter<CAPTURES>();
            remonter(cur, fin, INT_MIN);
            ++etape;
            break;

        case P_BONNES_PRISES : {
            // une prise perdante est mise de cote pour plus tard
            const Move m = premier([this](ExtMove* x) {
                if (pos.see_ge(*x))
                    return true;
                *finMauvaises++ = *x;
                return false;
            });
            if (m)
                return m;
            ++etape;
            break;
        }

        case P_REFUT :
            while (refutSuivante < 3)
            {
                const Move m = refut[refutSuivante++];
                if (m && m != coupTable && !pos.capture_stage(m) && pos.pseudo_legal(m))
                    return m;
            }
            ++etape;
            break;

        case P_TRANQ_GEN :
            if (!sansTranquilles)
            {
                cur = finMauvaises;
                fin = generate<QUIETS>(pos, cur);
                noter<QUIETS>();
                remonter(cur, fin, -Tune::TriPlancher * prof);
            }
            ++etape;
            break;

        case P_TRANQ :
            if (!sansTranquilles)
            {
                const int  bas = Tune::BadQuiets;
                const Move m   = premier([&](ExtMove* x) { return !est_refutation(*x) && (!bas || x->value > -bas); });
                if (m)
                    return m;
            }
            // on garde la tranche des tranquilles pour P_TRANQ_BAS, et on
            // revient aux mauvaises prises en tete du tampon
            debTranq = finMauvaises;
            finTranq = fin;
            cur      = tampon;
            fin      = finMauvaises;
            ++etape;
            break;

        case P_MAUVAISES : {
            const Move m = premier([](ExtMove*) { return true; });
            if (m || !Tune::BadQuiets || sansTranquilles)
                return m;
            cur = debTranq;
            fin = finTranq;
            ++etape;
            break;
        }

        case P_TRANQ_BAS :
            if (sansTranquilles)
                return Move::none();
            return premier([&](ExtMove* x) { return x->value <= -Tune::BadQuiets && !est_refutation(*x); });

        case E_GEN :
            cur = tampon;
            fin = generate<EVASIONS>(pos, tampon);
            noter<EVASIONS>();
            remonter(cur, fin, INT_MIN);
            ++etape;
            break;

        case E_PARADES :
        case Q_PRISES :
            return premier([](ExtMove*) { return true; });

        case C_PRISES :
            return premier([this](ExtMove* x) { return pos.see_ge(*x, seuil); });

        default :
            assert(false);
            return Move::none();
        }
}

}  // namespace one
