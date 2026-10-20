#include "bitboard.h"

#include <initializer_list>

namespace one {

std::uint8_t SquareDistance[SQUARE_NB][SQUARE_NB];

Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
Bitboard LineBB[SQUARE_NB][SQUARE_NB];
Bitboard PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];
Bitboard PawnAttacksBB[COLOR_NB][SQUARE_NB];

Magic RookMagics[SQUARE_NB];
Magic BishopMagics[SQUARE_NB];

namespace {

Bitboard RookTable[0x19000];   // 102400 entrees
Bitboard BishopTable[0x1480];  //   5248 entrees

// Generateur xorshift64* : suffisant pour la recherche de magics, et
// deterministe d'une execution a l'autre.
class PRNG {
    std::uint64_t s;

    std::uint64_t rand64() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }

   public:
    explicit PRNG(std::uint64_t seed) : s(seed) { assert(seed); }

    std::uint64_t rand() { return rand64(); }

    // Peu de bits a 1 : les bons magics en ont typiquement 6 a 8.
    std::uint64_t sparse_rand() { return rand64() & rand64() & rand64(); }
};

// Attaques d'une piece glissante, calculees case par case en s'arretant sur
// le premier bloqueur (inclus). Lent, mais n'est utilise qu'a l'init.
Bitboard sliding_attack(const std::initializer_list<Direction>& directions,
                        Square                                  sq,
                        Bitboard                                occupied) {
    Bitboard attack = 0;

    for (Direction d : directions)
        for (Square s = sq + d; is_ok(s) && distance(s, s - d) == 1; s += d)
        {
            attack |= s;
            if (occupied & s)
                break;
        }

    return attack;
}

void init_magics(PieceType pt, Bitboard table[], Magic magics[]) {
    const std::initializer_list<Direction> rookDirections   = {NORTH, EAST, SOUTH, WEST};
    const std::initializer_list<Direction> bishopDirections = {NORTH_EAST, SOUTH_EAST,
                                                               SOUTH_WEST, NORTH_WEST};

    // Graines par rangee : reprises des valeurs qui font converger la
    // recherche rapidement sur toutes les cases.
    constexpr int seeds[RANK_NB] = {728, 10316, 55013, 32803, 12281, 15100, 16645, 255};

    Bitboard occupancy[4096];
    Bitboard reference[4096];
    int      epoch[4096] = {};
    int      cnt         = 0;
    int      size        = 0;

    for (Square s = SQ_A1; s <= SQ_H8; ++s)
    {
        // Les cases de bord ne sont jamais des bloqueurs pertinents : une
        // piece posee sur le bord ne cache rien derriere elle.
        Bitboard edges = ((Rank1BB | Rank8BB) & ~rank_bb(s)) | ((FileABB | FileHBB) & ~file_bb(s));

        Magic& m = magics[s];
        m.mask   = sliding_attack(pt == ROOK ? rookDirections : bishopDirections, s, 0) & ~edges;

#if !defined(USE_PEXT)
        m.shift = 64 - popcount(m.mask);
#endif

        // Chaque case reprend la table la ou la precedente s'est arretee.
        m.attacks = s == SQ_A1 ? table : magics[s - 1].attacks + size;

        // Enumeration de tous les sous-ensembles du masque (Carry-Rippler).
        Bitboard b = 0;
        size       = 0;
        do
        {
            occupancy[size] = b;
            reference[size] = sliding_attack(pt == ROOK ? rookDirections : bishopDirections, s, b);

#if defined(USE_PEXT)
            m.attacks[_pext_u64(b, m.mask)] = reference[size];
#endif
            size++;
            b = (b - m.mask) & m.mask;
        } while (b);

#if !defined(USE_PEXT)
        PRNG rng(seeds[rank_of(s)]);

        // Cherche un multiplicateur qui n'entre en collision que sur des
        // ensembles d'attaques identiques.
        for (int i = 0; i < size;)
        {
            for (m.magic = 0; popcount((m.magic * m.mask) >> 56) < 6;)
                m.magic = rng.sparse_rand();

            // epoch[] evite de reinitialiser attacks[] a chaque essai.
            for (++cnt, i = 0; i < size; ++i)
            {
                unsigned idx = m.index(occupancy[i]);

                if (epoch[idx] < cnt)
                {
                    epoch[idx]     = cnt;
                    m.attacks[idx] = reference[i];
                }
                else if (m.attacks[idx] != reference[i])
                    break;
            }
        }
#else
        (void) seeds;
        (void) occupancy;
        (void) reference;
        (void) epoch;
        (void) cnt;
#endif
    }
}

}  // namespace

std::string Bitboards::pretty(Bitboard b) {
    std::string s = "+---+---+---+---+---+---+---+---+\n";

    for (Rank r = RANK_8; r >= RANK_1; --r)
    {
        for (File f = FILE_A; f <= FILE_H; ++f)
            s += b & make_square(f, r) ? "| X " : "|   ";

        s += "| " + std::to_string(1 + r) + "\n+---+---+---+---+---+---+---+---+\n";
    }
    s += "  a   b   c   d   e   f   g   h\n";

    return s;
}

void Bitboards::init() {
    for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1)
        for (Square s2 = SQ_A1; s2 <= SQ_H8; ++s2)
            SquareDistance[s1][s2] = std::max(distance<File>(s1, s2), distance<Rank>(s1, s2));

    init_magics(ROOK, RookTable, RookMagics);
    init_magics(BISHOP, BishopTable, BishopMagics);

    for (Square s = SQ_A1; s <= SQ_H8; ++s)
    {
        PawnAttacksBB[WHITE][s] = pawn_attacks_bb<WHITE>(square_bb(s));
        PawnAttacksBB[BLACK][s] = pawn_attacks_bb<BLACK>(square_bb(s));

        for (int step : {-9, -8, -7, -1, 1, 7, 8, 9})
            PseudoAttacks[KING][s] |= safe_destination(s, step);

        for (int step : {-17, -15, -10, -6, 6, 10, 15, 17})
            PseudoAttacks[KNIGHT][s] |= safe_destination(s, step);

        PseudoAttacks[QUEEN][s] = PseudoAttacks[BISHOP][s] = attacks_bb<BISHOP>(s, 0);
        PseudoAttacks[QUEEN][s] |= PseudoAttacks[ROOK][s] = attacks_bb<ROOK>(s, 0);

        for (PieceType pt : {BISHOP, ROOK})
            for (Square s2 = SQ_A1; s2 <= SQ_H8; ++s2)
            {
                if (!(PseudoAttacks[pt][s] & s2))
                    continue;

                LineBB[s][s2] = (attacks_bb(pt, s, 0) & attacks_bb(pt, s2, 0)) | s | s2;
                BetweenBB[s][s2] =
                  (attacks_bb(pt, s, square_bb(s2)) & attacks_bb(pt, s2, square_bb(s)));
            }

        // Convention : between_bb inclut toujours la case d'arrivee, alignee
        // ou non. Le masque "bloquer ou capturer" tombe alors directement.
        for (Square s2 = SQ_A1; s2 <= SQ_H8; ++s2)
            BetweenBB[s][s2] |= s2;
    }
}

}  // namespace one
