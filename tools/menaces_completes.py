"""Entrees "menaces completes" (Full Threats), reference en Python (phase 10, C).

Reprend la definition de Stockfish 18-19 (src/nnue/features/full_threats.cpp,
GPL-3, comme Onechess) : une entree par paire (attaquant, case de depart, piece
attaquee, case visee), vue de chaque camp. 59 808 entrees, tres creuses
(quelques dizaines actives par position).
  - le roi n'est ni attaquant ni cible ;
  - un pion n'entre que s'il attaque un cavalier ou une tour (pion contre pion :
    paires de pions, a part) ;
  - cavalier, dame : cibles pion, cavalier, fou, tour, dame ; fou, tour : sans la dame ;
  - deux pieces du meme type (meme camp pour les pions exceptes) : la paire n'est
    comptee que dans un sens (case de depart < case visee), sinon elle le serait deux fois.
Difference avec Stockfish : pas de miroir selon la colonne du roi (nos reseaux
n'en ont pas, miroir=0) ; seule la perspective retourne l'echiquier (^ 56).

Sert de juge a l'entraineur (calcul des indices) et au moteur (mises a jour
incrementales) : memes indices sur les memes positions.
    python tools/menaces_completes.py          # auto-verification
"""
import chess

DIMENSIONS = 59808
# Cibles valides par type d'attaquant (pion, cavalier, fou, tour, dame, roi), deux couleurs comprises
NB_CIBLES = {chess.PAWN: 4, chess.KNIGHT: 10, chess.BISHOP: 8, chess.ROOK: 8, chess.QUEEN: 10, chess.KING: 0}
# CARTE[attaquant-1][cible-1] : rang de la cible parmi celles de l'attaquant (-1 : exclue)
CARTE = [[-1, 0, -1, 1, -1, -1],
         [0, 1, 2, 3, 4, -1],
         [0, 1, 2, 3, -1, -1],
         [0, 1, 2, 3, -1, -1],
         [0, 1, 2, 3, 4, -1],
         [-1, -1, -1, -1, -1, -1]]
TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PIECES = [(c, t) for c in (chess.WHITE, chess.BLACK) for t in TYPES]   # ordre de Stockfish : blancs puis noirs


def _attaques_vides(t, sq, couleur_pion=None):
    """Attaques sur echiquier vide (PseudoAttacks de Stockfish)."""
    b = chess.BaseBoard.empty()
    b.set_piece_at(sq, chess.Piece(t, couleur_pion if t == chess.PAWN else chess.WHITE))
    return int(b.attacks_mask(sq))


def _tables():
    lut2 = {}
    decal = {}
    for t in TYPES:
        off = 0
        for fr in range(64):
            for to in range(64):
                if t != chess.PAWN:
                    att = _attaques_vides(t, fr)
                else:
                    att = _attaques_vides(t, fr, chess.BLACK if to < fr else chess.WHITE)
                lut2[(t, fr, to)] = off + bin(((1 << to) - 1) & att).count('1')
            if t != chess.PAWN:
                off += bin(_attaques_vides(t, fr)).count('1')
            elif chess.A2 <= fr <= chess.H7:
                off += bin(_attaques_vides(t, fr, chess.WHITE)).count('1')
        decal[t] = off
    lut1 = {}
    cumul = 0
    for (ca, ta) in PIECES:
        cp = decal[ta]
        nv = NB_CIBLES[ta]
        for (cd, td) in PIECES:
            ennemi = ca != cd
            m = CARTE[ta - 1][td - 1]
            semi = ta == td and (ennemi or ta != chess.PAWN)
            f = cumul + ((0 if cd == chess.WHITE else 1) * (nv // 2) + m) * cp
            exclu = m < 0
            lut1[((ca, ta), (cd, td), 0)] = DIMENSIONS if exclu else f
            lut1[((ca, ta), (cd, td), 1)] = DIMENSIONS if exclu or semi else f
        cumul += nv * cp
    assert cumul == DIMENSIONS, cumul
    return lut1, lut2


LUT1, LUT2 = _tables()


def indice(persp, attaquant, fr, to, attaque):
    """attaquant, attaque : (couleur, type) ; persp : chess.WHITE / chess.BLACK."""
    o = 0 if persp == chess.WHITE else 56
    fo, too = fr ^ o, to ^ o
    if persp == chess.BLACK:   # couleurs echangees, comme "^ 8" sur les pieces de Stockfish
        attaquant = (not attaquant[0], attaquant[1])
        attaque = (not attaque[0], attaque[1])
    return LUT1[(attaquant, attaque, int(fo < too))] + LUT2[(attaquant[1], fo, too)]


def indices_actifs(board, persp):
    """Indices actifs (tries) d'une position, vue du camp persp."""
    act = []
    occ = board.occupied
    for sq, p in board.piece_map().items():
        if p.piece_type == chess.KING:
            continue
        att = int(board.attacks_mask(sq)) if p.piece_type != chess.PAWN else int(chess.BB_PAWN_ATTACKS[p.color][sq])
        bb = att & occ
        while bb:
            to = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            q = board.piece_at(to)
            i = indice(persp, (p.color, p.piece_type), sq, to, (q.color, q.piece_type))
            if i < DIMENSIONS:
                act.append(i)
    return sorted(act)


def _verifier():
    import random
    random.seed(1)
    n, maxi, total = 0, 0, 0
    for partie in range(200):
        b = chess.Board()
        for _ in range(random.randint(0, 120)):
            coups = list(b.legal_moves)
            if not coups or b.is_game_over():
                break
            b.push(random.choice(coups))
        for persp in (chess.WHITE, chess.BLACK):
            a = indices_actifs(b, persp)
            assert len(a) == len(set(a)), "indice en double"
            assert all(0 <= i < DIMENSIONS for i in a)
            # symetrie : le camp noir voit ce que verrait le blanc sur l'echiquier retourne
            assert a == indices_actifs(b.mirror(), not persp), "symetrie rompue"
            maxi = max(maxi, len(a)); total += len(a); n += 1
    print(f"OK : {DIMENSIONS} entrees ; {n} vues, {total / n:.1f} actives en moyenne, {maxi} au plus")


_LIB = None


def _lib():
    """Charge (et compile au besoin) l'indexeur C++ tools/menaces/menaces.cpp."""
    global _LIB
    if _LIB is None:
        import ctypes, os, subprocess, sys
        ici = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'menaces')
        nom = os.path.join(ici, 'libmenaces' + ('.dll' if sys.platform == 'win32' else '.so'))
        src = os.path.join(ici, 'menaces.cpp')
        if not os.path.exists(nom) or os.path.getmtime(nom) < os.path.getmtime(src):
            cmd = ['g++', '-O3', '-std=c++17', '-shared', '-pthread', src, '-o', nom]
            if sys.platform != 'win32':
                cmd.insert(4, '-fPIC')
            else:
                cmd += ['-static']
            subprocess.run(cmd, check=True)
        _LIB = ctypes.CDLL(nom)
    return _LIB


def lot(recs, K=64, fils=0):
    """recs : tableau numpy (n, 32) uint8 -> (w, b) : (n, K) uint16 completes par DIMENSIONS."""
    import ctypes
    import numpy as np
    recs = np.ascontiguousarray(recs, dtype=np.uint8)
    n = recs.shape[0]
    w = np.empty((n, K), np.uint16)
    b = np.empty((n, K), np.uint16)
    p = ctypes.c_void_p
    m = _lib().menaces_lot(p(recs.ctypes.data), ctypes.c_long(n), K, p(w.ctypes.data), p(b.ctypes.data), fils)
    return w, b, m


def plateau(r):
    """Position de 32 octets -> chess.Board (le trait ne compte pas pour les menaces)."""
    occ = int.from_bytes(bytes(r[:8]), 'little')
    b = chess.Board(None)
    n = 0
    for sq in range(64):
        if occ >> sq & 1:
            q = (r[8 + n // 2] >> (4 * (n & 1))) & 15
            b.set_piece_at(sq, chess.Piece(q % 6 + 1, chess.WHITE if q < 6 else chess.BLACK))
            n += 1
    return b


def _verifier_cpp(fichier, n=20000):
    import numpy as np, time
    recs = np.fromfile(fichier, dtype=np.uint8).reshape(-1, 32)[:n]
    t = time.time()
    w, b, m = lot(recs)
    dt = time.time() - t
    for i in range(min(n, 3000)):
        bd = plateau(recs[i])
        rw = indices_actifs(bd, chess.WHITE)
        rb = indices_actifs(bd, chess.BLACK)
        cw = sorted(int(x) for x in w[i] if x < DIMENSIONS)
        cb = sorted(int(x) for x in b[i] if x < DIMENSIONS)
        assert cw == rw and cb == rb, f"ecart en position {i}"
    print(f"C++ = reference sur {min(n, 3000)} positions ; {len(recs) / dt:,.0f} positions/s ; "
          f"au plus {m} paires actives")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == '--cpp':
        _verifier_cpp(sys.argv[2])
    else:
        _verifier()
