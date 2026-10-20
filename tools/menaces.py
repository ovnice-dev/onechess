"""Entrees « menaces » pour l'entrainement NNUE (calcul sur le GPU).

Une entree par piece attaquee par le camp adverse : (piece relative a la
perspective, case relative). 768 entrees, peu actives a la fois (une poignee
de pieces sont en prise ou defendues par l'echange). Le moteur calcule la meme
chose avec ses tables d'attaques ; tools/check_nnue.py le verifie.

Plateau : tenseurs (B, 8, 8) booleens, rangee 0 = rangee 1, colonne 0 = a.
"""
import torch

THREAT_INPUTS = 768
THREAT_FLAG   = 512      # bit du champ « activation » de l'en-tete .nnue
MAX_THREATS   = 32

_DIRS_R = ((1, 0), (-1, 0), (0, 1), (0, -1))     # tour : (rangee, colonne)
_DIRS_B = ((1, 1), (1, -1), (-1, 1), (-1, -1))   # fou
_KNIGHT = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
_KING   = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))


def _shift(b, dr, dc):
    """Decale un plateau (B, 8, 8) de (dr, dc), sans enroulement."""
    out = torch.zeros_like(b)
    r0, r1 = max(dr, 0), 8 + min(dr, 0)
    c0, c1 = max(dc, 0), 8 + min(dc, 0)
    out[:, r0:r1, c0:c1] = b[:, r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


def _slide(src, occ, dirs):
    att = torch.zeros_like(src)
    for dr, dc in dirs:
        f = src
        for _ in range(7):
            f = _shift(f, dr, dc)
            att |= f
            f = f & ~occ
            if not f.any():
                break
    return att


def attacks(boards, colour):
    """Cases attaquees par le camp colour (0 blancs, 1 noirs).
    boards : (B, 12, 8, 8), plans dans l'ordre des codes (blancs P N B R Q K,
    puis noirs)."""
    occ = boards.any(dim=1)
    b   = boards[:, 6 * colour:6 * colour + 6]
    fwd = 1 if colour == 0 else -1
    att = _shift(b[:, 0], fwd, 1) | _shift(b[:, 0], fwd, -1)
    for dr, dc in _KNIGHT:
        att |= _shift(b[:, 1], dr, dc)
    for dr, dc in _KING:
        att |= _shift(b[:, 5], dr, dc)
    att |= _slide(b[:, 2] | b[:, 4], occ, _DIRS_B)
    att |= _slide(b[:, 3] | b[:, 4], occ, _DIRS_R)
    return att


def threat_indices(codes, sqs, persp, att_cache=None):
    """(B, K) : indices des pieces attaquees par leur adversaire, vus de persp,
    completes par THREAT_INPUTS. att_cache : (att_blancs, att_noirs) deja
    calcules pour ce lot (les deux perspectives les partagent)."""
    B, N = codes.shape
    valid = codes < 12
    if att_cache is None:
        boards = torch.zeros(B, 12, 64, dtype=torch.bool, device=codes.device)
        bi = torch.arange(B, device=codes.device).unsqueeze(1).expand(B, N)
        boards[bi[valid], codes[valid], sqs[valid]] = True
        boards = boards.view(B, 12, 8, 8)
        att_cache = (attacks(boards, 0).view(B, 64), attacks(boards, 1).view(B, 64))
    colour = torch.div(codes, 6, rounding_mode="floor").clamp(max=1)
    # une piece est menacee si le camp ADVERSE attaque sa case
    att_w, att_b = att_cache
    sq = sqs.clamp(0, 63)
    by_enemy = torch.where(colour == 0, att_b.gather(1, sq), att_w.gather(1, sq))
    hit = valid & by_enemy & (codes % 6 != 5)          # le roi en echec n'est pas une « menace »
    pidx = (colour != persp).long() * 6 + codes % 6
    rel  = sqs ^ (56 * persp)
    idx  = torch.where(hit, pidx * 64 + rel, THREAT_INPUTS)
    k    = int(hit.sum(dim=1).max().item()) if B else 0
    idx, _ = torch.sort(idx, dim=1)
    return idx[:, :max(k, 1)], att_cache
