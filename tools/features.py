#!/usr/bin/env python3
"""Jeux de caracteristiques NNUE, definis une seule fois.

Le moteur C++ et l'entrainement doivent indexer les caracteristiques
exactement de la meme facon. Tout est donc decrit ici, et `tools/check_nnue.py`
verifie que les deux implementations concordent au centipion pres.

Forme generale de l'index, vu par la perspective p :

    bucket(roi de p) * 768  +  pieceIdx * 64  +  case

  - pieceIdx = 6 * (la piece appartient a l'adversaire de p) + (type - 1)
  - la case est retournee verticalement pour la perspective des noirs, ce qui
    permet aux deux perspectives de partager les memes poids ;
  - avec le miroir horizontal, elle est en plus retournee gauche-droite quand
    le roi de p est sur les colonnes e a h, ce qui replie l'echiquier sur sa
    moitie dame et double le nombre d'exemples par poids.

Le decoupage en buckets donne au reseau ce que 768 entrees seules ne peuvent
pas exprimer : la position d'une piece RELATIVEMENT au roi. C'est la
difference entre un reseau qui plafonne et un reseau qui progresse.
"""

# Un bucket par case de roi, apres repliement sur les colonnes a-d.
# 0 = pas de decoupage, tout partage les memes poids.
BUCKET_LAYOUTS = {
    "plain": None,

    # Quatre zones : roque cote roi, centre, roque cote dame, et les rangees
    # avancees ou le roi ne devrait pas se trouver.
    "k4": [
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3,
    ],

    # Huit zones : on distingue en plus les deux premieres rangees.
    "k8": [
        0, 0, 1, 1, 1, 1, 2, 2,
        3, 3, 4, 4, 4, 4, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7,
    ],

    # Une entree par case de roi : le maximum d'information, au prix d'un
    # reseau bien plus gros et donc bien plus gourmand en donnees.
    "k32": list(range(64)),
}


# Bucket de SORTIE : plusieurs jeux de poids de sortie, choisis selon le
# materiel restant. Une finale et un milieu de partie n'ont pas la meme facon
# de convertir une configuration en centipions ; leur imposer les memes poids
# force le reseau a un compromis.
#
# Contrairement aux buckets de roi, ceux-ci ne multiplient QUE la derniere
# couche (quelques milliers de poids au lieu de millions) : ils paient donc
# des le volume de donnees dont on dispose.
#
# Cette formule doit rester identique au C++ (nnue.cpp).
def output_bucket(piece_count, n_buckets):
    return (piece_count - 2) // (32 // n_buckets)


def bucket_count(name, mirror):
    layout = BUCKET_LAYOUTS[name]
    if layout is None:
        return 1
    if name == "k32":
        # Avec miroir le roi est toujours colonnes a-d : 32 cases utiles
        return 32 if mirror else 64
    return max(layout) + 1


def compact_layout(name, mirror):
    """Rend la table de 64 cases -> index de bucket effectivement utilisee."""
    layout = BUCKET_LAYOUTS[name]
    if layout is None:
        return [0] * 64
    if name != "k32":
        return list(layout)

    # k32 : numerotation dense des cases repliees
    out = [0] * 64
    if mirror:
        for sq in range(64):
            f, r = sq & 7, sq >> 3
            fm = f if f < 4 else 7 - f
            out[sq] = r * 4 + fm
    else:
        out = list(range(64))
    return out
