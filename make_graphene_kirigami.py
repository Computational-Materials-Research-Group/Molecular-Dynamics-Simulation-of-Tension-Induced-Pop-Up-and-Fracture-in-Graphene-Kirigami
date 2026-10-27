#!/usr/bin/env python3
"""
Graphene kirigami sheet -> LAMMPS data file (units metal, atom_style atomic).
Same brick-wall parallel-cut pattern as the CalculiX model, at nanometre scale.
  x : zigzag direction (cuts run along x)
  y : armchair direction (loading direction)
Usage:  python3 make_graphene_kirigami.py          -> graphene_kirigami.data
        python3 make_graphene_kirigami.py N_COLS=3 N_ROWS=10 PASSIVATE_H=True   (override parameters)
"""
import numpy as np

# ------------------------------------------------------------------ parameters (Angstrom)
B        = 1.397   # C-C bond length (AIREBO equilibrium for graphene)
N_COLS   = 2       # unit cells across the width
CUT_L    = 60.0    # cut length
LIG      = 15.0    # hinge ligament between cut tips
CUT_W    = 5.0     # cut (gap) width; removes ~2 atom rows
ROW      = 25.0    # row pitch along the loading direction
N_ROWS   = 8       # number of cut rows
GRIP     = 15.0    # uncut grip length at each end
PASSIVATE_H = False  # True: add H atoms to every dangling edge bond (C-H 1.09 A)
OUT      = "graphene_kirigami.data"

def build():
    W = N_COLS * (CUT_L + LIG)
    L = 2 * GRIP + N_ROWS * ROW
    ax, ay = np.sqrt(3) * B, 3 * B                       # rectangular 4-atom cell
    basis = np.array([[0, 0], [ax / 2, B / 2], [ax / 2, 1.5 * B], [0, 2 * B]])
    sub = np.array([0, 1, 0, 1])                         # sublattice A/B
    nx, ny = int(np.floor(W / ax)), int(np.floor(L / ay))
    pts, sl = [], []
    for i in range(nx):
        for j in range(ny):
            pts.append(basis + [i * ax, j * ay]); sl.append(sub)
    P, S = np.vstack(pts), np.concatenate(sl)
    W, L = nx * ax, ny * ay                              # snap to lattice

    # ---- cuts: brick-wall pattern, rows offset by half a cell
    cell = CUT_L + LIG
    keep = np.ones(len(P), bool)
    for r in range(N_ROWS):
        yc = GRIP + (r + 0.5) * ROW
        shift = 0.0 if r % 2 == 0 else cell / 2
        for c in range(-1, N_COLS + 1):
            x0 = c * cell + shift + LIG / 2
            m = (P[:, 0] > x0) & (P[:, 0] < x0 + CUT_L) & (np.abs(P[:, 1] - yc) < CUT_W / 2)
            keep &= ~m
    P, S = P[keep], S[keep]

    # ---- remove atoms with fewer than 2 C neighbours (dangling chains), repeat
    while True:
        nb = neighbour_count(P)
        bad = nb < 2
        if not bad.any():
            break
        P, S = P[~bad], S[~bad]

    atoms = [(1, x, y, 0.0) for x, y in P]
    if PASSIVATE_H:
        atoms += passivate(P, S)
    return np.array(atoms), W, L

def neighbour_count(P, rc=1.6):
    from collections import defaultdict
    grid = defaultdict(list)
    key = np.floor(P / rc).astype(int)
    for n, k in enumerate(map(tuple, key)):
        grid[k].append(n)
    cnt = np.zeros(len(P), int)
    for n, (kx, ky) in enumerate(key):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for m in grid.get((kx + dx, ky + dy), ()):
                    if m != n and np.hypot(*(P[m] - P[n])) < rc:
                        cnt[n] += 1
    return cnt

def passivate(P, S, dCH=1.09):
    """One H on every missing bond direction of an edge carbon."""
    dirs = np.array([[0, 1], [np.sqrt(3) / 2, -0.5], [-np.sqrt(3) / 2, -0.5]])   # sublattice A
    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    H = []
    for n, (p, s) in enumerate(zip(P, S)):
        dset = -dirs if s == 0 else dirs      # bond directions of this sublattice
        for d in dset:
            if not tree.query_ball_point(p + B * d, 0.3):
                q = p + dCH * d
                # skip sites that clash with another C (across a narrow gap) or an earlier H
                if len(tree.query_ball_point(q, 1.3)) > 1:
                    continue
                if any(np.hypot(q[0] - h[1], q[1] - h[2]) < 1.5 for h in H[-12:]):
                    continue
                H.append((2, q[0], q[1], 0.0))
    return H

def write(atoms, W, L):
    pad = 50.0
    with open(OUT, "w") as f:
        f.write(f"Graphene kirigami {W:.1f} x {L:.1f} A, cuts {CUT_L}/{LIG}/{ROW}\n\n")
        f.write(f"{len(atoms)} atoms\n{2} atom types\n\n")
        f.write(f"{-pad:.4f} {W + pad:.4f} xlo xhi\n{-pad:.4f} {L + pad:.4f} ylo yhi\n"
                f"{-pad:.4f} {pad:.4f} zlo zhi\n\nMasses\n\n1 12.011\n2 1.008\n\nAtoms # atomic\n\n")
        for i, (t, x, y, z) in enumerate(atoms, 1):
            f.write(f"{i} {int(t)} {x:.6f} {y:.6f} {z:.6f}\n")
    nC, nH = int((atoms[:, 0] == 1).sum()), int((atoms[:, 0] == 2).sum())
    print(f"wrote {OUT}: {nC} C + {nH} H atoms, sheet {W:.1f} x {L:.1f} A "
          f"({W/10:.1f} x {L/10:.1f} nm), grips {GRIP} A")

if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        k, v = arg.split("=")
        old = globals()[k]
        globals()[k] = (v.lower() in ("1", "true", "yes")) if isinstance(old, bool) else type(old)(v)
    atoms, W, L = build()
    write(atoms, W, L)
