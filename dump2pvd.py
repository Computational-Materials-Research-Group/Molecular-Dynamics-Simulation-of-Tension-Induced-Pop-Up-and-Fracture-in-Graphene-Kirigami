#!/usr/bin/env python3
"""LAMMPS dump.kirigami -> ParaView pvd/kirigami.pvd (one .vtp per frame, atoms + bonds).
Usage: python3 dump2pvd.py [dump.kirigami] [outdir]
Bonds are drawn for C-C < 1.9 A and C-H < 1.3 A, so broken bonds disappear in the animation.
Point data: type, uz (out-of-plane displacement, A), pe (eV), syy (approx. per-atom stress, GPa)."""
import sys, os
import numpy as np
from scipy.spatial import cKDTree

DUMP = sys.argv[1] if len(sys.argv) > 1 else "dump.kirigami"
OUT = sys.argv[2] if len(sys.argv) > 2 else "pvd"
VATOM = 2.62 * 3.35          # volume per C atom in graphene (area x 3.35 A thickness), A^3
RCC, RCH = 1.9, 1.3

def frames(path):
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                return
            if line.startswith("ITEM: TIMESTEP"):
                step = int(f.readline())
                f.readline(); n = int(f.readline())
                f.readline(); [f.readline() for _ in range(3)]
                cols = f.readline().split()[2:]
                data = np.loadtxt([f.readline() for _ in range(n)])
                yield step, cols, data

def write_vtp(fn, xyz, typ, arrays, bonds):
    with open(fn, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n<PolyData>\n')
        f.write(f'<Piece NumberOfPoints="{len(xyz)}" NumberOfVerts="{len(xyz)}" NumberOfLines="{len(bonds)}">\n')
        f.write('<PointData Scalars="uz">\n')
        f.write('<DataArray type="Int32" Name="type" format="ascii">\n' + " ".join(map(str, typ)) + "\n</DataArray>\n")
        for name, a in arrays.items():
            f.write(f'<DataArray type="Float32" Name="{name}" format="ascii">\n'
                    + " ".join(f"{v:.5g}" for v in a) + "\n</DataArray>\n")
        f.write('</PointData>\n<Points>\n<DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        f.write("\n".join(f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in xyz) + "\n</DataArray>\n</Points>\n")
        n = len(xyz)
        f.write('<Verts>\n<DataArray type="Int32" Name="connectivity" format="ascii">\n'
                + " ".join(map(str, range(n))) + '\n</DataArray>\n<DataArray type="Int32" Name="offsets" format="ascii">\n'
                + " ".join(map(str, range(1, n + 1))) + "\n</DataArray>\n</Verts>\n")
        f.write('<Lines>\n<DataArray type="Int32" Name="connectivity" format="ascii">\n'
                + " ".join(f"{i} {j}" for i, j in bonds) + '\n</DataArray>\n<DataArray type="Int32" Name="offsets" format="ascii">\n'
                + " ".join(str(2 * (k + 1)) for k in range(len(bonds))) + "\n</DataArray>\n</Lines>\n")
        f.write("</Piece>\n</PolyData>\n</VTKFile>\n")

os.makedirs(OUT, exist_ok=True)
entries = []
for k, (step, cols, d) in enumerate(frames(DUMP)):
    c = {name: d[:, i] for i, name in enumerate(cols)}
    xyz = np.column_stack([c["x"], c["y"], c["z"]])
    typ = c["type"].astype(int)
    pairs = cKDTree(xyz).query_pairs(RCC, output_type="ndarray")
    if len(pairs):
        r = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
        hasH = (typ[pairs[:, 0]] == 2) | (typ[pairs[:, 1]] == 2)
        bothH = (typ[pairs[:, 0]] == 2) & (typ[pairs[:, 1]] == 2)
        pairs = pairs[~bothH & ((~hasH) | (r < RCH))]
    arrays = {"uz": c["c_dsp[3]"], "pe": c["c_pe"], "syy": c["c_sat[2]"] / VATOM * 1e-4}
    fn = f"kirigami_{k:04d}.vtp"
    write_vtp(os.path.join(OUT, fn), xyz, typ, arrays, pairs)
    entries.append((step, fn))
    print(f"frame {k:4d}  step {step:8d}  atoms {len(xyz)}  bonds {len(pairs)}")
with open(os.path.join(OUT, "kirigami.pvd"), "w") as f:
    f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n<Collection>\n')
    for step, fn in entries:
        f.write(f'  <DataSet timestep="{step}" group="" part="0" file="{fn}"/>\n')
    f.write("</Collection>\n</VTKFile>\n")
print(f"wrote {OUT}/kirigami.pvd with {len(entries)} frames")
