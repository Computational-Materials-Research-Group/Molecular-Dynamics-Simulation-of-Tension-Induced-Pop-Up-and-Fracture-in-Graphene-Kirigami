#!/bin/bash
# ============================================================
#  Graphene kirigami, AIREBO, uniaxial stretch -> ParaView .pvd
#
#  One-time install (Ubuntu 24.04 / WSL):
#     sudo apt install lammps openmpi-bin python3-numpy python3-scipy python3-matplotlib
#
#  Run:   bash run_lammps.sh
# ============================================================
set -e
NP=14                                   # MPI ranks (check with: nproc)
export OMP_NUM_THREADS=1

echo ">>> 1/4  Building graphene kirigami sheet"
python3 make_graphene_kirigami.py

echo ">>> 2/4  LAMMPS (relax, 300 K equilibration, stretch) - progress: tail -f log.lammps"
mpirun -np $NP lmp -in in.kirigami > lammps_screen.txt
grep -E "gauge length|stretching:" log.lammps

echo ">>> 3/4  Converting dump to ParaView (atoms + bonds)"
python3 dump2pvd.py dump.kirigami pvd

echo ">>> 4/4  Stress-strain and pop-up curves"
python3 plot_kirigami.py

echo "Done. Open pvd/kirigami.pvd in ParaView and press Play."
