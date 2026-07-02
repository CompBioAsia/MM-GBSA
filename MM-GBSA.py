from openmm import LangevinMiddleIntegrator

from openmm.unit import nanometer, kilojoule, mole, kelvin, picosecond
from openmm.app import HCT, NoCutoff, Simulation
from parmed.amber import AmberParm
import mdtraj as mdt
import numpy as np
import tqdm

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='MM-GBSA calculation')
    parser.add_argument('--top', type=str,  help='Amber topology file')
    parser.add_argument('--traj', type=str, help='Trajectory file')
    parser.add_argument('--ligand', type=str, help='Ligand residue name')
    return parser.parse_args()

args = parse_args()
t = mdt.load(args.traj, top=args.top)
receptor_atoms = t.topology.select('protein')
ligand_atoms = t.topology.select(f'resname {args.ligand}')

if len(ligand_atoms) == 0:
    print(f'No atoms found for ligand with residue name {args.ligand}. Please check your input files.')
    exit(1)

complex_atoms = np.sort(np.concatenate((receptor_atoms, ligand_atoms)))

print(f'There are {t.n_frames} frames in the trajectory.')
print(f'There are {len(receptor_atoms)} atoms in the receptor and {len(ligand_atoms)} in the ligand.')

prmtop = AmberParm(args.top)

# Slice the prmtop file as required:
c_prmtop = prmtop[complex_atoms]
r_prmtop = prmtop[receptor_atoms]
l_prmtop = prmtop[ligand_atoms]

# Now generate OMM "systems":
c_system = c_prmtop.createSystem(implicitSolvent=HCT, nonbondedMethod=NoCutoff)
r_system = r_prmtop.createSystem(implicitSolvent=HCT, nonbondedMethod=NoCutoff)
l_system = l_prmtop.createSystem(implicitSolvent=HCT, nonbondedMethod=NoCutoff)

c_integrator = LangevinMiddleIntegrator(310*kelvin, 1/picosecond, 0.002*picosecond)
r_integrator = LangevinMiddleIntegrator(310*kelvin, 1/picosecond, 0.002*picosecond)
l_integrator = LangevinMiddleIntegrator(310*kelvin, 1/picosecond, 0.002*picosecond)
c_simulation = Simulation(c_prmtop.topology, c_system, c_integrator)
r_simulation = Simulation(r_prmtop.topology, r_system, r_integrator)
l_simulation = Simulation(l_prmtop.topology, l_system, l_integrator)

# MDTraj stores coordinates in nanometers, but OMM needs to know this explicitly:
c_simulation.context.setPositions(t.xyz[0][complex_atoms] * nanometer)

c_energies = []
r_energies = []
l_energies = []

print("Calculating MM-GBSA energies for each frame in the trajectory...")
for i in tqdm.tqdm(range(t.n_frames)):
    c_simulation.context.setPositions(t.xyz[i][complex_atoms] * nanometer)
    c_state = c_simulation.context.getState(getEnergy=True)
    c_energies.append(c_state.getPotentialEnergy())
        
    r_simulation.context.setPositions(t.xyz[i][receptor_atoms] * nanometer)
    r_state = r_simulation.context.getState(getEnergy=True)
    r_energies.append(r_state.getPotentialEnergy())
    
    l_simulation.context.setPositions(t.xyz[i][ligand_atoms] * nanometer)
    l_state = l_simulation.context.getState(getEnergy=True)
    l_energies.append(l_state.getPotentialEnergy())

c_energies = np.array(c_energies)
r_energies = np.array(r_energies)
l_energies = np.array(l_energies)

interaction_energies = c_energies - (r_energies + l_energies)
print(f"MM-GBSA prediction of binding affinity = {interaction_energies.mean().format('%8.2f')} SD: {interaction_energies.std():5.2f}")

