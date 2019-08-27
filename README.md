# ENGSCI741 - Fracture Mechanics

This library contains a series of Jupyter notebooks for illustrating key concepts in the fracture mechanics module of ENGSCI741.

## Linear elastic fracture mechanics (lefm.ipynb)

This notebook uses complex Westergaard stress functions to contour stress and displacement fields around an opening crack in biaxial tension.
- Compute changes in the three non-zero stresses.
- Compute the crack-tip stress approximation and compare it to the true stress.
- Compute the displacement field and crack opening profile.

## Earthquake source simulation (eq-sim.ipynb)

This notebook introduces concepts important to understanding and simulating an earthquake source as a propagating crack.
- Compute slip-induced stress changes using a Fourier transform method.
- Solve a coupled set of ODEs for a 1D fault with linear slip-weakening friction.
- Understand PID timestep control when solving ODEs.

## Non-local elasticity kernel for hydraulic fracture problems (hf-sim.ipynb)

This notebook explores the kernel method for computing fracture opening due to non-local elasticity.
- Compute the Kernel for a matched pair of pressure impulses.
- Sum multiple Kernels functions to derive the opening profile for an arbitrary pressure distribution.
- Compute the stress intensity factor arising from the particular pressure distribution.

## Getting Started

Download this library, open and run the notebooks.
Visit [https://notebooks.azure.com/ddem014/projects/engsci741](https://notebooks.azure.com/ddem014/projects/engsci741) for an Azure notebook version

## Author

**David Dempsey**