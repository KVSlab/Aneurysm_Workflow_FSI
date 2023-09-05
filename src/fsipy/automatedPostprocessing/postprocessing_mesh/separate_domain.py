# Copyright (c) 2023 David Bruneau
# Modified by Kei Yamamoto 2023
# SPDX-License-Identifier: GPL-3.0-or-later

import pickle
import numpy as np
import h5py
from pathlib import Path

from fsipy.automatedPostprocessing.postprocessing_mesh import postprocessing_mesh_common

from dolfin import MPI, Mesh, MeshFunction, HDF5File, SubMesh, File


def separate_domain(mesh_path: Path, fluid_domain_id: int, solid_domain_id: int, view: bool = False) -> None:
    """
    args:
        folder_path (Path): Path to the simulation results folder.
        mesh_path (Path): Path to the mesh file.
        fluid_domain_id (int): Domain ID for fluid domain.
        solid_domain_id (int): Domain ID for solid domain.

    Returns:
        None
    """
    # Read in original FSI mesh
    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), str(mesh_path), "r") as hdf:
        hdf.read(mesh, "/mesh", False)
        domains = MeshFunction("size_t", mesh, mesh.topology().dim())
        hdf.read(domains, "/domains")
        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        hdf.read(boundaries, "/boundaries")

    for domain_id, domain_name in zip([fluid_domain_id, solid_domain_id], ["fluid", "solid"]):

        domain_of_interest = SubMesh(mesh, domains, domain_id)
        domain_of_interest_path = mesh_path.with_name(mesh_path.stem + f"_{domain_name}.h5")
        print(f" --- Saving {domain_name} domain to {domain_of_interest_path} \n")
        with HDF5File(domain_of_interest.mpi_comm(), str(domain_of_interest_path), "w") as hdf:
            hdf.write(domain_of_interest, "/mesh")

        # Save for viewing in paraview
        if view:
            domain_of_interest_pvd_path = domain_of_interest_path.with_suffix(".pvd")
            File(str(domain_of_interest_pvd_path)) << domain_of_interest
        

    print(" --- Done separating domains \n")
    
    with h5py.File(mesh_path) as vectorData:
        domains_values = vectorData['domains/values'][:]
        domain_topology = vectorData['domains/topology'][:, :]
        domain_coordinates = vectorData['mesh/coordinates'][:, :]
    

    for domain_id, domain_name in zip([fluid_domain_id, solid_domain_id], ["fluid", "solid"]):

        domain_of_interest_index = (domains_values == domain_id).nonzero()
        domain_of_interest_topology = domain_topology[domain_of_interest_index, :]
        domain_of_interest_ids = np.unique(domain_of_interest_topology)
        domain_of_interest_coordinates = domain_coordinates[domain_of_interest_ids, :]

         # Fix topology of the each domain
        print(f" --- Fixing topology of {domain_name} domain \n")
        for node_id in range(len(domain_of_interest_ids)):
            domain_of_interest_topology = np.where(domain_of_interest_topology == domain_of_interest_ids[node_id], 
                                                   node_id, domain_of_interest_topology)
        
        # Create path for fixed mesh file and copy mesh file to new "fixed" file for safety reasons
        print("--- Saving the fixed mesh file \n")
        domain_of_interest_path = mesh_path.with_name(mesh_path.stem + f"_{domain_name}.h5")
        domain_of_interest_fixed_path = mesh_path.with_name(mesh_path.stem + f"_{domain_name}_fixed.h5")
        domain_of_interest_fixed_path.write_bytes(domain_of_interest_path.read_bytes())

        with h5py.File(domain_of_interest_fixed_path, "r+") as vectorData:
            coordinate_array = vectorData["mesh/coordinates"]
            coordinate_array[...] = domain_of_interest_coordinates
            topology_array = vectorData["mesh/topology"]
            topology_array[...] = domain_of_interest_topology

        
        # remove the original mesh file and rename the fixed mesh file
        domain_of_interest_path.unlink()
        domain_of_interest_fixed_path.rename(domain_of_interest_path)


def main() -> None:

    assert MPI.size(MPI.comm_world) == 1, "This script only runs in serial."

    args = postprocessing_mesh_common.parse_arguments()

    folder_path = Path(args.folder)
    if args.mesh_path is None:
        mesh_path = folder_path / "Mesh" / "mesh.h5"
    else:
        mesh_path = Path(args.mesh_path)

    parameter_path = folder_path / "Checkpoint" / "default_variables.pickle"
    with open(parameter_path, "rb") as f:
        parameters = pickle.load(f)
        fluid_domain_id = parameters["dx_f_id"]
        solid_domain_id = parameters["dx_s_id"]

        if type(fluid_domain_id) is not int:
            fluid_domain_id = fluid_domain_id[0]
            print("fluid_domain_id is not int, using first element of list \n")
        if type(solid_domain_id) is not int:
            solid_domain_id = solid_domain_id[0]
            print("solid_domain_id is not int, using first element of list \n")

    print(" --- Separating fluid and solid domains using domain IDs \n")
    print(f" --- Fluid domain ID: {fluid_domain_id} and Solid domain ID: {solid_domain_id} \n")

    separate_domain(mesh_path, fluid_domain_id, solid_domain_id)

    # Check if refined mesh exists
    refined_mesh_path = mesh_path.with_name(mesh_path.stem + "_refined.h5")
    if refined_mesh_path.exists():
        print(" --- Refined mesh exists, separating domains for refined mesh \n")
        separate_domain(refined_mesh_path, fluid_domain_id, solid_domain_id, view=args.view)
    else:
        print(" --- Refined mesh does not exist \n")


if __name__ == "__main__":
    main()
