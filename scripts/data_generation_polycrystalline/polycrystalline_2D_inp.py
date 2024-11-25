from __future__ import division
import sys
import os
import numpy as np
from PIL import Image
from matscipy import elasticity
import matplotlib.pyplot as plt
import pickle

## Define orthotropic material - elasticity matrix
E1 = (1 / 204.6) * 1e3  # here GPa is multiplied by 10-6
v12 = -E1 * 137.7
G12 = 126.2 * 1e3
C = np.zeros((6, 6))
Ciiii = 204.6 * 1e3
Ciijj = 137.7 * 1e3
Cijij = 126.2 * 1e3
C[0, 0] = Ciiii
C[1, 1] = Ciiii
C[2, 2] = Ciiii
C[3, 3] = Cijij
C[4, 4] = Cijij
C[5, 5] = Cijij
C[0, 1] = Ciijj
C[0, 2] = Ciijj
C[1, 0] = Ciijj
C[1, 2] = Ciijj
C[2, 0] = Ciijj
C[2, 1] = Ciijj

## This is the section with all the needed functions
def get_num_sets(specifications, i):
    # This function extracts the number of sets from the polymesh.pkl file
    with open(f"{specifications['data_path']}/sample_{i}/polymesh_{i}.pkl", "rb") as file:
        pmesh = pickle.load(file)
        num_sets = len(pmesh.volumes)
        return num_sets
    
def rotate_C(C, num_sets,i):
    # This unction that rotates the elasticity matrix and takes as an input a random angle
    # Function for calculating the indices tensor to matrix notation
    # Thomas Ting - Anisotropic elasticity. Chapter 2, Eq. (2.5-6) and section 2.8
    # and Matscipy package
    # create num_set of independent random angles with seed for reproducibility
    np.random.seed(i)
    angles = np.random.uniform(0, np.pi, num_sets)
    # angles = (np.pi / 4) * np.ones_like(angles)
    C_rot = np.zeros((num_sets, C.shape[0], C.shape[1]))
    for a in range(len(angles)):
        # Define rotation matrix
        Omega = np.array(
            [
                [np.cos(angles[a]), -np.sin(angles[a]), 0],
                [np.sin(angles[a]), np.cos(angles[a]), 0],
                [0, 0, 1],
            ]
        )
        # Rotate the elasticity matrix
        C_rot[a, :, :] = elasticity.rotate_elastic_constants(C, Omega, tol=1e-6)
        np.set_printoptions(precision=3)
        # Write material and angle in two columns of a txt file in fixed width format

    np.set_printoptions(precision=3)
    return C_rot, angles

## Create as many materials as element sets
def mat_prop_from_Crot(C_rot, f, i):
    # This function writes the material properties in the trimesh file in the format that abaqus needs for the anistropic
    # material - https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node193.html
    f.write(
        " %f, %f, %f, %f, %f, %f, %f, %f\n"
        % (
            C_rot[i, 0, 0],
            C_rot[i, 0, 1],
            C_rot[i, 1, 1],
            C_rot[i, 0, 2],
            C_rot[i, 1, 2],
            C_rot[i, 2, 2],
            C_rot[i, 0, 3],
            C_rot[i, 1, 3],
        )
    )
    f.write(
        " %f, %f, %f, %f, %f, %f, %f, %f\n"
        % (
            C_rot[i, 2, 3],
            C_rot[i, 3, 3],
            C_rot[i, 0, 4],
            C_rot[i, 1, 4],
            C_rot[i, 2, 4],
            C_rot[i, 3, 4],
            C_rot[i, 4, 4],
            C_rot[i, 4, 0],
        )
    )
    f.write(
        " %f, %f, %f, %f, %f\n"
        % (
            C_rot[i, 1, 5],
            C_rot[i, 2, 5],
            C_rot[i, 3, 5],
            C_rot[i, 4, 5],
            C_rot[i, 5, 5],
        )
    )    
    
def mat_prop_from_C(C, f):
    # This is af function similar to mat_prop_from_Crot that prints D1111, D1122, D2222, D1133, D2233, D3333, D1212, D1313, D2323
    # for the isotropic material definition can be discarded as well in the future
    f.write(" %f, %f, %f, %f, %f, %f, %f, %f\n"
            % (C[0, 0], 
               C[0, 1], 
               C[1, 1], 
               C[0, 2], 
               C[1, 2], 
               C[2, 2], 
               C[5, 5], 
               C[4, 4]
               ))
    f.write(" %f, \n" % (C[3, 3]))    

def remove_surface(input_file):
    # This function removes the surface definitions from the abaqus inp file
    with open(input_file, "r+") as file:
        lines = file.readlines()
        file.seek(0)
        file.truncate()
        skip = False
        for line in lines:
            if "Surface" in line:
                skip = True
                continue
            elif line.startswith("*End Part"):
                file.write(line)  # Write the *End Part line to the file
                skip = False
                continue 
            if not skip:
                file.write(line)
    
def parse_nodes(lines):
    # This function parses the nodes from the abaqus inp file
    nodes = {}
    for line in lines:
        if line.startswith("*"):
            break
        node_id, x, y = map(float, line.split(","))
        if  0 < abs(x) < 1e-6:
            x = 0 
        if 0 < abs(y) < 1e-6:
            y = 0 
        nodes[int(node_id)] = (x, y)
    return nodes

def parse_elements(lines):
    # This function parses the elements from the abaqus inp file
    elements = {}
    for line in lines:
        if line.startswith("*"):
            break
        element_id, *node_ids = map(int, line.split(","))
        elements[element_id] = node_ids
    return elements

def find_boundary_nodes(nodes,xlim):
    # This function finds the nodes on the boundary of the mesh
    boundary_nodes = {i: set() for i in range(4)}
    for node_id, (x, y) in nodes.items():
        if x == 0 and y == 0:
            constraint_node_id = node_id
        if x == 0:
            boundary_nodes[0].add(node_id)
        if x == xlim:
            boundary_nodes[1].add(node_id)
        if y == 0:
            boundary_nodes[2].add(node_id)
        if y == xlim:
            boundary_nodes[3].add(node_id)
    return boundary_nodes, constraint_node_id

def create_node_set_section(boundary_nodes, constraint_node_id):
    # This function creates the nset section of the abaqus inp file
    nset_section = ""
    for i, node_set in boundary_nodes.items():
        sorted_node_set = sorted(list(node_set))
        chunked_node_set = [sorted_node_set[i:i+16] for i in range(0, len(sorted_node_set), 16)]
        nset_section += f"*Nset, nset=Ext-Nset-{i}\n"
        for chunk in chunked_node_set:
            nset_section += ','.join(map(str, chunk)) + ',\n'
    # Add constraint to node with coordinates (0,0)
    nset_section += f"*Nset, nset=Origin-Nset\n{constraint_node_id}\n"
        # nset_section += f"*Nset, nset=Ext-Nset-{i}\n{','.join(map(str, sorted_node_set))}\n"
    return nset_section

def update_nodes(input_file, updated_nodes, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        node_section = False
        for line in lines:
            if line.startswith("*Node"):
                node_section = True
                file.write(line)
                continue
            if node_section and not line.startswith("*"):
                node_id, _, _ = map(float, line.split(","))
                x, y = updated_nodes[int(node_id)]
                file.write(f"{int(node_id)},{x},{y}\n")
            else:
                node_section = False
                file.write(line)
## -----------------------------------------------------------------##
## -----------------------------------------------------------------##
def process_input_file(ii, xlim, upper_edge_displacement_u2, C):
    input_file = f"{specifications['data_path']}/sample_{ii}/trimesh_{ii}.inp"
    output_file = f"{specifications['data_path']}/sample_{ii}/trimesh_{ii}.inp"
    remove_surface(input_file)
    ## This is the processing part of the inp file
    # Remove the surface definitions
    remove_surface(input_file)
    # Create boundary node sets 
    with open(input_file, "r") as file:
        lines = file.readlines()
    node_start = lines.index("*Node\n") + 1
    element_start = lines.index("*Element, type=CPS3\n") + 1
    nodes = parse_nodes(lines[node_start:])
    update_nodes(input_file, nodes, output_file)
    elements = parse_elements(lines[element_start:])
    boundary_nodes, constraint_node_id = find_boundary_nodes(nodes,xlim)
    new_nset_section = create_node_set_section(boundary_nodes,constraint_node_id)
    nset_section_end = lines.index("*End Part\n")
    with open(output_file, "w") as file:
        file.writelines(lines[:nset_section_end])
        file.write(new_nset_section)
        file.writelines(lines[nset_section_end:])
    # Get number of grains and elastic matrices based on the orientation
    num_sets = get_num_sets(specifications,ii)
    C_rot, angles = rotate_C(C,num_sets,ii)

    with open(f"{specifications['data_path']}/sample_{ii}/trimesh_mat_angle_{ii}.pkl", 'wb') as file:
        pickle.dump(angles, file)
    
    with open(input_file, "r") as f:
        lines = f.readlines()
    # find the insertion point
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith("*End Part"):
            idx = i
            break
    # split the lines into two parts: before and after the insertion point
    lines_before = lines[:idx]
    lines_after = lines[idx + 1 :]
    # write the original lines, new code, and lines after the insertion point
    with open(input_file, "w") as f:
        f.writelines(lines_before)
        f.write("**\n")
        f.write("** SECTIONS\n")
        f.write("**\n")
        for j in range(num_sets):
            set_name = "Set-E-Seed-" + str(j)
            mat_name = "material-" + str(j)
            f.write(
                "*SOLID SECTION, elset={0}, material={1}, orientation=Orientation-1\n".format(
                    set_name, mat_name
                )
            )
            # f.write(
            #     "*SOLID SECTION, elset={0}, material={1}\n".format(
            #         set_name, mat_name
            #     )
            # )
            f.write(",\n")
        # Define material orientation
        f.write("**\n")
        f.write("*Orientation, name=Orientation-1")
        f.write("\n")
        f.write("1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0\n")
        f.write("*End Part\n")
        f.writelines(lines_after)
        f.write("**\n")
        # Create Materials
        f.write("**\n")
        f.write("** MATERIALS\n")
        f.write("**\n")
        for i in range(num_sets):
            mat_name = "material-" + str(i)
            f.write("*Material, name={0}\n".format(mat_name))
            f.write("*Elastic, TYPE=ANISOTROPIC\n")
            # f.write("*Elastic, TYPE=ORTHOTROPIC\n")
            mat_prop_from_Crot(C_rot, f, i)
            # mat_prop_from_C(C, f)
            # f.write("*Elastic\n")
            # f.write(" %f, %f\n"%(1e+5,0.3))

        # Create BC
        f.write("**")
        f.write("** BOUNDARY CONDITIONS\n")
        f.write("**\n")
        f.write("** Name: Disp-BC-1 Type: Displacement/Rotation\n")
        f.write("*Boundary\n")
        f.write("I-PART-1.Origin-Nset, 2, 2\n")
        f.write("** Name: Disp-BC-2 Type: Displacement/Rotation\n")
        f.write("*Boundary\n")
        f.write("I-PART-1.Ext-Nset-0, 1, 1\n")
        f.write("** ----------------------------------------------------------------\n")
        
        # Create Step
        f.write("**\n")
        f.write("** STEP: Step-1\n")
        f.write("**\n")
        f.write("*Step, name=Step-1, nlgeom=NO\n")
        f.write("*Static\n")
        f.write("1., 1., 1e-05, 1.\n")
        f.write("**\n")
        f.write("**\n")
        f.write("** BOUNDARY CONDITIONS\n")
        f.write("**\n")
        f.write("** Name: BC-3 Type: Displacement/Rotation\n")
        f.write("*Boundary\n")
        f.write("I-PART-1.Ext-Nset-1, 1, 1, %f\n" % upper_edge_displacement_u2)
        
        # Create output requests
        f.write("** OUTPUT REQUESTS\n")
        f.write("**\n")
        f.write("*Restart, write, frequency=0\n")
        f.write("**\n")
        f.write("** FIELD OUTPUT: F-Output-1\n")
        f.write("**\n")
        f.write("*OUTPUT, FIELD\n")
        f.write("*Node Output\n")
        f.write("COORD, U\n")
        f.write("*ELEMENT OUTPUT\n")
        f.write("S, MISES\n")
        f.write("*FILE FORMAT, ASCII\n")
        f.write("*EL FILE\n")
        f.write("COORD, S, SINV\n")
        f.write("*NODE FILE\n")
        f.write("COORD, U\n")
        f.write("*EL PRINT, POSITION=NODES, FREQUENCY=1\n")
        f.write("S, MISES\n")
        f.write("**HISTORY OUTPUT: H-Output-1\n")
        f.write("**\n")
        f.write("*Output, history, variable=PRESELECT\n")
        f.write("*End Step\n")
## Read the volume numbers from the polymesh.txt

with open('specifications_2D.pkl', 'rb') as f:
    specifications = pickle.load(f) 

for i in range(specifications["num_samples"]):
    process_input_file(i, specifications["side_length"], specifications["displacement"], C)    
