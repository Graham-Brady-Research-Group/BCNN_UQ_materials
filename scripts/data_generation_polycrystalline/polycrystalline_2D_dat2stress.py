import csv
import re 
import pandas
import pickle

with open('specifications_2D.pkl', 'rb') as f:
    specifications = pickle.load(f)
    
# This script reads the stresses at the element nodes from the .dat file
num_samples = specifications["num_samples"]
for i in range(num_samples):
    inp_file = f"{specifications['data_path']}/sample_{i}/trimesh_{i}.dat"    
    with open(inp_file, "r") as inp:
        lines = inp.readlines()
        lookup = '   THE FOLLOWING TABLE IS PRINTED FOR ALL ELEMENTS WITH TYPE CPS3 AT THE NODES'
        for num, line in enumerate(lines, 1):
            if lookup in line:
                idx = num
        
    filtered_lines = lines[idx+4:]
    # look up the line where '/n' is found
    lookup= ' MAXIMUM'
    for num, line in enumerate(filtered_lines, 1):
        if lookup in line:
            idx = num
            idx = idx-2 

    filtered_lines = filtered_lines[:idx]
    # Convert filtered lines to csv
    with open(f"{specifications['data_path']}/sample_{i}/stress_{i}.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Element', 'Node', 'OR' ,'S11', 'S22','S12','Mises'])  # Header row
        for line in filtered_lines:
            csv_writer.writerow(line.split())