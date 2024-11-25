# First, run the script to get the specification for the simulation
python polycrystalline_2D_specs.py
# Then, run the meshing to generate the files and wait for all the samples to be generated
python polycrystalline_2D_mesh_parallel.py
# Next, run the .inp modifying script
python polycrystalline_2D_inp.py
# Next, transport the data_polycrystalline_2D data to the cluster and run the bash script
filezilla and zip/unzip back and forth - 
    zip data_polycrystalline_2D.zip -r data_polycrystalline_2D
    unzip data_polycrystalline_2D.zip -d .
# Run the abaqus sh script
sh polycrystalline_2D_abaqus.sh    
# Read the .dat file and extract the stresses at the nodes 
python polycrystalline_2D_dat2stress.py
# Run the matlab script to get the stress data,the nodal coordinates and the elset info from the fil files 
matlab -nodisplay -nosplash -nodesktop -r "run(polycrystalline_2D_fil2csv)'; exit;"
# Run polycrystalline_2D_angle_grid to create the a matrix with angle values at each pixel  
python polycrystalline_2D_angle_grid.py
# Run the plotting script to generate the figures 
python polycrystalline_2D_plots.py
# Run the load data script to create training data 
python polycrystalline_2D_load_data.py
