#!/bin/bash -l

#SBATCH
#SBATCH --job-name=AbaqusRun
#SBATCH --time=2:00:00
#SBATCH --partition=defqs
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2

cd /home/gpaspar/scratch4-lgraham1/George/polycrystalline_2D/data_polycrystalline_2D
ml anaconda
ml cuda
echo $CUDA_VISIBLE_DEVICES
ml matlab
ml abaqus/2020.2
ml list
pip list
abaqus job=trimesh.inp interactive ask_delete=OFF

for i in {0..999}
do
    # Change to the sample directory
    cd sample_$i

    # Run Abaqus command
    abaqus job=trimesh_$i.inp interactive ask_delete=OFF
    rm *.msg *.odb *.sta *.com *.prt
    # Go back to the main directory
    cd ..
done

# Print CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
