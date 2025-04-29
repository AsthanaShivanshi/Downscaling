#!/bin/bash
#SBATCH --job-name=UNet_training       
#SBATCH --output=job_output-%j.txt 
#SBATCH --error=job_error-%j.txt  
#SBATCH --ntasks=1             
#SBATCH --cpus-per-task=4     
#SBATCH --time=02:00:00         
#SBATCH --mem=32G                

#My environment activation
module load python

source /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/MyPythonEnvironment/bin/activate

#Containing the pipeline: the directory

cd /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling/ML_models


#For quick test module
#python Main.py --quick_test

#For full training modzle

python Main.py 