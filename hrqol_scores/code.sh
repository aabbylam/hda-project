#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=20:mem=400gb
#PBS -N eq5d_round4


cd /rds/general/user/hsl121/home/hda_project/hrqol_scores

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate TDS

python eq5d.py
