#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=20:mem=400gb
#PBS -N gad7_round2


cd /rds/general/user/hsl121/home/hda_project/final_pipeline

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate TDS

python gad7.py
