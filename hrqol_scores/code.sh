#PBS -l walltime=50:00:00
#PBS -l select=1:ncpus=30:mem=500gb
#PBS -N sqs_2


cd /rds/general/user/hsl121/home/hda_project/hrqol_scores

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate TDS

python sqs.py
