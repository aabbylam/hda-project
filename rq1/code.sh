#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=20:mem=400gb
#PBS -N multiclass_hyperparam_pls_cpu


cd /rds/general/user/hsl121/home/hda_project/rq1

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate TDS

python random_forest.py
