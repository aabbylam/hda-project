#PBS -l walltime=20:00:00
#PBS -l select=1:ncpus=12:mem=200gb
#PBS -N multiclass_hyperparam_pls_cpu


cd /rds/general/user/hsl121/home/hda_project/rq1

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate TDS

python random_forest.py
