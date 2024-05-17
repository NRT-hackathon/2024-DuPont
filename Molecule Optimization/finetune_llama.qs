#!/bin/bash -l

#SBATCH --account=ea-nrtmidas
#SBATCH --partition=gpu-v100
#SBATCH --job-name="llama"
#SBATCH --time=1-00:00:00
#SBATCH -o stdout_%j.txt
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1  
#SBATCH --gpus-per-node=1 
#SBATCH --cpus-per-task=16
#SBATCH --export=NONE
#SBATCH --mail-user='akepati@udel.edu'
#SBATCH --mail-type=ALL

vpkg_require finetune
source activate /home/3205/.conda/finetune/202404
export PYTHONPATH="/home/3205/.conda/finetune/202404/lib/python3.9/site-packages:$PYTHONPATH"
export HF_TOKEN="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"
export WANDB_API_KEY='ef02d6f015904a20486667e85cf966cb46765887'
export HF_HOME=/lustre/ea-nrtmidas/users/3205/finetuning_llama3/test_llama/hf_cache
mkdir -p $HF_HOME


cd $SLURM_SUBMIT_DIR

# Run your Python script
python3 infer_llama.py