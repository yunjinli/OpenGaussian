#!/bin/bash
#SBATCH --job-name="[OpenGaussian] hypernerf"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --time=10:00:00
#SBATCH --partition=DEADLINEBIG
#SBATCH --comment="iccv"
#SBATCH --output=/usr/stud/lyun/OpenGaussian/log/slurm-%j.out
#SBATCH --error=/usr/stud/lyun/OpenGaussian/log/slurm-%j.out

# dataset_name="interp/torchocolate"
# model_name="torchocolate_custom_corr"

dataset_name="misc/split-cookie"
model_name="split-cookie_new_corr"

python train.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --load_iteration 20000 --start_ins_feat_iter 20000 --start_root_cb_iter 30000 --start_leaf_cb_iter 50000 --root_node_num 8 --leaf_node_num 4 --eval --iterations 70000
