#!/bin/bash
#SBATCH --job-name="[OpenGaussian] NeRF-DS"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --time=5:00:00
#SBATCH --output=/usr/stud/lyun/OpenGaussian/log/slurm-%j.out
#SBATCH --error=/usr/stud/lyun/OpenGaussian/log/slurm-%j.out

dataset_name="basin_novel_view"
python train.py -s /usr/stud/lyun/4dgsam/data/NeRF-DS/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --load_iteration 20000 --start_ins_feat_iter 20000 --start_root_cb_iter 30000 --start_leaf_cb_iter 50000 --root_node_num 8 --leaf_node_num 4 --eval --iterations 70000
