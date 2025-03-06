#!/bin/bash
#SBATCH --job-name="[OpenGaussian] technicolor"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --time=10:00:00
#SBATCH --partition=DEADLINEBIG
#SBATCH --comment="iccv"
#SBATCH --output=/usr/stud/lyun/OpenGaussian/log/slurm-%j.out
#SBATCH --error=/usr/stud/lyun/OpenGaussian/log/slurm-%j.out

# dataset_name="interp/torchocolate"
# model_name="torchocolate_custom_corr"

dataset_name="Fabien"

python train.py --source_path /usr/stud/lyun/storage/user/4dgsam/data/technicolor/Undistorted/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name} --load_iteration 20000 --start_ins_feat_iter 20000 --start_root_cb_iter 30000 --start_leaf_cb_iter 50000 --root_node_num 8 --leaf_node_num 4 --eval --iterations 70000 --save_memory
# python train.py --source_path /usr/stud/lyun/storage/user/4dgsam/data/technicolor/Undistorted/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name} --load_iteration 50000 --start_ins_feat_iter 20000 --start_root_cb_iter 30000 --start_leaf_cb_iter 50000 --root_node_num 16 --leaf_node_num 4 --eval --iterations 70000 --save_memory --start_checkpoint /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/Painter/chkpnt50000.pth
