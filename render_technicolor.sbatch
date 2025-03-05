#!/bin/bash
#SBATCH --job-name="[OpenGaussian] render technicolor"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --time=01:00:00
#SBATCH --partition=DEADLINEBIG
#SBATCH --comment="iccv"
#SBATCH --output=/usr/stud/lyun/OpenGaussian/log/slurm-%j.out
#SBATCH --error=/usr/stud/lyun/OpenGaussian/log/slurm-%j.out

# dataset_name="Fabien"
# python render_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/data/technicolor/Undistorted/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name} --iteration 50000 --eval --skip_train
# python render_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/data/technicolor/Undistorted/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name} --iteration 70000 --eval --skip_train
# python render_by_click_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/data/technicolor/Undistorted/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name} --eval --iteration 70000 --skip_train --points "(38, 311)" "(287, 399)" "(63, 495)" "(142, 421)" "(252, 505)" "(438, 510)" "(588, 506)" "(570, 450)" "(872, 447)" "(725, 446)" "(675, 497)" "(759, 487)" "(846, 499)"
# python metrics_segmentation.py --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name} --no_psnr --benchmark_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name}/test/ours_30000/

# dataset_name="Birthday"
# python render_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/data/technicolor/Undistorted/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name} --iteration 50000 --eval --skip_train
# python render_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/data/technicolor/Undistorted/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name} --iteration 70000 --eval --skip_train
# python render_by_click_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/data/technicolor/Undistorted/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name} --eval --iteration 70000 --skip_train --points "(780, 97)" "(796, 125)" "(812, 154)" "(823, 75)" "(838, 98)" "(848, 115)" "(929, 140)" "(893, 121)" "(846, 143)" "(904, 162)" "(866, 174)" "(851, 200)" "(907, 197)" "(881, 229)" "(860, 268)" "(889, 264)" "(894, 283)" "(856, 311)" "(860, 343)" "(846, 395)" "(836, 424)" "(813, 461)" "(801, 467)" "(819, 317)" "(786, 348)" "(749, 367)" "(728, 383)" "(709, 396)" "(685, 399)" "(669, 395)" "(877, 424)" "(910, 426)" "(947, 426)" "(986, 422)" "(963, 284)" "(954, 228)" "(956, 193)" "(918, 223)" "(917, 262)" "(861, 469)" "(917, 470)" "(982, 469)" "(954, 491)" "(922, 514)" "(891, 499)" "(880, 499)" "(863, 509)" "(848, 528)" "(885, 530)"
# python metrics_segmentation.py --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name} --no_psnr --benchmark_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name}/test/ours_30000/

# dataset_name="Painter"
# python render_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/data/technicolor/Undistorted/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name} --iteration 50000 --eval --skip_train
# python render_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/data/technicolor/Undistorted/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name} --iteration 70000 --eval --skip_train
# python render_by_click_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/data/technicolor/Undistorted/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name} --eval --iteration 70000 --skip_train --points "(521, 234)" "(495, 250)" "(544, 314)" "(409, 362)" "(291, 358)" "(449, 393)" "(539, 395)" "(507, 431)" "(504, 481)" "(571, 489)"
# python metrics_segmentation.py --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name} --no_psnr --benchmark_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name}/test/ours_30000/


# dataset_name="Theater"
# python render_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/data/technicolor/Undistorted/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name} --iteration 50000 --eval --skip_train
# python render_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/data/technicolor/Undistorted/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name} --iteration 70000 --eval --skip_train
# python render_by_click_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/data/technicolor/Undistorted/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name} --eval --iteration 70000 --skip_train --points "(557, 471)" "(463, 532)" "(497, 534)" "(537, 533)" "(544, 487)" "(591, 462)" "(605, 500)" "(579, 502)" "(588, 538)" "(563, 522)" "(543, 511)"
# python metrics_segmentation.py --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name} --no_psnr --benchmark_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/technicolor/${dataset_name}/test/ours_30000/
