#!/bin/bash
#SBATCH --job-name="[OpenGaussian] render Immersive"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --time=2:00:00
#SBATCH --partition=DEADLINEBIG
#SBATCH --comment="iccv"
#SBATCH --output=/usr/stud/lyun/OpenGaussian/log/slurm-%j.out
#SBATCH --error=/usr/stud/lyun/OpenGaussian/log/slurm-%j.out

# dataset_name="11_Alexa_Meade_Face_Paint_2"
# python render_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/new_data/immersive/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f --iteration 50000 --eval --skip_train --end_frame 50 --load_mask_on_the_fly --load_image_on_the_fly --load2gpu_on_the_fly
# python render_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/new_data/immersive/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f --iteration 70000 --eval --skip_train --end_frame 50 --load_mask_on_the_fly --load_image_on_the_fly --load2gpu_on_the_fly
# python render_by_click_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/new_data/immersive/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f --eval --iteration 70000 --skip_train --points "(728, 528)" "(810, 545)" "(815, 649)" "(754, 606)" "(700, 582)" "(713, 601)" "(595, 579)" "(596, 555)" "(622, 611)" "(655, 636)" "(676, 664)" "(692, 658)" "(715, 701)" "(726, 771)" "(753, 830)" "(773, 877)" "(735, 904)" "(704, 916)" "(850, 911)" "(849, 873)" "(894, 812)" "(929, 763)" "(948, 833)" "(967, 882)" "(980, 918)" "(862, 716)" "(805, 720)" "(774, 933)" --end_frame 50 --load_mask_on_the_fly --load_image_on_the_fly --load2gpu_on_the_fly
# python metrics_segmentation.py --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f --no_psnr --benchmark_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f/test/ours_30000/

# dataset_name="10_Alexa_Meade_Face_Paint_1"
# python render_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/new_data/immersive/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f --iteration 50000 --eval --skip_train --end_frame 50 --load_mask_on_the_fly --load_image_on_the_fly --load2gpu_on_the_fly
# python render_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/new_data/immersive/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f --iteration 70000 --eval --skip_train --end_frame 50 --load_mask_on_the_fly --load_image_on_the_fly --load2gpu_on_the_fly
# python render_by_click_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/new_data/immersive/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f --eval --iteration 70000 --skip_train --points "(565, 849)" "(555, 865)" "(515, 804)" "(427, 817)" "(360, 875)" "(284, 903)" "(353, 934)" --end_frame 50 --load_mask_on_the_fly --load_image_on_the_fly --load2gpu_on_the_fly
# python metrics_segmentation.py --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f --no_psnr --benchmark_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f/test/ours_30000/

dataset_name="02_Flames"
# python render_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/new_data/immersive/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f --iteration 50000 --eval --skip_train --end_frame 50 --load_mask_on_the_fly --load_image_on_the_fly --load2gpu_on_the_fly
# python render_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/new_data/immersive/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f --iteration 70000 --eval --skip_train --end_frame 50 --load_mask_on_the_fly --load_image_on_the_fly --load2gpu_on_the_fly
# python render_by_click_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/new_data/immersive/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f --eval --iteration 70000 --skip_train --points "(499, 460)" "(428, 462)" "(549, 472)" "(410, 532)" "(462, 533)" "(593, 538)" "(422, 621)" "(524, 584)" "(601, 650)" "(482, 685)" "(418, 726)" "(542, 721)" "(589, 753)" "(473, 816)" "(468, 855)" "(435, 892)" "(608, 842)" "(577, 844)" "(553, 801)" --end_frame 50 --load_mask_on_the_fly --load_image_on_the_fly --load2gpu_on_the_fly
python render_by_click_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/new_data/immersive/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f --eval --iteration 70000 --skip_train --points "(496, 459)" "(500, 551)" "(455, 549)" "(556, 544)" "(471, 632)" "(526, 628)" "(468, 729)" "(534, 737)" "(472, 858)" "(571, 839)" --end_frame 50 --load_mask_on_the_fly --load_image_on_the_fly --load2gpu_on_the_fly
# python render_by_click_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/new_data/immersive/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f --eval --iteration 70000 --skip_train --points "(508, 470)" --end_frame 50 --load_mask_on_the_fly --load_image_on_the_fly --load2gpu_on_the_fly
# python render_by_click_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/new_data/immersive/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f --eval --iteration 70000 --skip_train --points "(470, 870)" "(574, 845)" "(470, 811)" "(554, 789)" "(507, 475)"  --end_frame 50 --load_mask_on_the_fly --load_image_on_the_fly --load2gpu_on_the_fly
python metrics_segmentation.py --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f --no_psnr --benchmark_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f/test/ours_30000/

dataset_name="01_Welder"
python render_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/new_data/immersive/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f --iteration 50000 --eval --skip_train --end_frame 50 --load_mask_on_the_fly --load_image_on_the_fly --load2gpu_on_the_fly
python render_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/new_data/immersive/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f --iteration 70000 --eval --skip_train --end_frame 50 --load_mask_on_the_fly --load_image_on_the_fly --load2gpu_on_the_fly
python render_by_click_new.py --source_path /usr/stud/lyun/storage/user/4dgsam/new_data/immersive/${dataset_name} --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f --eval --iteration 70000 --skip_train --points "(1150, 353)" "(1203, 309)" "(1255, 358)" "(1197, 428)" "(1159, 446)" "(1253, 462)" "(1182, 487)" "(1259, 503)" "(1071, 513)" "(1021, 541)" "(966, 573)" "(1013, 615)" "(1238, 587)" "(1210, 653)" "(1143, 843)" "(1071, 894)" "(1234, 903)" "(1129, 916)" --end_frame 50 --load_mask_on_the_fly --load_image_on_the_fly --load2gpu_on_the_fly
python metrics_segmentation.py --model_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f --no_psnr --benchmark_path /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/immersive/${dataset_name}_50f/test/ours_30000/


