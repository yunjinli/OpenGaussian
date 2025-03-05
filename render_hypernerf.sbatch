#!/bin/bash
#SBATCH --job-name="[OpenGaussian] hypernerf"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --time=10:00:00
#SBATCH --partition=DEADLINEBIG
#SBATCH --comment="iccv"
#SBATCH --output=/usr/stud/lyun/OpenGaussian/log/slurm-%j.out
#SBATCH --error=/usr/stud/lyun/OpenGaussian/log/slurm-%j.out


dataset_name="interp/hand1-dense-v2"
model_name="hand1-dense-v2_custom_corr"

python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 50000 --eval --skip_train
python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train
python render_by_click_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train --points "(87, 333)" "(276, 384)"

dataset_name="misc/americano"
model_name="americano_custom_corr"

python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 50000 --eval --skip_train
python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train
python render_by_click_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train --points "(169, 159)" "(202, 191)" "(153, 250)" 

dataset_name="interp/chickchicken"
model_name="chickchicken_custom_corr"

python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 50000 --eval --skip_train
python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train
python render_by_click_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train --points "(201, 193)" "(200, 235)" "(293, 146)" "(372, 224)" "(298, 274)" "(375, 322)" "(342, 391)" "(422, 420)" "(354, 466)" "(416, 510)" "(387, 519)" "(290, 520)" "(277, 570)" "(279, 607)" "(148, 557)" "(54, 569)"

dataset_name="interp/cut-lemon1"
model_name="cut-lemon1_custom_corr"

python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 50000 --eval --skip_train
python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train
python render_by_click_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train --points "(412, 210)" "(401, 248)" "(469, 274)" "(513, 219)" "(329, 32)" "(386, 71)" "(478, 123)" "(554, 157)" "(523, 12)" "(478, 9)" "(489, 37)" "(553, 77)" "(528, 84)"

dataset_name="misc/keyboard"
model_name="keyboard_custom_corr"

python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 50000 --eval --skip_train
python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train
python render_by_click_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train --points "(452, 283)" "(517, 366)" "(569, 435)" "(587, 114)" "(682, 146)" "(874, 134)"

dataset_name="misc/espresso"
model_name="espresso_new_corr"

python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 50000 --eval --skip_train
python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train
python render_by_click_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train --points "(327, 270)" "(413, 285)" "(324, 321)" "(407, 336)"

dataset_name="misc/oven-mitts"
model_name="oven-mitts_custom_corr"

python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 50000 --eval --skip_train
python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train
python render_by_click_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train --points "(290, 403)" "(339, 306)" "(395, 452)" "(385, 163)" "(414, 36)" "(519, 100)" "(514, 231)" "(512, 331)"

dataset_name="interp/slice-banana"
model_name="slice-banana_custom_corr"

python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 50000 --eval --skip_train
python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train
python render_by_click_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train --points "(393, 22)" "(177, 137)" "(31, 46)" "(27, 105)" "(35, 202)" "(101, 160)" "(151, 262)" "(100, 302)" "(129, 314)" "(119, 360)" "(167, 366)" "(193, 332)" "(233, 337)" "(286, 337)" "(333, 345)" "(261, 450)" "(261, 502)" "(281, 544)" "(325, 395)" "(387, 381)" "(452, 379)" "(434, 346)" "(314, 296)" "(337, 269)" "(277, 289)" "(362, 249)" "(448, 236)"

dataset_name="misc/split-cookie"
model_name="split-cookie_new_corr"

python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 50000 --eval --skip_train
python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train
python render_by_click_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train --points "(222, 386)" "(268, 397)" "(306, 421)" "(319, 350)" "(276, 360)" "(236, 350)"

dataset_name="interp/torchocolate"
model_name="torchocolate_custom_corr"

python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 50000 --eval --skip_train
python render_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train
python render_by_click_new.py -s /usr/stud/lyun/4dgsam/data/HyperNeRF/${dataset_name} -m /usr/stud/lyun/storage/user/4dgsam_output/OpenGaussian/HyperNeRF/${model_name} --iteration 70000 --eval --skip_train --points "(129, 285)" "(138, 250)" "(155, 173)" "(173, 217)" "(202, 269)" "(215, 306)" "(234, 350)"
