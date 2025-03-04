# dataset_name="as_novel_view"
# python render_new.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --iteration 50000 --eval --skip_train
# python render_new.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --iteration 70000 --eval --skip_train
# python render_by_click_new.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --iteration 70000 --eval --skip_train --points "(122, 93)" "(50, 98)" "(86, 99)" "(143, 126)" "(106, 106)" "(123, 105)"

# dataset_name="basin_novel_view"
# python render_new.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --iteration 50000 --eval --skip_train
# python render_new.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --iteration 70000 --eval --skip_train
# python render_by_click_new.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --iteration 70000 --eval --skip_train --points "(155, 46)" "(184, 63)" "(168, 77)" "(138, 67)"

# dataset_name="cup_novel_view"
# python render_new.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --iteration 50000 --eval --skip_train
# python render_new.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --iteration 70000 --eval --skip_train
# python render_by_click_new.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --iteration 70000 --eval --skip_train --points "(67, 76)"

# dataset_name="press_novel_view"
# python render_new.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --iteration 50000 --eval --skip_train
# python render_new.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --iteration 70000 --eval --skip_train
# python render_by_click_new.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --iteration 70000 --eval --skip_train --points "(83, 49)" "(81, 48)" "(87, 61)" "(100, 71)" "(100, 62)" "(93, 49)"

# dataset_name="plate_novel_view"
# python render_new.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --iteration 50000 --eval --skip_train
# python render_new.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --iteration 70000 --eval --skip_train
# python render_by_click_new.py -s /mnt/sda/4dgsam_data/NeRF-DS/${dataset_name} -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --iteration 70000 --eval --skip_train --points "(67, 55)" "(122, 42)" "(151, 67)" "(131, 69)" "(117, 96)"

dataset_name="as_novel_view"
python metrics_segmentation.py --no_psnr -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --benchmark_path /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr/test/ours_30000/

dataset_name="basin_novel_view"
python metrics_segmentation.py --no_psnr -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --benchmark_path /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr/test/ours_30000/

# dataset_name="cup_novel_view"
# python metrics_segmentation.py --no_psnr -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --benchmark_path /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr/test/ours_30000/

# dataset_name="press_novel_view"
# python metrics_segmentation.py --no_psnr -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --benchmark_path /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr/test/ours_30000/

# dataset_name="plate_novel_view"
# python metrics_segmentation.py --no_psnr -m /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr --benchmark_path /mnt/sda/4dgsam_output/OpenGaussian/NeRF-DS/${dataset_name}_new_corr/test/ours_30000/
