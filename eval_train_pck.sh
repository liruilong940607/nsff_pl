
SUBJECT=mochi-high-five_0-180-1_aligned_gq90_bk10
CUDA_VISIBLE_DEVICES=7 python eval_nerfbios.py \
  --dataset_name monocular --root_dir ./data/${SUBJECT} \
  --img_wh 360 480 --start_end 0 180 \
  --N_samples 128 --N_importance 0  --encode_t \
  --output_transient \
  --split eval_kps \
  --ckpt_path ckpts/${SUBJECT}/epoch=29-v1.ckpt \
  --scene_name ${SUBJECT} \
  --root_dir_raw /home/ruilongli/workspace/nerfbios/datasets/iphone-captures_v3/${SUBJECT}


# SUBJECT=haru-sit_400-600-1_aligned_gq90_bk10
# CUDA_VISIBLE_DEVICES=7 python eval_nerfbios.py \
#   --dataset_name monocular --root_dir ./data/${SUBJECT} \
#   --img_wh 480 360 --start_end 0 200 \
#   --N_samples 128 --N_importance 0  --encode_t \
#   --output_transient \
#   --split eval_kps \
#   --ckpt_path ckpts/${SUBJECT}/epoch=29.ckpt \
#   --scene_name ${SUBJECT} \
#   --root_dir_raw /home/ruilongli/workspace/nerfbios/datasets/iphone-captures_v3/${SUBJECT}


# SUBJECT=backpack-swing_0-180-1_aligned_gq95_bk120
# CUDA_VISIBLE_DEVICES=7 python eval_nerfbios.py \
#   --dataset_name monocular --root_dir ./data/${SUBJECT} \
#   --img_wh 360 480 --start_end 0 180 \
#   --N_samples 128 --N_importance 0  --encode_t \
#   --output_transient \
#   --split eval_kps \
#   --ckpt_path ckpts/${SUBJECT}/epoch=29.ckpt \
#   --scene_name ${SUBJECT} \
#   --root_dir_raw /home/ruilongli/workspace/nerfbios/datasets/iphone-captures_v3/${SUBJECT}


# SUBJECT=creeper-sway_0-210-1_aligned_gq90_bk120
# CUDA_VISIBLE_DEVICES=7 python eval_nerfbios.py \
#   --dataset_name monocular --root_dir ./data/${SUBJECT} \
#   --img_wh 360 480 --start_end 0 210 \
#   --N_samples 128 --N_importance 0  --encode_t \
#   --output_transient \
#   --split eval_kps \
#   --ckpt_path ckpts/${SUBJECT}/epoch=29.ckpt \
#   --scene_name ${SUBJECT} \
#   --root_dir_raw /home/ruilongli/workspace/nerfbios/datasets/iphone-captures_v3/${SUBJECT}


# SUBJECT=sriracha-tree-2_0-220-1_aligned_gq95_bk120
# CUDA_VISIBLE_DEVICES=7 python eval_nerfbios.py \
#   --dataset_name monocular --root_dir ./data/${SUBJECT} \
#   --img_wh 360 480 --start_end 0 220 \
#   --N_samples 128 --N_importance 0  --encode_t \
#   --output_transient \
#   --split eval_kps \
#   --ckpt_path ckpts/${SUBJECT}/epoch=29.ckpt \
#   --scene_name ${SUBJECT} \
#   --root_dir_raw /home/ruilongli/workspace/nerfbios/datasets/iphone-captures_v3/${SUBJECT}


# SUBJECT=hang-dance-0_120-550-1_aligned_gq95_bk120
# CUDA_VISIBLE_DEVICES=7 python eval_nerfbios.py \
#   --dataset_name monocular --root_dir ./data/${SUBJECT} \
#   --img_wh 360 480 --start_end 0 429 \
#   --N_samples 128 --N_importance 0  --encode_t \
#   --output_transient \
#   --split eval_kps \
#   --ckpt_path ckpts/${SUBJECT}/epoch=29.ckpt \
#   --scene_name ${SUBJECT} \
#   --root_dir_raw /home/ruilongli/workspace/nerfbios/datasets/iphone-captures_v3/${SUBJECT}

  
# SUBJECT=hang-dance-1_0-250-1_aligned_gq95_bk120
# CUDA_VISIBLE_DEVICES=7 python eval_nerfbios.py \
#   --dataset_name monocular --root_dir ./data/${SUBJECT} \
#   --img_wh 360 480 --start_end 0 250 \
#   --N_samples 128 --N_importance 0  --encode_t \
#   --output_transient \
#   --split eval_kps \
#   --ckpt_path ckpts/${SUBJECT}/epoch=29.ckpt \
#   --scene_name ${SUBJECT} \
#   --root_dir_raw /home/ruilongli/workspace/nerfbios/datasets/iphone-captures_v3/${SUBJECT}


# SUBJECT=block-move_0-350-1_aligned_gq95_bk120
# CUDA_VISIBLE_DEVICES=7 python eval_nerfbios.py \
#   --dataset_name monocular --root_dir ./data/${SUBJECT} \
#   --img_wh 360 480 --start_end 0 350 \
#   --N_samples 128 --N_importance 0  --encode_t \
#   --output_transient \
#   --split eval_kps \
#   --ckpt_path ckpts/${SUBJECT}/epoch=29.ckpt \
#   --scene_name ${SUBJECT} \
#   --root_dir_raw /home/ruilongli/workspace/nerfbios/datasets/iphone-captures_v3/${SUBJECT}


# SUBJECT=teddy-move_0-350-1_aligned_gq95_bk120
# CUDA_VISIBLE_DEVICES=7 python eval_nerfbios.py \
#   --dataset_name monocular --root_dir ./data/${SUBJECT} \
#   --img_wh 360 480 --start_end 0 350 \
#   --N_samples 128 --N_importance 0  --encode_t \
#   --output_transient \
#   --split eval_kps \
#   --ckpt_path ckpts/${SUBJECT}/epoch=29.ckpt \
#   --scene_name ${SUBJECT} \
#   --root_dir_raw /home/ruilongli/workspace/nerfbios/datasets/iphone-captures_v3/${SUBJECT}


# SUBJECT=wheel-rotate_0-250-1_aligned_gq95_bk120
# CUDA_VISIBLE_DEVICES=7 python eval_nerfbios.py \
#   --dataset_name monocular --root_dir ./data/${SUBJECT} \
#   --img_wh 360 480 --start_end 0 250 \
#   --N_samples 128 --N_importance 0  --encode_t \
#   --output_transient \
#   --split eval_kps \
#   --ckpt_path ckpts/${SUBJECT}/epoch=29.ckpt \
#   --scene_name ${SUBJECT} \
#   --root_dir_raw /home/ruilongli/workspace/nerfbios/datasets/iphone-captures_v3/${SUBJECT}

  
  