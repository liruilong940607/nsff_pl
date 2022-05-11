
SUBJECT=hang-dance-1_0-250-1_aligned_gq95_bk120
CUDA_VISIBLE_DEVICES=4 python eval_nerfbios.py \
  --dataset_name monocular --root_dir ./data/${SUBJECT} \
  --img_wh 360 480 --start_end 0 250 \
  --N_samples 128 --N_importance 0  --encode_t \
  --output_transient \
  --split eval_test \
  --ckpt_path ckpts/${SUBJECT}/epoch=29.ckpt \
  --scene_name ${SUBJECT} \
  --root_dir_raw /home/ruilongli/workspace/nerfbios/datasets/iphone-captures_v3/${SUBJECT}


SUBJECT=block-move_0-350-1_aligned_gq95_bk120
CUDA_VISIBLE_DEVICES=4 python eval_nerfbios.py \
  --dataset_name monocular --root_dir ./data/${SUBJECT} \
  --img_wh 360 480 --start_end 0 350 \
  --N_samples 128 --N_importance 0  --encode_t \
  --output_transient \
  --split eval_test \
  --ckpt_path ckpts/${SUBJECT}/epoch=29.ckpt \
  --scene_name ${SUBJECT} \
  --root_dir_raw /home/ruilongli/workspace/nerfbios/datasets/iphone-captures_v3/${SUBJECT}


SUBJECT=teddy-move_0-350-1_aligned_gq95_bk120
CUDA_VISIBLE_DEVICES=4 python eval_nerfbios.py \
  --dataset_name monocular --root_dir ./data/${SUBJECT} \
  --img_wh 360 480 --start_end 0 350 \
  --N_samples 128 --N_importance 0  --encode_t \
  --output_transient \
  --split eval_test \
  --ckpt_path ckpts/${SUBJECT}/epoch=29.ckpt \
  --scene_name ${SUBJECT} \
  --root_dir_raw /home/ruilongli/workspace/nerfbios/datasets/iphone-captures_v3/${SUBJECT}


SUBJECT=wheel-rotate_0-250-1_aligned_gq95_bk120
CUDA_VISIBLE_DEVICES=4 python eval_nerfbios.py \
  --dataset_name monocular --root_dir ./data/${SUBJECT} \
  --img_wh 360 480 --start_end 0 250 \
  --N_samples 128 --N_importance 0  --encode_t \
  --output_transient \
  --split eval_test \
  --ckpt_path ckpts/${SUBJECT}/epoch=29.ckpt \
  --scene_name ${SUBJECT} \
  --root_dir_raw /home/ruilongli/workspace/nerfbios/datasets/iphone-captures_v3/${SUBJECT}

  
  