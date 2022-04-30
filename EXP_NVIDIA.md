```
# Balloon1-2  Balloon2-2  DynamicFace-2  Jumping  Playground  Skating-2  Truck-2  Umbrella

python train.py \
  --dataset_name monocular --root_dir ./data/Balloon1-2 \
  --img_wh 540 288 --start_end 0 24 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 500 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name Balloon1-2

python train.py \
  --dataset_name monocular --root_dir ./data/Balloon2-2 \
  --img_wh 544 288 --start_end 0 24 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 500 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name Balloon2-2

python train.py \
  --dataset_name monocular --root_dir ./data/DynamicFace-2 \
  --img_wh 547 288 --start_end 0 24 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 500 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name DynamicFace-2

python train.py \
  --dataset_name monocular --root_dir ./data/Jumping \
  --img_wh 540 288 --start_end 0 24 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 500 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name Jumping

python train.py \
  --dataset_name monocular --root_dir ./data/Playground \
  --img_wh 540 288 --start_end 0 24 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 500 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name Playground

python train.py \
  --dataset_name monocular --root_dir ./data/Skating-2 \
  --img_wh 540 288 --start_end 0 24 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 500 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name Skating-2

python train.py \
  --dataset_name monocular --root_dir ./data/Truck-2 \
  --img_wh 540 288 --start_end 0 24 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 50 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name Truck-2

python train.py \
  --dataset_name monocular --root_dir ./data/Umbrella \
  --img_wh 543 288 --start_end 0 24 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 50 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name Umbrella

```

```
python eval_nerfbios.py \
  --dataset_name monocular --root_dir ./data/mochi-high-five_0-180-1_aligned_gq90_bk10 \
  --N_samples 128 --N_importance 0  --encode_t --img_wh 360 480 --start_end 0 163 \
  --output_transient \
  --split test_extreme --video_format mp4 --fps 5 \
  --ckpt_path ckpts/mochi-high-five_0-180-1_aligned_gq90_bk10/epoch=31.ckpt \
  --scene_name mochi-high-five_0-180-1_aligned_gq90_bk10 \
  --root_dir_raw /home/ruilongli/workspace/nerfbios/datasets/iphone-captures_v2/mochi-high-five_0-180-1_aligned_gq90_bk10

python eval_nerfbios.py \
  --dataset_name monocular --root_dir ./data/hang-dance-0_120-550-1_aligned_gq95_bk120 \
  --N_samples 128 --N_importance 0  --encode_t --img_wh 360 480 --start_end 0 429 \
  --output_transient \
  --split test_extreme --video_format mp4 --fps 5 \
  --ckpt_path ckpts/hang-dance-0_120-550-1_aligned_gq95_bk120/epoch=13.ckpt \
  --scene_name hang-dance-0_120-550-1_aligned_gq95_bk120 \
  --root_dir_raw /home/ruilongli/workspace/nerfbios/datasets/iphone-captures_v2/hang-dance-0_120-550-1_aligned_gq95_bk120
```

```
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset_name monocular --root_dir ./data/haru-sit_400-600-1_aligned_gq90_bk10 \
  --img_wh 480 360 --start_end 0 200 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 30 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name haru-sit_400-600-1_aligned_gq90_bk10

CUDA_VISIBLE_DEVICES=1 python train.py \
  --dataset_name monocular --root_dir ./data/sriracha-tree-2_0-220-1_aligned_gq95_bk120 \
  --img_wh 360 480 --start_end 0 220 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 30 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name sriracha-tree-2_0-220-1_aligned_gq95_bk120

CUDA_VISIBLE_DEVICES=2 python train.py \
  --dataset_name monocular --root_dir ./data/backpack-swing_0-180-1_aligned_gq95_bk120 \
  --img_wh 360 480 --start_end 0 180 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 30 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name backpack-swing_0-180-1_aligned_gq95_bk120

CUDA_VISIBLE_DEVICES=3 python train.py \
  --dataset_name monocular --root_dir ./data/creeper-sway_0-210-1_aligned_gq90_bk120 \
  --img_wh 360 480 --start_end 0 210 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 30 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name creeper-sway_0-210-1_aligned_gq90_bk120

CUDA_VISIBLE_DEVICES=4 python train.py \
  --dataset_name monocular --root_dir ./data/book-close_0-160-1_aligned_gq95_bk120 \
  --img_wh 360 480 --start_end 0 160 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 30 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name book-close_0-160-1_aligned_gq95_bk120

CUDA_VISIBLE_DEVICES=5 python train.py \
  --dataset_name monocular --root_dir ./data/apple-rotate_0-150-1_aligned_gq95_bk120 \
  --img_wh 360 480 --start_end 0 150 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 30 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name apple-rotate_0-150-1_aligned_gq95_bk120

CUDA_VISIBLE_DEVICES=6 python train.py \
  --dataset_name monocular --root_dir ./data/hand-twist-0_0-300-1_aligned_gq95_bk120 \
  --img_wh 360 480 --start_end 0 300 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 30 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name hand-twist-0_0-300-1_aligned_gq95_bk120

CUDA_VISIBLE_DEVICES=7 python train.py \
  --dataset_name monocular --root_dir ./data/pillow-squeeze_0-300-1_aligned_gq90_bk120 \
  --img_wh 360 480 --start_end 0 300 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 30 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name pillow-squeeze_0-300-1_aligned_gq90_bk120

CUDA_VISIBLE_DEVICES=8 python train.py \
  --dataset_name monocular --root_dir ./data/block-move_0-350-1_aligned_gq95_bk120 \
  --img_wh 360 480 --start_end 0 350 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 30 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name block-move_0-350-1_aligned_gq95_bk120

CUDA_VISIBLE_DEVICES=9 python train.py \
  --dataset_name monocular --root_dir ./data/teddy-move_0-350-1_aligned_gq95_bk120 \
  --img_wh 360 480 --start_end 0 350 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 30 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name teddy-move_0-350-1_aligned_gq95_bk120
```

```
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset_name monocular --root_dir ./data/Idle_2 \
  --img_wh 250 250 --start_end 0 101 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 30 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name Idle_2
```