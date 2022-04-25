```
# Balloon1-2  Balloon2-2  DynamicFace-2  Jumping  Playground  Skating-2  Truck-2  Umbrella

python train.py \
  --dataset_name monocular --root_dir ./data/Balloon1-2 \
  --img_wh 540 288 --start_end 0 24 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 50 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name Balloon1-2

python train.py \
  --dataset_name monocular --root_dir ./data/Balloon2-2 \
  --img_wh 544 288 --start_end 0 24 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 50 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name Balloon2-2

python train.py \
  --dataset_name monocular --root_dir ./data/DynamicFace-2 \
  --img_wh 547 288 --start_end 0 24 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 50 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name DynamicFace-2

python train.py \
  --dataset_name monocular --root_dir ./data/Jumping \
  --img_wh 540 288 --start_end 0 24 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 50 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name Jumping

python train.py \
  --dataset_name monocular --root_dir ./data/Playground \
  --img_wh 540 288 --start_end 0 24 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 50 --batch_size 512 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name Playground

python train.py \
  --dataset_name monocular --root_dir ./data/Skating-2 \
  --img_wh 540 288 --start_end 0 24 \
  --N_samples 128 --N_importance 0 --encode_t \
  --num_epochs 50 --batch_size 512 \
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