import argparse
import glob
import os
import shutil
from pathlib import Path

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare data for nsff training')
    parser.add_argument('--data_dir', type=str, help='data root directory', required=True)
    parser.add_argument('--cuda-device',type=str, default='0',help='cuda device to use')
    parser.add_argument(
        '--overwrite', default=False, action='store_true', help='overwrite cache'
    )

    args = parser.parse_args()
    return args


def copy_data(args):
    root_dir = os.path.join(OUT_DIR, os.path.basename(args.data_dir))
    
    image_dir = os.path.join(root_dir, "images")
    mask_dir = os.path.join(root_dir, "masks")
    colmap_dir = os.path.join(root_dir, "sparse/0")

    if not os.path.exists(image_dir) or args.overwrite:
        os.makedirs(image_dir, exist_ok=True)
        files = glob.glob(os.path.join(args.data_dir, "rgb", "2x", "0_*.png"))
        for file in files:
            shutil.copyfile(
                file, os.path.join(image_dir, os.path.basename(file))
            )
    if not os.path.exists(mask_dir) or args.overwrite:
        os.makedirs(mask_dir, exist_ok=True)
        files = glob.glob(os.path.join(args.data_dir, "mask", "2x", "0_*.png"))
        for file in files:
            shutil.copyfile(
                file, os.path.join(mask_dir, os.path.basename(file))
            )
    if not os.path.exists(colmap_dir) or args.overwrite:
        os.makedirs(colmap_dir, exist_ok=True)
        files = glob.glob(os.path.join(args.data_dir, "colmap", "2x", "sparse", "*.bin"))
        for file in files:
            shutil.copyfile(
                file, os.path.join(colmap_dir, os.path.basename(file))
            )


def generate_depth(args):
    root_dir = os.path.join(OUT_DIR, os.path.basename(args.data_dir))
    disp_dir = os.path.join(root_dir, 'disps')
    if not os.path.exists(disp_dir) or args.overwrite:
        cur_dir = Path(__file__).absolute().parent
        os.chdir(f'{str(cur_dir)}/third_party/depth')
        os.environ['MKL_THREADING_LAYER'] = 'GNU'
        #os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
        # os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_device} python run.py --Final --data_dir {args.root_dir}/images --output_dir {args.root_dir}/disps --depthNet 0')
        os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_device} python run_monodepth.py -i {root_dir}/images -o {root_dir}/disps -t dpt_large')
        os.chdir(f'{str(cur_dir)}')


def generate_flow(args):
    root_dir = os.path.join(OUT_DIR, os.path.basename(args.data_dir))
    flow_fw_dir = os.path.join(root_dir, 'flow_fw')
    flow_bw_dir = os.path.join(root_dir, 'flow_bw')
    if not os.path.exists(flow_fw_dir) or not os.path.exists(flow_bw_dir) or args.overwrite:
        cur_dir = Path(__file__).absolute().parent
        os.chdir(f'{str(cur_dir)}/third_party/flow')
        os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_device} python demo.py --model models/raft-things.pth --path {root_dir}')
        os.chdir(f'{str(cur_dir)}')


if __name__ == '__main__':
    args = parse_args()

    copy_data(args)
    generate_depth(args)
    generate_flow(args)
    print('finished!')
