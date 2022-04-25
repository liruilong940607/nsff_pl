import argparse
import glob
import os
from pathlib import Path

import cv2
from tqdm import tqdm

from datasets.colmap_utils import Camera, read_model, write_model

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def parse_args():

    parser = argparse.ArgumentParser(description='Prepare data for nsff training')
    parser.add_argument('--data_dir', type=str, help='data root directory', required=True)
    parser.add_argument('--cuda-device',type=str,default='0',help='cuda device to use')

    parser.add_argument('--max-width', type=int, default=1e10, help='max image width')
    parser.add_argument('--max-height', type=int, default=288, help='max image height')
    parser.add_argument(
        '--overwrite', default=False,action='store_true', help='overwrite cache')

    args = parser.parse_args()
    return args


def resize_frames(args):
    vid_name = os.path.basename(args.data_dir)
    root_dir = os.path.join(OUT_DIR, vid_name)
    frames_dir = os.path.join(root_dir, "images")
    os.makedirs(frames_dir, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(args.data_dir, 'dense/images/*.jpg')) +
        glob.glob(os.path.join(args.data_dir, 'dense/images/*.png')))

    print('Resizing images ...')
    factor = 1
    for file_ind, file in enumerate(tqdm(files, desc=f'imresize: {vid_name}')):
        out_frame_fn = f'{frames_dir}/{file_ind:05}.png'

        # skip if both the output frame and the mask exist
        if os.path.exists(out_frame_fn) and not args.overwrite:
            continue

        im = cv2.imread(file)

        # resize if too big
        if im.shape[1] > args.max_width or im.shape[0] > args.max_height:
            factor = max(im.shape[1] / args.max_width, im.shape[0] / args.max_height)
            dsize = (int(im.shape[1] / factor), int(im.shape[0] / factor))
            im = cv2.resize(src=im, dsize=dsize, interpolation=cv2.INTER_AREA)

        cv2.imwrite(out_frame_fn, im)
    return factor


def generate_masks(args):
    # NOT USED. so just hack with rgb images.
    vid_name = os.path.basename(args.data_dir)
    root_dir = os.path.join(OUT_DIR, vid_name)
    mask_dir = os.path.join(root_dir, "masks")
    if not os.path.exists(mask_dir) or args.overwrite:
        os.makedirs(mask_dir, exist_ok=True)
        os.system(f'cp -r {root_dir}/images/*.png {mask_dir}')


def copy_colmap(args, factor: float):
    vid_name = os.path.basename(args.data_dir)
    root_dir = os.path.join(OUT_DIR, vid_name)
    colmap_dir = os.path.join(root_dir, "sparse/0")
    print ("[copying colmap] factor is %.4f" % factor)
    if not os.path.exists(colmap_dir) or args.overwrite:
        os.makedirs(colmap_dir, exist_ok=True)
        path = os.path.join(args.data_dir, "dense/sparse/")
        cameras, images, points3D = read_model(path, ext=".bin")
        for key in cameras.keys():
            cameras[key] = Camera(
                id=cameras[key].id,
                model=cameras[key].model,
                width=int(cameras[key].width / factor),
                height=int(cameras[key].height / factor),
                params=cameras[key].params / factor,
            )
        write_model(cameras, images, points3D, colmap_dir, ext=".bin")

def generate_depth(args):
    vid_name = os.path.basename(args.data_dir)
    root_dir = os.path.join(OUT_DIR, vid_name)
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
    vid_name = os.path.basename(args.data_dir)
    root_dir = os.path.join(OUT_DIR, vid_name)
    flow_fw_dir = os.path.join(root_dir, 'flow_fw')
    flow_bw_dir = os.path.join(root_dir, 'flow_bw')
    if not os.path.exists(flow_fw_dir) or not os.path.exists(flow_bw_dir) or args.overwrite:
        cur_dir = Path(__file__).absolute().parent
        os.chdir(f'{str(cur_dir)}/third_party/flow')
        os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_device} python demo.py --model models/raft-things.pth --path {root_dir}')
        os.chdir(f'{str(cur_dir)}')

if __name__ == '__main__':
    args = parse_args()

    factor = resize_frames(args)
    copy_colmap(args, factor)
    generate_masks(args)
    generate_depth(args)
    generate_flow(args)
    print('finished!')
