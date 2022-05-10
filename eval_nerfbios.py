import copy
import os
from argparse import ArgumentParser
from collections import defaultdict

import cv2
import imageio
import numpy as np
import torch
from tqdm import tqdm

import metrics
import third_party.lpips.lpips.lpips as lpips
from datasets import dataset_dict
from models.nerf import NeRF, PosEmbedding
from models.rendering import interpolate, render_rays
from utils import load_ckpt, visualize_depth

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir_raw', type=str, required=True,
                        help='raw root directory of dataset')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='monocular',
                        choices=['monocular'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='test',
                        help='''test or test_spiral or 
                                test_spiralX or test_fixviewX_interpY''')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[512, 288],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--start_end', nargs="+", type=int, default=[0, 100],
                        help='start frame and end frame')

    parser.add_argument('--use_viewdir', default=False, action="store_true",
                        help='whether to use view dependency in static network')
    parser.add_argument('--N_samples', type=int, default=128,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=0,
                        help='number of additional fine samples')
    parser.add_argument('--chunk', type=int, default=32*1024,
                        help='chunk size to split the input to avoid OOM')

    # NeRF-W parameters
    parser.add_argument('--encode_a', default=False, action="store_true",
                        help='whether to encode appearance (NeRF-A)')
    parser.add_argument('--N_a', type=int, default=48,
                        help='number of embeddings for appearance')
    parser.add_argument('--encode_t', default=False, action="store_true",
                        help='whether to encode transient object (NeRF-U)')
    parser.add_argument('--N_tau', type=int, default=48,
                        help='number of embeddings for transient objects')
    parser.add_argument('--flow_scale', type=float, default=0.2,
                        help='flow scale to multiply to flow network output')
    parser.add_argument('--output_transient', default=False, action="store_true",
                        help='whether to output the full result (static+transient)')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--video_format', type=str, default='mp4',
                        choices=['mp4', 'gif'],
                        help='which format to save')
    parser.add_argument('--fps', type=int, default=10,
                        help='video frame per second')

    parser.add_argument('--save_depth', default=False, action="store_true",
                        help='whether to save depth prediction')
    parser.add_argument('--depth_format', type=str, default='png',
                        help='which format to save')

    return parser.parse_args()


@torch.no_grad()
def f(models, embeddings,
      rays, ts, max_t, N_samples, N_importance,
      chunk,
      **kwargs):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    kwargs_ = copy.deepcopy(kwargs)
    for i in range(0, B, chunk):
        if 'view_dir' in kwargs:
            kwargs_['view_dir'] = kwargs['view_dir'][i:i+chunk]
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        None if ts is None else ts[i:i+chunk],
                        max_t,
                        N_samples,
                        0,
                        0,
                        N_importance,
                        chunk,
                        test_time=True,
                        **kwargs_)
        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]
    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


def save_depth(depth, h, w, dir_name, filename):
    depth_pred = np.nan_to_num(depth.view(h, w).numpy())
    depth_pred_img = visualize_depth(torch.from_numpy(depth_pred)).permute(1, 2, 0).numpy()
    depth_pred_img = (depth_pred_img*255).astype(np.uint8)
    imageio.imwrite(os.path.join(dir_name, filename), depth_pred_img)
    return depth_pred_img


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh

    # dataset
    if args.split == "test_xv":
        kwargs = {
            'root_dir': args.root_dir,
            'split': "test",
            'img_wh': (w, h),
            'start_end': tuple(args.start_end)
        }
    if args.split == "eval_train":
        kwargs = {
            'root_dir': args.root_dir,
            'split': "eval_train",
            'img_wh': (w, h),
            'start_end': tuple(args.start_end)
        }
    else:
        raise ValueError(args.split)
    dataset = dataset_dict[args.dataset_name](**kwargs)

    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    kwargs = {'K': dataset.K, 'dataset': dataset}
    kwargs['output_transient'] = args.output_transient
    kwargs['output_transient_flow'] = []

    embeddings = {'xyz': PosEmbedding(9, 10), 'dir': PosEmbedding(3, 4)}

    if args.encode_a:
        embedding_a = torch.nn.Embedding(dataset.N_frames, args.N_a).to(device)
        embeddings['a'] = embedding_a
        load_ckpt(embedding_a, args.ckpt_path, 'embedding_a')
    if args.encode_t:
        embedding_t = torch.nn.Embedding(dataset.N_frames, args.N_tau).to(device)
        embeddings['t'] = embedding_t
        load_ckpt(embedding_t, args.ckpt_path, 'embedding_t')

    nerf_fine = NeRF(typ='fine',
                     use_viewdir=args.use_viewdir,
                     encode_appearance=args.encode_a,
                     in_channels_a=args.N_a,
                     encode_transient=args.encode_t,
                     in_channels_t=args.N_tau,
                     output_flow=len(kwargs['output_transient_flow'])>0,
                     flow_scale=args.flow_scale).to(device)
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    models = {'fine': nerf_fine}
    if args.N_importance > 0:
        nerf_coarse = NeRF(typ='coarse',
                           use_viewdir=args.use_viewdir,
                           encode_transient=args.encode_t,
                           in_channels_t=args.N_tau).to(device)
        load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
        models['coarse'] = nerf_coarse

    imgs, depths = [], []

    if args.split == "test_xv":
        from utils.nerfbios import get_xv_ids
        idxs_xv = get_xv_ids(data_dir=args.root_dir_raw, num_xv_steps=4)
        data_xv = [dataset[idx] for idx in idxs_xv]

        imgs = []
        for data_ts in data_xv:
            imgs.append([])
            for data_view in data_xv:
                results = f(
                    models, embeddings, 
                    data_view["rays"].to(device), data_ts["ts"].to(device),
                    dataset.N_frames-1, args.N_samples, args.N_importance,
                    args.chunk, **kwargs
                )
                # one image
                img_pred = torch.clip(results['rgb_fine'].view(h, w, 3), 0, 1)
                img_pred_ = (img_pred.numpy()*255).astype(np.uint8)
                imgs[-1].append(img_pred_)
            imgs[-1] = np.hstack(imgs[-1])
        imgs = np.vstack(imgs)
        imageio.imwrite(
            os.path.join(dir_name, f'{args.scene_name}.jpg'),
            imgs
        )

    elif args.split == "eval_train":
        os.makedirs(os.path.join(dir_name, f'{args.split}'), exist_ok=True)
        psnrs = []
        for id, data in tqdm(enumerate(dataset)):
            with torch.no_grad():
                results = f(
                    models, embeddings, 
                    data["rays"].to(device), data["ts"].to(device),
                    dataset.N_frames-1, args.N_samples, args.N_importance,
                    args.chunk, **kwargs
                )
            # one image
            img_gt = data['rgbs'].view(h, w, 3)
            img_pred = torch.clip(results['rgb_fine'].view(h, w, 3), 0, 1)
            psnrs.append(metrics.psnr(img_gt, img_pred).item())
            # vis
            imageio.imwrite(
                os.path.join(
                    dir_name, 
                    f'{args.split}', 
                    '%03d_%04d.jpg' % (data["cam_ids"], id)
                ),
                (torch.hstack([img_pred, img_gt]).numpy()*255).astype(np.uint8)
            )
        psnrs = sum(psnrs) / len(psnrs)
        with open(os.path.join(dir_name, f'{args.split}', 'psnr.txt'), 'w') as fp:
            fp.write("psnr: %.3f" % psnrs)                    

    else:
        raise NotImplementedError
        
