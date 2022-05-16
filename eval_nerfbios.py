import copy
import os
from argparse import ArgumentParser
from collections import defaultdict
from unittest import result

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import metrics
import third_party.lpips.lpips.lpips as lpips
from datasets import dataset_dict, ray_utils
from models.nerf import NeRF, PosEmbedding
from models.rendering import interpolate, render_flow, render_rays
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


@torch.no_grad()
def f_flow(models, embeddings,
      rays, ts, max_t, N_samples, N_importance,
      chunk, xyz_fine=None,
      **kwargs):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    kwargs_ = copy.deepcopy(kwargs)
    for i in range(0, B, chunk):
        if 'view_dir' in kwargs:
            kwargs_['view_dir'] = kwargs['view_dir'][i:i+chunk]
        rendered_ray_chunks = \
            render_flow(models,
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
                        xyz_fine=xyz_fine,
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
    print ("args.root_dir", args.root_dir)

    # dataset
    if args.split == "test_xv":
        kwargs = {
            'root_dir': args.root_dir,
            'split': "test",
            'img_wh': (w, h),
            'start_end': tuple(args.start_end)
        }
    elif args.split in ["eval_train", "eval_test"]:
        kwargs = {
            'root_dir': args.root_dir,
            'split': args.split,
            'img_wh': (w, h),
            'start_end': tuple(args.start_end),
            'raw_root_dir': args.root_dir_raw,
        }
    elif args.split == "eval_kps":
        kwargs = {
            'root_dir': args.root_dir,
            'split': "eval_kps",
            'img_wh': (w, h),
            'start_end': tuple(args.start_end),
            'raw_root_dir': args.root_dir_raw,
        }
    else:
        raise ValueError(args.split)
    dataset = dataset_dict[args.dataset_name](**kwargs)

    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    kwargs = {'K': dataset.K, 'dataset': dataset}
    kwargs['output_transient'] = args.output_transient

    if args.split == "eval_kps":
        kwargs['output_transient_flow'] = ['fw', 'bw', 'disocc']
    else:
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

    elif args.split == "eval_kps":
        from utils.nerfbios import compute_pcks_per_ratio, get_kp_xv_ids
        idxs_xv = get_kp_xv_ids(data_dir=args.root_dir_raw, num_xv_steps=10)
        data_xv = [dataset[idx] for idx in idxs_xv]
        save_dir = os.path.join(dir_name, f'{args.split}')
        os.makedirs(save_dir, exist_ok=True)

        pred_keypoints = []
        gt_keypoints = []
        for data_src in tqdm(data_xv):
            keypoints_src = data_src["keypoints"].clone()
            for data_tgt in data_xv:
                keypoints_tgt = data_tgt["keypoints"].clone()
                results = f_flow(
                    models, embeddings, 
                    data_src["rays"].to(device), data_src["ts"].to(device),
                    dataset.N_frames-1, args.N_samples, args.N_importance,
                    args.chunk, **kwargs
                )
                weights = results["weights_fine"].clone()  # [J, 128]
                xyzs = results["xyzs_fine"].clone()  # [J, 128, 3]
                
                delta_t = data_tgt['t'] - data_src['t']
                # if delta_t == 0:
                #     continue
                flow = "fw" if delta_t > 0 else "bw"
        
                for i in range(abs(delta_t)):
                    xyzs = results["xyzs_%s" % flow]
                    ts = data_src["ts"] + (i + 1) * (-1 if delta_t < 0 else 1)
                    results = f_flow(
                        models, embeddings, 
                        data_src["rays"].to(device), ts.to(device),
                        dataset.N_frames-1, args.N_samples, args.N_importance,
                        args.chunk, xyz_fine=xyzs.to(device), **kwargs
                    )
                # Integrate.
                exp_warped_points_ndc = (weights[..., None] * xyzs).sum(dim=-2)
                # Warp
                K = torch.tensor(dataset.K)
                exp_warped_points = ray_utils.ndc2world(exp_warped_points_ndc, K)
                Ps = dataset.Ps[0, data_tgt['ts']] # (N_rays, 3, 4)
                uvd = Ps[:, :3, :3] @ exp_warped_points.unsqueeze(-1) + Ps[:, :3, 3:]
                warped_kps = uvd[:, :2, 0] / (torch.abs(uvd[:, 2:, 0])+1e-8)
                warped_kps = torch.cat([warped_kps, keypoints_src[..., -1:]], dim=-1)

                # common kpts
                selector = (keypoints_src[..., -1] != 0) & (keypoints_tgt[..., -1] != 0)
                # if selector.sum() == 0:
                #     continue
                warped_kps[~selector] = 0
                keypoints_tgt[~selector] = 0
                
                pred_keypoints.append(warped_kps.cpu().numpy())
                gt_keypoints.append(keypoints_tgt.cpu().numpy())

        pred_keypoints = np.stack(pred_keypoints)
        gt_keypoints = np.stack(gt_keypoints)
        
        pck_ratios = np.linspace(0.1, 0.01, 10)
        pcks_per_ratio = compute_pcks_per_ratio(
            gt_keypoints, pred_keypoints, dataset.img_wh, pck_ratios
        )
        np.savetxt(os.path.join(save_dir, 'pcks.txt'), pcks_per_ratio)

    elif args.split in ["eval_train", "eval_test"]:
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
            
            mask_gt = data["masks"].view(h, w, 3)[..., 0] > 0.5
            if args.split == "eval_test":
                assert mask_gt is not None
            psnrs.append(metrics.psnr(img_gt, img_pred, mask_gt).item())
            # vis
            imageio.imwrite(
                os.path.join(
                    dir_name, 
                    f'{args.split}', 
                    '%s' % data["image_id"]
                ),
                (torch.hstack([img_pred, img_gt]).numpy()*255).astype(np.uint8)
            )
        psnrs = sum(psnrs) / len(psnrs)
        with open(os.path.join(dir_name, f'{args.split}', 'psnr.txt'), 'w') as fp:
            fp.write("psnr: %.3f" % psnrs)                    

    else:
        raise NotImplementedError
        
