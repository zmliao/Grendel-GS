#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from utils.general_utils import set_args
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    DistributionParams,
    BenchmarkParams,
    DebugParams,
    get_combined_args,
    print_all_args,
    init_args,
)
from utils.general_utils import (
    safe_state,
    set_args,
    init_distributed,
    set_log_file,
    set_cur_iter,
)
import torch.distributed as dist

import utils.general_utils as utils

def readImages(renders_dir, gt_dir):
    print("Reading images from", renders_dir)
    renders = []
    gts = []
    image_names = []
    fnames = sorted(os.listdir(renders_dir))
    total_size = len(fnames)
    rank = utils.DEFAULT_GROUP.rank()
    world_size = utils.DEFAULT_GROUP.size()
    fnames = fnames[rank::world_size]
    for fname in fnames:
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names, total_size


def evaluate(model_paths, mode):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / mode

        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names, total_size = readImages(renders_dir, gt_dir)

            print("Number of renders images:", len(renders))
            print("Number of gt images:", len(gts))
            ssims = []
            psnrs = []
            lpipss = []

            progress_bar = tqdm(range(total_size), desc="Metric evluation progress", disable=(utils.DEFAULT_GROUP.rank() != 0))
            for idx in range(len(renders)):
                progress_bar.update(utils.DEFAULT_GROUP.size())
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips(renders[idx], gts[idx], net_type="vgg"))

            # per_view_dict[scene_dir][method].update(
            #     {
            #         "SSIM": {
            #             name: ssim
            #             for ssim, name in zip(
            #                 torch.tensor(ssims).tolist(), image_names
            #             )
            #         },
            #         "PSNR": {
            #             name: psnr
            #             for psnr, name in zip(
            #                 torch.tensor(psnrs).tolist(), image_names
            #             )
            #         },
            #         "LPIPS": {
            #             name: lp
            #             for lp, name in zip(
            #                 torch.tensor(lpipss).tolist(), image_names
            #             )
            #         },
            #     }
            # )

            # with open(scene_dir + f"/per_view_{mode}.json", "w") as fp:
            #     json.dump(per_view_dict[scene_dir], fp, indent=True)
            
            ssims_sum = torch.tensor(ssims, device="cuda").sum()
            psnrs_sum = torch.tensor(psnrs, device="cuda").sum()
            lpipss_sum = torch.tensor(lpipss, device="cuda").sum()

            dist.reduce(ssims_sum, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(psnrs_sum, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(lpipss_sum, dst=0, op=dist.ReduceOp.SUM)

            if utils.DEFAULT_GROUP.rank() > 0:
                continue

            print("  SSIM : {:>12.7f}".format(ssims_sum.item() / total_size, ".5"))
            print("  PSNR : {:>12.7f}".format(psnrs_sum.item() / total_size, ".5"))
            print("  LPIPS: {:>12.7f}".format(lpipss_sum.item() / total_size, ".5"))
            print("")

            full_dict[scene_dir][method].update(
                {
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item(),
                }
            )

            with open(scene_dir + f"/results_{mode}.json", "w") as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dist_p = DistributionParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    # device = torch.device("cuda:0")
    # torch.cuda.set_device(device)

    # # Set up command line argument parser
    parser.add_argument(
        "--model_paths", required=True, nargs="+", type=str, default=[]
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="train or test",
    )
    args = get_combined_args(parser)
    init_distributed(args)
    utils.set_args(args)

    if utils.WORLD_SIZE > 1:
        torch.distributed.barrier(group=utils.DEFAULT_GROUP)
    safe_state(args.quiet)

    evaluate(args.model_paths, args.mode)
    # args = parser.parse_args()

    # set_args(args)
    # evaluate(args.model_paths, args.mode)
