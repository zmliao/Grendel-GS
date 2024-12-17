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

import os
import random
import json
from random import randint
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import utils.general_utils as utils
import torch


class Scene:

    gaussians: GaussianModel

    def __init__(
        self, args, gaussians: GaussianModel, load_iteration=None, shuffle=True
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        log_file = utils.get_log_file()

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        utils.log_cpu_memory_usage("before loading images meta data")

        if os.path.exists(
            os.path.join(args.source_path, "sparse")
        ):  # This is the format from colmap.
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval, args.llffhold
            )
        elif "matrixcity" in args.source_path:  # This is for matrixcity
            print("Assuming MatrixCity Dataset")
            scene_info = sceneLoadTypeCallbacks["City"](
                args.source_path,
                args.random_background,
                args.white_background,
                llffhold=args.llffhold,
            )
        else:
            raise ValueError("No valid dataset found in the source path")

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(
                scene_info.train_cameras
            )  # Multi-res consistent random shuffling
            random.shuffle(
                scene_info.test_cameras
            )  # Multi-res consistent random shuffling

        utils.log_cpu_memory_usage("before decoding images")

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # Set image size to global variable
        orig_w, orig_h = (
            scene_info.train_cameras[0].width,
            scene_info.train_cameras[0].height,
        )
        utils.set_img_size(orig_h, orig_w)
        # Dataset size in GB
        dataset_size_in_GB = (
            1.0
            * (len(scene_info.train_cameras) + len(scene_info.test_cameras))
            * orig_w
            * orig_h
            * 3
            / 1e9
        )
        log_file.write(f"Dataset size: {dataset_size_in_GB} GB\n")
        if (
            dataset_size_in_GB < args.preload_dataset_to_gpu_threshold
        ):  # 10GB memory limit for dataset
            log_file.write(
                f"[NOTE]: Preloading dataset({dataset_size_in_GB}GB) to GPU. Disable local_sampling and distributed_dataset_storage.\n"
            )
            print(
                f"[NOTE]: Preloading dataset({dataset_size_in_GB}GB) to GPU. Disable local_sampling and distributed_dataset_storage."
            )
            args.preload_dataset_to_gpu = True
            args.local_sampling = False  # TODO: Preloading dataset to GPU is not compatible with local_sampling and distributed_dataset_storage for now. Fix this.
            args.distributed_dataset_storage = False

        # Train on original resolution, no downsampling in our implementation.
        utils.print_rank_0("Decoding Training Cameras")
        self.train_cameras = None
        self.test_cameras = None
        if args.num_train_cameras >= 0:
            train_cameras = scene_info.train_cameras[: args.num_train_cameras]
        else:
            train_cameras = scene_info.train_cameras
        self.train_cameras = cameraList_from_camInfos(train_cameras, args)
        # output the number of cameras in the training set and image size to the log file
        log_file.write(
            "Number of local training cameras: {}\n".format(len(self.train_cameras))
        )
        if len(self.train_cameras) > 0:
            log_file.write(
                "Image size: {}x{}\n".format(
                    self.train_cameras[0].image_height,
                    self.train_cameras[0].image_width,
                )
            )

        if args.eval:
            utils.print_rank_0("Decoding Test Cameras")
            if args.num_test_cameras >= 0:
                test_cameras = scene_info.test_cameras[: args.num_test_cameras]
            else:
                test_cameras = scene_info.test_cameras
            self.test_cameras = cameraList_from_camInfos(test_cameras, args)
            # output the number of cameras in the training set and image size to the log file
            log_file.write(
                "Number of local test cameras: {}\n".format(len(self.test_cameras))
            )
            if len(self.test_cameras) > 0:
                log_file.write(
                    "Image size: {}x{}\n".format(
                        self.test_cameras[0].image_height,
                        self.test_cameras[0].image_width,
                    )
                )

        utils.check_initial_gpu_memory_usage("after Loading all images")
        utils.log_cpu_memory_usage("after decoding images")

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter)
                )
            )
        elif hasattr(args, "load_ply_path"):
            self.gaussians.load_ply(args.load_ply_path)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        utils.check_initial_gpu_memory_usage("after initializing point cloud")
        utils.log_cpu_memory_usage("after loading initial 3dgs points")

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras

    def log_scene_info_to_file(self, log_file, prefix_str=""):

        # Print shape of gaussians parameters.
        log_file.write("xyz shape: {}\n".format(self.gaussians._xyz.shape))
        log_file.write("f_dc shape: {}\n".format(self.gaussians._features_dc.shape))
        log_file.write("f_rest shape: {}\n".format(self.gaussians._features_rest.shape))
        log_file.write("opacity shape: {}\n".format(self.gaussians._opacity.shape))
        log_file.write("scaling shape: {}\n".format(self.gaussians._scaling.shape))
        log_file.write("rotation shape: {}\n".format(self.gaussians._rotation.shape))


class SceneDataset:
    def __init__(self, cameras):
        self.cameras = cameras
        self.camera_size = len(self.cameras)
        self.sample_camera_idx = []
        for i in range(self.camera_size):
            if self.cameras[i].original_image_backup is not None:
                self.sample_camera_idx.append(i)
        # print("Number of cameras with sample images: ", len(self.sample_camera_idx))

        self.cur_epoch_cameras = []
        self.cur_iteration = 0

        self.iteration_loss = []
        self.epoch_loss = []

        self.log_file = utils.get_log_file()
        self.args = utils.get_args()

        self.last_time_point = None
        self.epoch_time = []
        self.epoch_n_sample = []

    @property
    def cur_epoch(self):
        return len(self.epoch_loss)

    @property
    def cur_iteration_in_epoch(self):
        return len(self.iteration_loss)

    def get_one_camera(self, batched_cameras_uid):
        args = utils.get_args()
        if len(self.cur_epoch_cameras) == 0:
            # start a new epoch
            if args.local_sampling:
                self.cur_epoch_cameras = self.sample_camera_idx.copy()
            else:
                self.cur_epoch_cameras = list(range(self.camera_size))
            # random.shuffle(self.cur_epoch_cameras)
            indices = torch.randperm(len(self.cur_epoch_cameras))
            self.cur_epoch_cameras = [self.cur_epoch_cameras[i] for i in indices]

        self.cur_iteration += 1

        idx = 0
        while self.cameras[self.cur_epoch_cameras[idx]].uid in batched_cameras_uid:
            idx += 1
        camera_idx = self.cur_epoch_cameras.pop(idx)
        viewpoint_cam = self.cameras[camera_idx]
        return camera_idx, viewpoint_cam

    def get_batched_cameras(self, batch_size):
        assert (
            batch_size <= self.camera_size
        ), "Batch size is larger than the number of cameras in the scene."
        batched_cameras = []
        batched_cameras_uid = []
        for i in range(batch_size):
            _, camera = self.get_one_camera(batched_cameras_uid)
            batched_cameras.append(camera)
            batched_cameras_uid.append(camera.uid)

        return batched_cameras

    def get_batched_cameras_idx(self, batch_size):
        assert (
            batch_size <= self.camera_size
        ), "Batch size is larger than the number of cameras in the scene."
        batched_cameras_idx = []
        batched_cameras_uid = []
        for i in range(batch_size):
            idx, camera = self.get_one_camera(batched_cameras_uid)
            batched_cameras_uid.append(camera.uid)
            batched_cameras_idx.append(idx)

        return batched_cameras_idx

    def get_batched_cameras_from_idx(self, idx_list):
        return [self.cameras[i] for i in idx_list]

    def update_losses(self, losses):
        for loss in losses:
            self.iteration_loss.append(loss)
            if len(self.iteration_loss) % self.camera_size == 0:
                self.epoch_loss.append(
                    sum(self.iteration_loss[-self.camera_size :]) / self.camera_size
                )
                self.log_file.write(
                    "epoch {} loss: {}\n".format(
                        len(self.epoch_loss), self.epoch_loss[-1]
                    )
                )
                self.iteration_loss = []
