class SingleViewDataset(Dataset):
    GAMMA = 1./2.2
    IMAGE_DIR = "background"
    INPUT_POSTFIX = "input_srgb.png"
    MIN_VALID_DEPTH = 1e-2
    MAX_VALID_DEPTH = 50
    env_size = (160, 120)

    def __init__(self, data_root, cfg, output_root, img_w=320, img_h=240):
        self.img_w = img_w
        self.img_h = img_h
        self.cfg = cfg
        self.d_type = cfg.d_type
        self.img_size = (img_w, img_h)
        self.data_root = data_root
        self.output_root = output_root

        # colmap depth is so big compared to openrooms(meter)
        self.max_depth_type = 'adaptive'
        self.depth_max_scale = 10.0
        print(self.max_depth_type, self.depth_max_scale)

        # Get the scene list
        self.scene_list = os.listdir(self.data_root)
        self.scene_list.sort()
        print(f"Read scene from {self.data_root}, {len(self.scene_list)} in total")
        # print(f"Scene list: {self.scene_list}")

        # Voxel grid
        self.xy_offset = 1.3
        x, y, z = np.meshgrid(np.arange(self.cfg.VSGEncoder.vsg_res),
                              np.arange(self.cfg.VSGEncoder.vsg_res),
                              np.arange(self.cfg.VSGEncoder.vsg_res // 2), indexing='xy')
        x = x.astype(dtype=np.float32) + 0.5  # add half pixel
        y = y.astype(dtype=np.float32) + 0.5
        z = z.astype(dtype=np.float32) + 0.5
        z = z / (self.cfg.VSGEncoder.vsg_res // 2)
        x = self.xy_offset * (2.0 * x / self.cfg.VSGEncoder.vsg_res - 1)
        y = self.xy_offset * (2.0 * y / self.cfg.VSGEncoder.vsg_res - 1)
        self.voxel_grid = [x, y, z]

        # Image path and output_name
        self.img_list = []
        self.out_dirs = []
        for scene in self.scene_list:
            img_dir = os.path.join(self.data_root, scene, self.IMAGE_DIR)
            img_path = glob.glob(os.path.join(img_dir, f"*{self.INPUT_POSTFIX}"))
            assert len(img_path) == 1, f"Found {len(img_path)} images in {img_dir}: {img_path}, expected 1"
            img_path = img_path[0]
            assert os.path.exists(img_path), f"Image not found: {img_path}"
            self.img_list.append(img_path)
            self.out_dirs.append(os.path.join(self.output_root, scene))
            os.makedirs(self.out_dirs[-1], exist_ok=True)
        # for i in range(len(self.img_list)):
        #     print(f"img_list: {self.img_list[i]}, out_dirs: {self.out_dirs[i]}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        batch = {}
        batch['outname'] = self.out_dirs[idx]

        # Read the input image
        input_img_path = self.img_list[idx]
        im = image_util.read_image(input_img_path, "numpy")
        im = cv2.resize(im, self.img_size, interpolation=cv2.INTER_AREA)
        im = image_util.srgb_to_rgb(im, gamma=self.GAMMA).transpose([2, 0, 1])  # to linear RGB. C x H x W
        batch['i'] = im  # input image
        batch['m'] = np.ones_like(im[:1])  # mask

        # Load depth map
        depth_path = input_img_path.replace("input_srgb.png", "depth.exr")
        depth_map = image_util.read_image(depth_path, "numpy", inf_v=0.0, nan_v=0.0, preserve_alpha=True)[:, :, 0]
        depth_map = cv2.resize(depth_map, self.img_size, interpolation=cv2.INTER_AREA)[..., None].transpose([2, 0, 1])  # depth map
        mask_depth = (depth_map >= self.MIN_VALID_DEPTH) * (depth_map <= self.MAX_VALID_DEPTH)  # remove invalid depth
        depth_map = depth_map * mask_depth
        if self.max_depth_type == 'adaptive':
            # This computation refers to cds-mvsnet/colmap2mvsnet.py: processing_single_scene_my, Line 392 - Line 397
            max_ratio = 0.1
            min_ratio = 0.1
            zs = depth_map[depth_map > self.MIN_VALID_DEPTH].flatten()
            if len(zs < 5):
                min_depth, max_depth = 0.0, self.depth_max_scale
            else:
                zs_sorted = np.sort(zs)  # small to large
                num_max = max(5, int(len(zs) * max_ratio))
                num_min = max(1, int(len(zs) * min_ratio))
                max_depth = 1.0 * sum(zs_sorted[-num_max:]) / len(zs_sorted[-num_max:])
                min_depth = 1.0 * sum(zs_sorted[:num_min]) / len(zs_sorted[:num_min])
                # cam_mats[1, -1, int(target_idx) - 1] = max_depth
                # cam_mats[0, -1, int(target_idx) - 1] = min_depth
                # print(f"adaptive max depth: {depth_max}, min depth: {depth_min}")
        else:
            assert False, f"unknown max_depth_type: {self.max_depth_type}"
        batch['cds_dn'] = np.clip(depth_map / max_depth, 0, 1)  # normalized depth map
        grad_x = cv2.Sobel(batch['cds_dn'][0], -1, 1, 0)
        grad_y = cv2.Sobel(batch['cds_dn'][0], -1, 0, 1)
        batch['cds_dg'] = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)[None]  # depth gradient

        # Load normal map
        normal_path = input_img_path.replace("input_srgb.png", "normal.exr")
        normal_map = image_util.read_image(normal_path, "numpy", inf_v=0.0, nan_v=0.0, preserve_alpha=True)[:, :, :3]
        mask_normal = (normal_map ** 2).sum(axis=-1) > 1e-3  # H x W
        normal_map = cv2.resize(normal_map, self.img_size, interpolation=cv2.INTER_AREA).transpose([2, 0, 1])  # normal map. 3 x H x W
        mask_normal = cv2.resize(mask_normal.astype(np.float32), self.img_size, interpolation=cv2.INTER_AREA)[None]  # mask. 1 x H x W
        mask_normal = (mask_normal > 0.99).astype(np.float32)
        normal_map = normal_map * 2.0 - 1.0  # from [0, 1] to [-1, 1]
        normal_map = normal_map / np.linalg.norm(normal_map, axis=0, keepdims=True).clip(min=1e-6)  # normalize
        normal_map = normal_map.clip(min=-1.0, max=1.0) * mask_normal  # remove invalid normal
        batch['normal'] = normal_map

        # Load albedo map
        albedo_path = input_img_path.replace("input_srgb.png", "albedo.exr")
        albedo_map = image_util.read_image(albedo_path, "numpy", inf_v=0.0, nan_v=0.0, preserve_alpha=True)[:, :, :3]
        albedo_map = cv2.resize(albedo_map, self.img_size, interpolation=cv2.INTER_AREA).transpose([2, 0, 1])  # albedo map
        albedo_map = albedo_map.clip(min=0.0, max=1.0)
        batch['albedo'] = albedo_map

        # Confidence map
        # confidence_map = np.ones((1, self.img_size[1], self.img_size[0]), dtype=np.float32)
        confidence_map = mask_depth * mask_normal #* self.MAX_VALID_DEPTH/2.0/depth_map.clip(min=1e-6)
        assert confidence_map.shape[0] == 1, f"confidence_map shape: {confidence_map.shape}"
        batch['cds_conf'] = confidence_map

        # Set fixed camera matrix and remove other views
        cam_mats = np.zeros((3, 6, 1), dtype=np.float32)
        cam_mats[:3, :3, :] = np.eye(3).reshape(3, 3, 1)  # identity rotation
        cam_mats[:3, 3:4, :] = np.zeros((3, 1, 1))  # zero translation
        cam_mats[0, 4, :] = batch['i'].shape[1]  # H
        cam_mats[1, 4, :] = batch['i'].shape[2]  # W
        cam_mats[0, 5, :] = min_depth
        cam_mats[1, 5, :] = max_depth
        cam_mats[2, 4, :] = 225.8  # focal length
        cam_mats[2, 5, :] = 225.8  # focal length
        target_idx = 1  # remove other views

        # Camera intrinsics
        poses_hwf_bounds = cam_mats[..., int(target_idx) - 1]
        h, w, f = poses_hwf_bounds[:, -2]
        intrinsic = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=float).astype(np.float32)
        batch['cam'] = intrinsic
        batch['hwf'] = np.array([h, w, f])

        # Voxel grid
        if hasattr(self, 'voxel_grid'):
            fov_x = intrinsic[0, 2] / intrinsic[0, 0]
            fov_y = intrinsic[1, 2] / intrinsic[0, 0]
            batch['bb'] = np.array([self.xy_offset * fov_x, self.xy_offset * fov_y, 1.05], dtype=np.float32)
            x = self.voxel_grid[0] * fov_x
            y = self.voxel_grid[1] * fov_y
            z = self.voxel_grid[2] * 1.05
            batch['voxel_grid_front'] = np.stack([x, y, z], axis=-1)

        # Scale the camera matrix
        # depth_scale = 1.0
        # if scene's max depth is larger than depth_max_scale, we scale down depth.
        depth_scale = max(1.0, max_depth / self.depth_max_scale)
        cam_mats[:, 3, :] /= depth_scale

        src_c2w_list = []
        src_int_list = []
        rgb_list = []
        depthest_list = []
        fac = self.env_size[1] / self.img_size[1]
        if True:
            im = cv2.imread(input_img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            im = cv2.resize(im, self.env_size, interpolation=cv2.INTER_AREA)
            im = ldr2hdr(im[..., ::-1].astype(np.float32) / 255.0)
            rgb_list.append(im)

            idx = 1
            poses_hwf_bounds = cam_mats[..., int(idx) - 1]
            src_c2w_list.append(np34_to_44(poses_hwf_bounds[:, :4]))
            cy2, cx2, fx = poses_hwf_bounds[:, -2]
            fy = poses_hwf_bounds[-1, -1]
            if fy == 0:
                fy = fx
            intrinsic = np.array([[fx * fac, 0, cx2 / 2 * fac], [0, fy * fac, cy2 / 2 * fac], [0, 0, 1]], dtype=float)
            src_int_list.append(intrinsic)
            if self.d_type == 'cds':
                d = image_util.read_image(depth_path, "numpy", inf_v=0.0, nan_v=0.0, preserve_alpha=True)[:, :,
                            0]
                d = cv2.resize(d, self.env_size, interpolation=cv2.INTER_AREA)[..., None]  # depth map
                # depth = loadImage(name.format('cdsdepthest', 'dat'), 'd', self.env_size, False)
                # depth = loadImage(cds_depth_name, 'd', self.env_size, False)
                d = d * (d >= self.MIN_VALID_DEPTH) * (d <= self.MAX_VALID_DEPTH)  # remove invalid depth
                d = d / depth_scale
            else:
                assert False, "not implemented"
            depthest_list.append(d)

        batch['all_i'] = np.stack(rgb_list, axis=0).transpose([0, 3, 1, 2])
        batch['all_cam'] = np.stack(src_int_list, axis=0).astype(np.float32)
        w2target = np.linalg.inv(src_c2w_list[0])
        batch['c2w'] = (w2target @ np.stack(src_c2w_list, 0)).astype(np.float32)
        batch['all_depth'] = np.stack(depthest_list, axis=0).transpose([0, 3, 1, 2])
        # for key, item in batch.items():
        #     if torch.is_tensor(item) or isinstance(item, np.ndarray):
        #         print(f"key: {key}, item: {item.shape}")
        return batch