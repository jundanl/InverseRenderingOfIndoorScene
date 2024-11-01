import os
import glob
import json

from torch.utils.data import Dataset

from openrooms_utils import image_util


class LightSORDataset(Dataset):
    class ImagePostfix:
        input = "input_srgb.png"
        depth = "depth.exr"
        normal = "normal.exr"
        albedo = "albedo.exr"
        shading = "shading.exr"

    GAMMA = 1. / 2.2
    BACKGROUND_DIR = "background"
    image_postfix = ImagePostfix()
    VALID_DEPTH_RANGE = [0.1, 50.0]

    def __init__(self, data_root, only_sunlight_scene, load_gt_images=[]):
        self.data_root = data_root
        self.only_sunlight_scene = only_sunlight_scene
        self.load_gt_images = [] if load_gt_images is None else load_gt_images

        # Get the scene list
        scene_list = os.listdir(self.data_root)
        scene_list.sort()

        # Image path and output_name
        self.bk_img_list = []
        self.scene_list = []  # exclude not needed scenes
        for scene in scene_list:
            bk_img_dir = os.path.join(self.data_root, scene, self.BACKGROUND_DIR)
            input_srgb_path = glob.glob(os.path.join(bk_img_dir, f"*{self.image_postfix.input}"))
            assert len(input_srgb_path) == 1, \
                f"Found {len(input_srgb_path)} images in {bk_img_dir}: {input_srgb_path}, expected 1"
            input_srgb_path = input_srgb_path[0]
            assert os.path.exists(input_srgb_path), f"Image not found: {input_srgb_path}"

            if self.only_sunlight_scene:
                scene_info_path = os.path.join(bk_img_dir, "scene_information.json")
                assert os.path.exists(scene_info_path), f"Scene information not found: {scene_info_path}."
                scene_info = json.load(open(scene_info_path, "r"))
                if "primary_light_params" in scene_info:
                    primary_light_info = scene_info["primary_light_params"]
                else:
                    primary_light_info = None
                if "type" not in primary_light_info:
                    print(f"Primary light type not found in {scene_info_path}. Skip this scene.")
                    continue
                if primary_light_info["type"].lower() != "sun":
                    print(f"Primary light type is not sun in {scene_info_path}. Skip this scene.")
                    continue
            self.bk_img_list.append(input_srgb_path)
            self.scene_list.append(os.path.join(self.data_root, scene))
        assert len(self.bk_img_list) == len(self.scene_list), "bk_img_list and scene_list should have the same length."
        print(f"Found {len(self.scene_list)} scenes in {self.data_root}.")

    def __len__(self):
        return len(self.bk_img_list)

    def __getitem__(self, idx):
        # Read the input srgb image
        bk_srgb_image_path = self.bk_img_list[idx]
        bk_srgb_image = image_util.read_image(bk_srgb_image_path, "tensor", inf_v=0.0, nan_v=0.0,
                                              preserve_alpha=True)[:3]  # C x H x W

        # Dir
        bk_dir = os.path.dirname(bk_srgb_image_path)
        scene_dir = os.path.dirname(bk_dir)
        # check scene_dir and self.scene_list[idx] are the same, using absolute path
        assert os.path.abspath(scene_dir) == os.path.abspath(self.scene_list[idx]), \
            f"scene_dir and self.scene_list[idx] are not the same: {scene_dir}, {self.scene_list[idx]}"

        # img_name is the name of the scene directory
        img_name = f"{os.path.basename(scene_dir)}_background"

        # Load camera information
        camera_info_path = os.path.join(bk_dir, "camera_params.json")
        assert os.path.exists(camera_info_path), f"Camera information not found: {camera_info_path}."
        camera_info = json.load(open(camera_info_path, "r"))

        # Load scene information
        scene_info_path = os.path.join(bk_dir, "scene_information.json")
        assert os.path.exists(scene_info_path), f"Scene information not found: {scene_info_path}."
        scene_info = json.load(open(scene_info_path, "r"))
        if "object_params" in scene_info:
            object_info = scene_info["object_params"]
        else:
            object_info = None
        if "primary_light_params" in scene_info:
            primary_light_info = scene_info["primary_light_params"]
        else:
            primary_light_info = None

        # Output dict
        out_dict = {
            "index": idx,
            "img_name": img_name,
            "bk_srgb_image_path": bk_srgb_image_path,
            "scene_path": scene_dir,
            "bk_srgb_image": bk_srgb_image,
            "camera_info": camera_info,
            "scene_info": scene_info,
            "object_info": object_info,
            "primary_light_info": primary_light_info,
        }

        # Image path dict
        img_path_dict = {
            "albedo": bk_srgb_image_path.replace(self.image_postfix.input, self.image_postfix.albedo),
            "shading": bk_srgb_image_path.replace(self.image_postfix.input, self.image_postfix.shading),
            "depth": bk_srgb_image_path.replace(self.image_postfix.input, self.image_postfix.depth),
            "normal": bk_srgb_image_path.replace(self.image_postfix.input, self.image_postfix.normal),
        }

        # Load images
        for type in self.load_gt_images:
            type = type.lower()
            assert type in img_path_dict, f"Invalid image type: {type}."
            image = image_util.read_image(img_path_dict[type], "tensor", inf_v=0.0, nan_v=0.0,
                                          preserve_alpha=True)
            if type in ["depth"]:
                image = image[:1]  # 1 x H x W
                image = image * (image >= self.VALID_DEPTH_RANGE[0]) * (image <= self.VALID_DEPTH_RANGE[1])
            else:
                image = image[:3]  # 3 x H x W
            if type == "normal":
                valid_normal_mask = (image ** 2).sum(dim=0, keepdim=True) > 1e-5
                image = image * 2.0 - 1.0  # from [0, 1] to [-1, 1]
                image = image / (image ** 2).sum(dim=0, keepdim=True).sqrt().clamp(min=1e-6)  # normalize
                image = image * valid_normal_mask  # remove invalid normal
            out_dict[f"{type}_image"] = image
        return out_dict