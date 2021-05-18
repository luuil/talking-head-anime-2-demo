# coding: utf-8
# Created by luuil@outlook.com at 5/11/2021
import logging
import os
import sys

sys.path.append(os.getcwd())

import torch
import numpy as np
from moviepy.editor import ImageSequenceClip
from joblib import Parallel, delayed
from PIL import Image

from tha2.poser.modes.mode_custom import PoserCustom
from tha2.util import extract_PIL_image_from_filelike


class CustomPoser(object):
    def __init__(self, poser: PoserCustom, device: torch.device):
        self.poser = poser
        self.device = device

        self.source_image = None
        self.pose_data = None
        self.images = None

    def load_image(self, image_file: str):
        self.images = None
        try:
            pil_image = extract_PIL_image_from_filelike(image_file)
            self.source_image = np.asarray(pil_image)
        except Exception as e:
            logging.error(e)

    def load_pose_data(self, pose_file: str):
        self.images = None
        try:
            self.pose_data = np.load(pose_file).astype(np.float32)
            logging.info(f'Pose data load successfully, len={len(self.pose_data)}')
        except Exception as e:
            logging.error(f"Could not load file({pose_file}): {e}")

    def pose_image(self, current_pose: np.ndarray, output_index: int):
        def _none_zero():
            return {idx: v for idx, v in enumerate(current_pose) if v != 0}

        assert self.pose_data is not None \
               and self.source_image is not None, "should load image ang pose data first!"
        pose = torch.tensor(current_pose, device=self.device)
        output_image = self.poser.pose(self.source_image, pose, output_index)
        output_image = np.uint8(np.rint(output_image * 255.))
        logging.debug(f'posing with pose(none-zero values): {_none_zero()}')
        return output_image

    def save_as_images(self, path: str, output_index: int = 0, num_thread: int = 3):
        def _save_img(img, filename):
            img_pil = Image.fromarray(img, 'RGBA')
            img_pil.save(filename)

        if self.images is None:
            self.images = [self.pose_image(pose, output_index) for pose in self.pose_data]
        logging.info(f'generate images: {len(self.images)}')
        _results = Parallel(n_jobs=num_thread)(delayed(_save_img)(img, os.path.join(path, f"{idx}.png"))
                                               for (idx, img) in enumerate(self.images))
        logging.info(f'saved images successful: {path}')

    def save_as_video(self, path: str, output_index: int = 0, fps: int = 15):
        if self.images is None:
            self.images = [self.pose_image(pose, output_index) for pose in self.pose_data]

        try:
            clip = ImageSequenceClip(self.images, fps=fps)
            clip.write_videofile(path)
            clip.close()
        except Exception as e:
            logging.error(e)
        logging.info(f'saved video successful: {path}')


if __name__ == "__main__":
    log_fmt = '%(asctime)s|%(levelname)s|%(filename)s@%(funcName)s(%(lineno)d): %(message)s'
    logging.basicConfig(format=log_fmt, level=logging.DEBUG)

    import tha2.poser.modes.mode_custom

    images = ['waifu_01.jpg']
    # pose_files = ['all.npy', 'Paomeiyan_P0.npy', 'Gangimari_Gao.npy']
    pose_files = ['Kunkun_P1.npy']

    output_dir = 'data/output'

    source_image_dir = 'data/illust'
    images = [os.path.join(source_image_dir, name) for name in images]

    pose_file_dir = 'data/poses_npy'
    pose_files = [os.path.join(pose_file_dir, f) for f in pose_files]

    cuda = torch.device('cuda')
    poser = tha2.poser.modes.mode_custom.create_poser('data', cuda)

    cposer = CustomPoser(poser, cuda)
    for pose_file in pose_files:
        cposer.load_pose_data(pose_file)
        for path in images:
            cposer.load_image(path)

            image_dir = os.path.join(output_dir, os.path.basename(path)[:-4], os.path.basename(pose_file)[:-4])
            os.makedirs(image_dir, exist_ok=True)
            cposer.save_as_images(image_dir)
            cposer.save_as_video(image_dir + '.mp4')
