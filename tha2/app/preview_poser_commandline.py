# coding: utf-8
# Created by luuil@outlook.com at 5/8/2021

import logging
import os
import sys
from typing import Callable

sys.path.append(os.getcwd())

import torch

import numpy as np
from moviepy.editor import ImageSequenceClip
from joblib import Parallel, delayed
from PIL import Image

from tha2.poser.poser import Poser
from tha2.util import extract_PIL_image_from_filelike, resize_PIL_image, extract_pytorch_image_from_PIL_image, \
    convert_output_image_from_torch_to_numpy


class PreviewPoser(object):
    def __init__(self, poser: Poser, device: torch.device):
        self.poser = poser
        self.device = device

        self.torch_source_image = None
        self.pose_data = None
        self.images = None

    def load_image(self, image_file: str):
        self.images = None
        try:
            pil_image = resize_PIL_image(extract_PIL_image_from_filelike(image_file))
            if pil_image.mode != 'RGBA':
                self.torch_source_image = None
                msg = "Image must have alpha channel!"
                logging.error(msg)
            else:
                self.torch_source_image = extract_pytorch_image_from_PIL_image(pil_image).to(self.device)
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
        assert self.pose_data is not None \
               and self.torch_source_image is not None, "should load image ang pose data first!"
        pose = torch.tensor(current_pose, device=self.device)
        output_image = self.poser.pose(self.torch_source_image, pose, output_index)[0].detach().cpu()
        numpy_image_rgba = np.uint8(np.rint(convert_output_image_from_torch_to_numpy(output_image) * 255.0))
        return numpy_image_rgba

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

    import tha2.poser.modes.mode_20

    images = ['waifu_06.png']
    # pose_files = ['all.npy', 'Paomeiyan_P0.npy', 'Gangimari_Gao.npy']
    pose_files = ['angry.npy']

    output_dir = 'data/output'

    source_image_dir = 'data/illust'
    images = [os.path.join(source_image_dir, name) for name in images]

    pose_file_dir = 'pose_data/pose_sequence_npy'
    pose_files = [os.path.join(pose_file_dir, f) for f in pose_files]

    cuda = torch.device('cuda')
    poser = tha2.poser.modes.mode_20.create_poser(cuda)

    pposer = PreviewPoser(poser, cuda)
    for pose_file in pose_files:
        pposer.load_pose_data(pose_file)
        for path in images:
            pposer.load_image(path)

            image_dir = os.path.join(output_dir, os.path.basename(path)[:-4], os.path.basename(pose_file)[:-4])
            os.makedirs(image_dir, exist_ok=True)
            pposer.save_as_images(image_dir)
            pposer.save_as_video(image_dir + '.mp4')
