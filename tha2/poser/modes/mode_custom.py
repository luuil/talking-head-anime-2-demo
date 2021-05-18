# coding: utf-8
# Created by luuil@outlook.com at 5/11/2021
import os
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Callable

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import make_grid

from tha2.nn.landmark_detector.cfa import LandmarkDetectorFactory
from tha2.nn.segmentation.waifu_seg import WaifuSeg
from tha2.nn.super_resolution.sr import SRFactory
from tha2.poser.modes import mode_20
from tha2.poser.modes.mode_20 import KEY_EYEBROW_DECOMPOSER, KEY_EYEBROW_MORPHING_COMBINER, KEY_FACE_MORPHER, \
    KEY_FACE_ROTATER, KEY_COMBINER
from tha2.util import extract_pytorch_image_from_PIL_image, show_cv_image, linear_to_srgb_pytorch, \
    convert_output_image_from_torch_to_numpy

NUM_LANDMARK = 24
LANDMARK_DETECTOR_IMAGE_WIDTH = 128

SEGMENT_SIZE = 512

KEY_SEGMENT = 'segment'
KEY_FACE_DETECTOR = 'face_detector'
KEY_LANDMARK_DETECTOR = 'landmark_detector'
KEY_SUPER_RESOLUTION = 'super_resolution'
KEY_POSING = 'posing'


class ListInputObject(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def run_list(self, input_list: List[Union[np.ndarray, torch.Tensor, int, bool]]):
        pass


class PoserCustom(object):
    def __init__(self, module_loaders: Dict[str, Callable[[], ListInputObject]], device: torch.device):
        self.module_loaders = module_loaders
        self.device = device
        self.modules = None

    def get_modules(self):
        if self.modules is None:
            self.modules = {}
            for key in self.module_loaders:
                module = self.module_loaders[key]()
                self.modules[key] = module
        return self.modules

    def pose(self, image: np.ndarray, pose: torch.Tensor, output_index: Optional[int] = None) -> np.ndarray:
        modules = self.get_modules()

        # landmarks = modules[KEY_LANDMARK_DETECTOR].run_list([image])
        # TODO(2021.05.14): add alignment method by landmarks
        face_rgba = modules[KEY_SEGMENT].run_list([image])
        out = modules[KEY_POSING].run_list([face_rgba, pose, output_index, False])
        out = modules[KEY_SUPER_RESOLUTION].run_list([out])
        return out


class AnimeFaceDetector(object):
    def __init__(self, filename):
        super().__init__()

        self.net = cv2.CascadeClassifier(filename)

    def run(self, image: np.ndarray) -> np.ndarray:
        faces = self.net.detectMultiScale(image)

        # adjust face size
        rois = list()
        image_width = image.shape[1]
        for x_, y_, w_, h_ in faces:
            x = max(x_ - w_ / 8, 0)
            rx = min(x_ + w_ * 9 / 8, image_width)
            y = max(y_ - h_ / 4, 0)
            by = y_ + h_
            w = rx - x
            h = by - y

            roi = x, y, w, h
            roi = tuple(map(int, roi))
            rois.append(roi)

        return np.array(rois)


class AnimeLandmarkDetector(object):
    def __init__(self, filename, device=torch.device('cuda')):
        super().__init__()

        self.device = device

        state_dict = torch.load(filename)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        net = LandmarkDetectorFactory().create()
        net.load_state_dict(state_dict)
        net.to(device)
        net.train(False)
        self.net = net

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        self.transform = transforms.Compose([transforms.ToTensor(), normalize])

    def run(self, image: np.ndarray) -> np.ndarray:
        image_pil = Image.fromarray(image)
        img = image_pil.resize((LANDMARK_DETECTOR_IMAGE_WIDTH, LANDMARK_DETECTOR_IMAGE_WIDTH), Image.BICUBIC)
        img = self.transform(img)
        img = img.unsqueeze(0).to(self.device)

        heat_maps = self.net(img)[-1].cpu().detach().numpy()[0]

        w, h = image_pil.size
        landmarks = list()
        for i in range(NUM_LANDMARK):
            heat_map = cv2.resize(heat_maps[i], (LANDMARK_DETECTOR_IMAGE_WIDTH, LANDMARK_DETECTOR_IMAGE_WIDTH),
                                  interpolation=cv2.INTER_CUBIC)
            landmark = np.unravel_index(np.argmax(heat_map), heat_map.shape)
            landmark_x = landmark[1] * w / LANDMARK_DETECTOR_IMAGE_WIDTH
            landmark_y = landmark[0] * h / LANDMARK_DETECTOR_IMAGE_WIDTH
            landmarks.append((landmark_x, landmark_y))
        return np.array(landmarks)


class LandmarkDetector(ListInputObject):
    def __init__(self, filename_face, filename_landmark, device=torch.device('cuda')):
        super().__init__()

        self.device = device
        self.face_detector = AnimeFaceDetector(filename_face)
        self.landmark_detector = AnimeLandmarkDetector(filename_landmark, device)

    def run(self, image: np.ndarray, debug=True) -> np.ndarray:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        face_rois = self.face_detector.run(image_bgr)

        landmarks = list()
        for roi in face_rois:
            x, y, w, h = roi  # first face
            face = image[y: y + h, x: x + w]

            landmark = self.landmark_detector.run(face)
            landmark = landmark + np.array([x, y])
            landmarks.append(landmark)
            if debug:
                for x_, y_ in landmark:
                    cv2.circle(image_bgr, (int(x_), int(y_)), 2, (0, 0, 255), 1)
                show_cv_image(image_bgr)
        return np.array(landmarks)

    def run_list(self, input_list: List[Union[np.ndarray, torch.Tensor, int, bool]]):
        return self.run(input_list[0])


class SegmentAlpha(ListInputObject):
    def __init__(self, filename):
        super().__init__()

        self.net = WaifuSeg(filename, SEGMENT_SIZE)

    def run(self, image: np.ndarray) -> np.ndarray:
        alpha = self.net.single_inference(image)
        return np.dstack([image, alpha]).astype(np.uint8)

    def run_list(self, input_list: List[Union[np.ndarray, torch.Tensor, int]]):
        return self.run(input_list[0])


class SuperResolution(ListInputObject):
    KEY_PREFIX = 'module.'

    def __init__(self, filename, device=torch.device('cuda')):
        super().__init__()

        self.device = device

        state_dict = torch.load(filename)['params']
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        state_dict = {(k[len(self.KEY_PREFIX):] if k.startswith(self.KEY_PREFIX) else k): state_dict[k] for k in
                      state_dict}

        net = SRFactory().create()
        net.load_state_dict(state_dict)
        net.to(device)
        net.train(False)
        self.net = net

    @staticmethod
    def posing_output_process(x: torch.Tensor):
        torch_image = (x + 1.0) * 0.5
        torch_image[:, 0:3, :, :] = linear_to_srgb_pytorch(torch_image[:, 0:3, :, :])
        return torch_image

    def run(self, image: torch.Tensor, min_max=(0, 1)) -> np.ndarray:
        """ Run super resolution, and convert output into numpy image.
            Values will be clamping to [min, max], and normalized to [0, 1].
        :param image: torch.Tensor. Input image
        :param min_max: tuple[int]. Min and Max values for clamp.
        :return: image after sr.
        """
        image = self.posing_output_process(image)
        image_rgb, image_alpha = image[:, :-1, :, :], image[:, -1, :, :]

        out = self.net(image_rgb)
        out = out.clamp_(*min_max)
        out = (out - min_max[0]) / (min_max[1] - min_max[0])

        dim = out.dim()
        if dim == 4:
            out = make_grid(out, nrow=int(math.sqrt(out.size(0))), normalize=False)
            out_np = out.cpu().detach().numpy().transpose(1, 2, 0)
        elif dim == 3:
            out_np = out.cpu().detach().numpy().transpose(1, 2, 0)
        elif dim == 2:
            out_np = out.cpu().detach().numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {dim}')

        alpha_np = image_alpha.cpu().detach().numpy().transpose(1, 2, 0)
        c_, h, w = out.size()
        alpha_np = cv2.resize(alpha_np, (w, h), interpolation=cv2.INTER_LINEAR)

        return np.dstack([out_np, alpha_np])

    def run_list(self, input_list: List[Union[np.ndarray, torch.Tensor, int, bool]]):
        return self.run(input_list[0])


class Posing(ListInputObject):
    def __init__(self, device=torch.device('cuda'), module_file_names=None):
        super().__init__()

        self.device = device
        self.net = mode_20.create_poser(device, module_file_names)

    def run(self,
            image: np.ndarray,
            pose: torch.Tensor,
            output_index: Optional[int] = None,
            numpy: Optional[bool] = False) -> torch.Tensor:
        image_pil = Image.fromarray(image)
        image_torch = extract_pytorch_image_from_PIL_image(image_pil).to(self.device)
        output_image = self.net.pose(image_torch, pose, output_index)
        if numpy:
            output_image = output_image[0].detach().cpu()
            output_image = convert_output_image_from_torch_to_numpy(output_image)
        return output_image

    def run_list(self, input_list: List[Union[np.ndarray, torch.Tensor, int, bool]]):
        return self.run(input_list[0], input_list[1], input_list[2], input_list[3])


def create_poser(model_dir: str = "data",
                 device: torch.device = torch.device('cuda'),
                 module_file_names: Optional[Dict[str, str]] = None) -> PoserCustom:
    dir = model_dir
    if module_file_names is None:
        module_file_names = {}

    # added models
    if KEY_LANDMARK_DETECTOR not in module_file_names:
        file_name = os.path.join(dir, "landmark.pth.tar")
        module_file_names[KEY_LANDMARK_DETECTOR] = file_name
    if KEY_FACE_DETECTOR not in module_file_names:
        file_name = os.path.join(dir, "lbpcascade_animeface.xml")
        module_file_names[KEY_FACE_DETECTOR] = file_name
    if KEY_SEGMENT not in module_file_names:
        file_name = os.path.join(dir, "segmentation_withbg.onnx")
        module_file_names[KEY_SEGMENT] = file_name
    if KEY_SUPER_RESOLUTION not in module_file_names:
        file_name = os.path.join(dir, "net_g_latest.pth")
        module_file_names[KEY_SUPER_RESOLUTION] = file_name

    # posing models
    if KEY_EYEBROW_DECOMPOSER not in module_file_names:
        file_name = os.path.join(dir, "eyebrow_decomposer.pt")
        module_file_names[KEY_EYEBROW_DECOMPOSER] = file_name
    if KEY_EYEBROW_MORPHING_COMBINER not in module_file_names:
        file_name = os.path.join(dir, "eyebrow_morphing_combiner.pt")
        module_file_names[KEY_EYEBROW_MORPHING_COMBINER] = file_name
    if KEY_FACE_MORPHER not in module_file_names:
        file_name = os.path.join(dir, "face_morpher.pt")
        module_file_names[KEY_FACE_MORPHER] = file_name
    if KEY_FACE_ROTATER not in module_file_names:
        file_name = os.path.join(dir, "two_algo_face_rotator.pt")
        module_file_names[KEY_FACE_ROTATER] = file_name
    if KEY_COMBINER not in module_file_names:
        file_name = os.path.join(dir, "combiner.pt")
        module_file_names[KEY_COMBINER] = file_name

    loaders = {
        KEY_LANDMARK_DETECTOR: lambda: LandmarkDetector(module_file_names[KEY_FACE_DETECTOR],
                                                        module_file_names[KEY_LANDMARK_DETECTOR],
                                                        device),
        KEY_SEGMENT: lambda: SegmentAlpha(module_file_names[KEY_SEGMENT]),
        KEY_SUPER_RESOLUTION: lambda: SuperResolution(module_file_names[KEY_SUPER_RESOLUTION], device),
        KEY_POSING: lambda: Posing(device, module_file_names),
    }
    return PoserCustom(loaders, device)
