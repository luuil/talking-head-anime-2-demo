from typing import List, Union

from torch import Tensor
from tha2.nn.batch_module.batch_input_module import BatchInputModule, BatchInputModuleFactory

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


class CFA(BatchInputModule):
    """
        ref: https://github.com/KumapowerLIU/anime_face_landmark_detection
    """

    def __init__(self,
                 output_channel_num=25,
                 stage_channel_num=128,
                 stage_num=2):
        super(CFA, self).__init__()

        self.output_channel_num = output_channel_num
        self.stage_channel_num = stage_channel_num
        self.stage_num = stage_num

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True))

        self.CFM_features = nn.Sequential(
            # nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, self.stage_channel_num, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        # cascaded regression
        stages = [self.make_stage(self.stage_channel_num)]
        for _ in range(1, self.stage_num):
            stages.append(self.make_stage(self.stage_channel_num + self.output_channel_num))
        self.stages = nn.ModuleList(stages)

    def make_stage(self, nChannels_in):
        layers = list()
        layers.append(nn.Conv2d(nChannels_in, self.stage_channel_num, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(4):
            layers.append(nn.Conv2d(self.stage_channel_num, self.stage_channel_num, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(self.stage_channel_num, self.output_channel_num, kernel_size=3, padding=1))
        return nn.Sequential(*layers)

    def load_weight_from_dict(self):
        model_urls = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        weight_state_dict = model_zoo.load_url(model_urls)
        all_parameter = self.state_dict()
        all_weights = []
        for key, value in all_parameter.items():
            if key in weight_state_dict:
                all_weights.append((key, weight_state_dict[key]))
            else:
                all_weights.append((key, value))
        all_weights = OrderedDict(all_weights)
        self.load_state_dict(all_weights)

    def forward(self, x):
        feature = self.features(x)
        feature = self.CFM_features(feature)
        heatmaps = [self.stages[0](feature)]
        for i in range(1, self.stage_num):
            heatmaps.append(self.stages[i](torch.cat([feature, heatmaps[i - 1]], 1)))
        return heatmaps

    def forward_from_batch(self, batch: List[Tensor]):
        return self.forward(batch[0])


class LandmarkDetectorFactory(BatchInputModuleFactory):
    def __init__(self,
                 output_channel_num=25,
                 stage_channel_num=128,
                 stage_num=2):
        super().__init__()
        self.output_channel_num = output_channel_num
        self.stage_channel_num = stage_channel_num
        self.stage_num = stage_num

    def create(self) -> BatchInputModule:
        return CFA(self.output_channel_num, self.stage_channel_num, self.stage_num)


if __name__ == "__main__":
    cuda = torch.device('cuda')
    landmark_detector = LandmarkDetectorFactory().create().to(cuda)

    image = torch.randn(8, 3, 128, 128, device=cuda)
    outputs = landmark_detector(image)[-1]
    for i in range(len(outputs)):
        print(i, outputs[i].shape)
