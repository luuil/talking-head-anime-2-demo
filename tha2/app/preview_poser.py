import logging
import os
import sys
from typing import Callable

sys.path.append(os.getcwd())

import torch
import wx

import numpy as np
from moviepy.editor import ImageSequenceClip
from joblib import Parallel, delayed
from PIL import Image

from tha2.poser.poser import Poser
from tha2.util import extract_PIL_image_from_filelike, resize_PIL_image, extract_pytorch_image_from_PIL_image, convert_output_image_from_torch_to_numpy


class MainFrame(wx.Frame):
    def __init__(self, poser: Poser, device: torch.device):
        super().__init__(None, wx.ID_ANY, "Preview Poser")
        self.poser = poser
        self.device = device

        self.wx_source_image = None
        self.torch_source_image = None
        self.source_image_string = "Nothing yet!"

        self.main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(self.main_sizer)
        self.SetAutoLayout(1)
        self.init_left_panel()
        self.init_right_panel()
        self.main_sizer.Fit(self)

        self.timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.update_result_image_panel, self.timer)

        accelerator_table = [
            (wx.ACCEL_CTRL, ord('P'), self.on_next_image),  # Ctrl + P
            (wx.ACCEL_CTRL, ord('S'), self.on_save_video),  # Ctrl + S
            (wx.ACCEL_CTRL | wx.ACCEL_SHIFT, ord('S'), self.on_save_images),  # Ctrl + Shift + S
        ]
        ctrl_ids = wx.NewIdRef(count=len(accelerator_table))

        accelerator_entries = list()
        for entry, ctrl_id in zip(accelerator_table, ctrl_ids):
            key1, key2, func = entry
            self.Bind(wx.EVT_MENU, func, id=ctrl_id)
            accelerator_entries.append((key1, key2, ctrl_id))
        self.SetAcceleratorTable(wx.AcceleratorTable(accelerator_entries))

        self.last_pose = None
        self.last_output_index = self.output_index_choice.GetSelection()
        self.last_output_numpy_image = None

        self.pose_data = None
        self.num_poses = 0
        self.pose_id = 0
        self.images_dict = [dict() for _ in range(self.poser.get_output_length())]
        self.cache_images = False

    def init_left_panel(self):
        # self.control_panel = wx.Panel(self, style=wx.SIMPLE_BORDER, size=(256, -1))
        self.left_panel = wx.Panel(self, style=wx.SIMPLE_BORDER)
        left_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.left_panel.SetSizer(left_panel_sizer)
        self.left_panel.SetAutoLayout(1)

        self.source_image_panel = wx.Panel(self.left_panel, size=(256, 256), style=wx.SIMPLE_BORDER)
        self.source_image_panel.Bind(wx.EVT_PAINT, self.paint_source_image_panel)
        left_panel_sizer.Add(self.source_image_panel, 0, wx.FIXED_MINSIZE)

        self.load_image_button = wx.Button(self.left_panel, wx.ID_ANY, "\nLoad Image\n\n")
        left_panel_sizer.Add(self.load_image_button, 1, wx.EXPAND)
        self.load_image_button.Bind(wx.EVT_BUTTON, self.load_image)

        self.load_poses_button = wx.Button(self.left_panel, wx.ID_ANY, "\nLoad Pose Data\n\n")
        left_panel_sizer.Add(self.load_poses_button, 1, wx.EXPAND)
        self.load_poses_button.Bind(wx.EVT_BUTTON, self.load_poses_data)

        left_panel_sizer.Fit(self.left_panel)
        self.main_sizer.Add(self.left_panel, 0, wx.FIXED_MINSIZE)

    def init_right_panel(self):
        self.right_panel = wx.Panel(self, style=wx.SIMPLE_BORDER)
        right_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.right_panel.SetSizer(right_panel_sizer)
        self.right_panel.SetAutoLayout(1)

        self.result_image_panel = wx.Panel(self.right_panel, size=(256, 256), style=wx.SIMPLE_BORDER)
        right_panel_sizer.Add(self.result_image_panel, 0, wx.FIXED_MINSIZE)

        self.output_index_choice = wx.Choice(
            self.right_panel,
            choices=[str(i) for i in range(self.poser.get_output_length())])
        self.output_index_choice.SetSelection(0)
        right_panel_sizer.Add(self.output_index_choice, 0, wx.EXPAND)

        self.next_image_button = wx.Button(self.right_panel, wx.ID_ANY, "\nPreview (Ctrl+P)\n\n")
        right_panel_sizer.Add(self.next_image_button, 1, wx.EXPAND)
        self.next_image_button.Bind(wx.EVT_BUTTON, self.on_next_image)

        self.save_video_button = wx.Button(self.right_panel, wx.ID_ANY, "\nSave as Video (Ctrl+S)\n\n")
        right_panel_sizer.Add(self.save_video_button, 1, wx.EXPAND)
        self.save_video_button.Bind(wx.EVT_BUTTON, self.on_save_video)

        self.save_images_button = wx.Button(self.right_panel, wx.ID_ANY, "\nSave as Images (Ctrl+Shift+S)\n\n")
        right_panel_sizer.Add(self.save_images_button, 1, wx.EXPAND)
        self.save_images_button.Bind(wx.EVT_BUTTON, self.on_save_images)

        right_panel_sizer.Fit(self.right_panel)
        self.main_sizer.Add(self.right_panel, 0, wx.FIXED_MINSIZE)

    def load_image(self, event: wx.Event):
        dir_name = "data/illust"
        file_dialog = wx.FileDialog(self, "Choose an image", dir_name, "", "*.png", wx.FD_OPEN)
        if file_dialog.ShowModal() == wx.ID_OK:
            image_file_name = os.path.join(file_dialog.GetDirectory(), file_dialog.GetFilename())
            self.image_name = file_dialog.GetFilename()[:-4]
            self.pose_id = 0

            try:
                pil_image = resize_PIL_image(extract_PIL_image_from_filelike(image_file_name))
                w, h = pil_image.size
                if pil_image.mode != 'RGBA':
                    self.source_image_string = "Image must have alpha channel!"
                    self.wx_source_image = None
                    self.torch_source_image = None
                else:
                    self.wx_source_image = wx.Bitmap.FromBufferRGBA(w, h, pil_image.convert("RGBA").tobytes())
                    self.torch_source_image = extract_pytorch_image_from_PIL_image(pil_image).to(self.device)

                self.images_dict = [dict() for _ in range(self.poser.get_output_length())]

                self.Refresh()
            except Exception as e:
                logging.error(e)
                self.show_message(f"Could not load image {image_file_name}")
        file_dialog.Destroy()

    def load_poses_data(self, event: wx.Event):
        dir_name = "data/poses_npy"
        file_dialog = wx.FileDialog(self, "Choose an file", dir_name, "", "*.npy", wx.FD_OPEN)
        if file_dialog.ShowModal() == wx.ID_OK:
            poses_file_name = os.path.join(file_dialog.GetDirectory(), file_dialog.GetFilename())
            try:
                self.pose_data = np.load(poses_file_name).astype(np.float32)
                self.num_poses = len(self.pose_data)

                self.images_dict = [dict() for _ in range(self.poser.get_output_length())]
                self.pose_id = 0
                self.Refresh()

                msg = f'Pose data load successfully, len={self.num_poses}'
                logging.info(msg)
                self.show_message(msg)
            except Exception as e:
                logging.error(e)
                self.show_message(f"Could not load file: {poses_file_name}")
        file_dialog.Destroy()

    def paint_source_image_panel(self, event: wx.Event):
        if self.wx_source_image is None:
            self.draw_source_image_string(self.source_image_panel, use_paint_dc=True)
        else:
            dc = wx.PaintDC(self.source_image_panel)
            dc.Clear()
            dc.DrawBitmap(self.wx_source_image, 0, 0, True)

    def draw_source_image_string(self, widget, use_paint_dc: bool = True):
        if use_paint_dc:
            dc = wx.PaintDC(widget)
        else:
            dc = wx.ClientDC(widget)
        dc.Clear()
        font = wx.Font(wx.FontInfo(14).Family(wx.FONTFAMILY_SWISS))
        dc.SetFont(font)
        w, h = dc.GetTextExtent(self.source_image_string)
        dc.DrawText(self.source_image_string, 128 - w // 2, 128 - h // 2)

    def get_current_pose_id(self):
        if self.pose_data is None:
            return -1
        if self.pose_id >= len(self.pose_data):
            self.pose_id = 0
        return self.pose_id

    def get_current_image(self, pose_id, draw=True):
        current_pose = self.pose_data[pose_id]
        output_index = self.output_index_choice.GetSelection()
        self.last_pose = pose_id

        def _pose_image():
            pose = torch.tensor(current_pose, device=self.device)
            output_image = self.poser.pose(self.torch_source_image, pose, output_index)[0].detach().cpu()
            numpy_image_rgba = np.uint8(np.rint(convert_output_image_from_torch_to_numpy(output_image) * 255.0))
            if self.cache_images:
                self.images_dict[output_index].update({pose_id: numpy_image_rgba})
            logging.info(f'image: output_index={output_index}, no.={pose_id}')
            return numpy_image_rgba

        if self.cache_images:
            if pose_id not in self.images_dict[output_index]:
                numpy_image = _pose_image()
            else:
                numpy_image = self.images_dict[output_index][pose_id]
        else:
            numpy_image = _pose_image()

        if draw:
            self.last_output_numpy_image = numpy_image
            wx_image = wx.ImageFromBuffer(
                numpy_image.shape[0],
                numpy_image.shape[1],
                numpy_image[:, :, 0:3].tobytes(),
                numpy_image[:, :, 3].tobytes())
            wx_bitmap = wx_image.ConvertToBitmap()

            dc = wx.ClientDC(self.result_image_panel)
            dc.Clear()
            dc.DrawBitmap(wx_bitmap, (256 - numpy_image.shape[0]) // 2, (256 - numpy_image.shape[1]) // 2, True)

        if self.cache_images:
            return
        else:
            return numpy_image

    def update_result_image_panel(self, event: wx.Event):
        current_pose_id = self.get_current_pose_id()
        current_output_index = self.output_index_choice.GetSelection()
        if self.last_pose is not None \
                and self.last_pose == current_pose_id \
                and self.last_output_index == current_output_index:
            return
        logging.info(f'{current_pose_id}, {current_output_index}')
        self.last_pose = current_pose_id
        self.last_output_index = current_output_index

        if self.torch_source_image is None:
            self.draw_source_image_string(self.result_image_panel, use_paint_dc=False)
            return

        self.get_current_image(current_pose_id)

    def show_message(self, msg: str, callback: Callable[..., None] = None, caption: int = wx.OK):
        message_dialog = wx.MessageDialog(self, msg, "Message Dialog", caption)
        code = message_dialog.ShowModal()
        if (code == wx.ID_OK or code == wx.ID_YES) and callback is not None:
            callback()
        message_dialog.Destroy()

    def on_next_image(self, event: wx.Event):
        if self.pose_data is None or self.torch_source_image is None:
            msg = "No pose data or image loaded!!!"
            logging.warning(msg)
            self.show_message(msg)
            return

        for idx in range(len(self.pose_data)):
            self.get_current_image(idx)

        # self.pose_id += 1
        # logging.info(f'pose_id = {self.pose_id}, num images = {len(self.images_dict)}')
        #
        # if self.pose_id >= len(self.pose_data):
        #     self.show_message(f"Last pose reached {self.pose_id}, reset to first.")
        #     self.pose_id = 0

    def on_save_video(self, event: wx.Event):
        def _save_video(video_file_name, fps=15):
            if self.cache_images:
                [self.get_current_image(idx) for idx in range(self.num_poses)]
                output_index = self.output_index_choice.GetSelection()
                img_sequence = [self.images_dict[output_index][idx][:, :, 0:3] for idx in
                                self.images_dict[output_index]]  # rgb only
            else:
                img_sequence = [self.get_current_image(idx) for idx in range(self.num_poses)]

            clip = ImageSequenceClip(img_sequence, fps=fps)
            clip.write_videofile(video_file_name)
            clip.close()

        if self.pose_data is None or self.torch_source_image is None:
            msg = "There is no data to save!!!"
            logging.warning(msg)
            self.show_message(msg)
            return

        dir_name = "data/output"
        file_dialog = wx.FileDialog(self, "Choose an file", dir_name, "", "*.mp4", wx.FD_SAVE)
        if file_dialog.ShowModal() == wx.ID_OK:
            file_name = os.path.join(file_dialog.GetDirectory(), file_dialog.GetFilename())
            try:
                if os.path.exists(file_name):
                    self.show_message(f"Override {file_name}",
                                      callback=lambda: _save_video(file_name),
                                      caption=wx.YES_NO | wx.ICON_QUESTION)
                else:
                    _save_video(file_name)
                self.show_message(f"Saved {file_name}")
            except Exception as e:
                logging.error(e)
                self.show_message(f"Could not save {file_name}: {e}")
        file_dialog.Destroy()

    def on_save_images(self, event: wx.Event):
        def _save_img(img, filename):
            img_pil = Image.fromarray(img, 'RGBA')
            img_pil.save(filename)

        def _save_images(file_dir):
            if self.cache_images:
                [self.get_current_image(idx) for idx in range(self.num_poses)]
                output_index = self.output_index_choice.GetSelection()
                img_sequence = [self.images_dict[output_index][idx] for idx in self.images_dict[output_index]]
            else:
                img_sequence = [self.get_current_image(idx) for idx in range(self.num_poses)]

            _results = Parallel(n_jobs=3)(delayed(_save_img)(img, os.path.join(file_dir, f"{idx}.png"))
                                          for (idx, img) in enumerate(img_sequence))

        if self.pose_data is None or self.torch_source_image is None:
            msg = "There is no data to save!!!"
            logging.warning(msg)
            self.show_message(msg)
            return

        dir_name = "data/output"
        dir_dialog = wx.DirDialog(self, "Choose an dir", dir_name, wx.FD_SAVE)
        if dir_dialog.ShowModal() == wx.ID_OK:
            file_dir = dir_dialog.GetPath()
            try:
                _save_images(file_dir)
                self.show_message(f"Saved images to: {file_dir}")
            except Exception as e:
                logging.error(e)
                self.show_message(f"Could not save images to {file_dir}: {e}")
        dir_dialog.Destroy()


if __name__ == "__main__":
    format = '%(asctime)s|%(levelname)s|%(filename)s@%(funcName)s(%(lineno)d): %(message)s'
    logging.basicConfig(format=format, level=logging.DEBUG)

    cuda = torch.device('cuda')
    import tha2.poser.modes.mode_20

    poser = tha2.poser.modes.mode_20.create_poser(cuda)

    app = wx.App()
    main_frame = MainFrame(poser, cuda)
    main_frame.Show(True)
    main_frame.timer.Start(30)
    app.MainLoop()
