import os
import cv2
import numpy as np
from .onnx_ie import InferenceWithOnnx
from skimage.morphology import remove_small_objects
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def remove_small_cnt_in_mask(img, mask, cnt_threshold=0.2):
    """ 去除游离块"""
    mask = np.squeeze(mask)
    _ret, binary_mask = cv2.threshold(mask, 0.5 * mask.max(), 1, cv2.THRESH_BINARY)
    binary_mask = np.uint8(binary_mask)
    contours, _hierarchy = cv2.findContours(binary_mask, mode=cv2.RETR_EXTERNAL,
                                            method=cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 1:
        contours_area_list = [cv2.contourArea(item) for item in contours]
        contours_largest = max(contours_area_list)
        cnt_area_threshold = int(contours_largest * cnt_threshold)
        binary_mask = binary_mask > 0
        binary_mask = remove_small_objects(binary_mask, cnt_area_threshold, connectivity=1)
        mask = binary_mask * mask
    return img, mask


def pre_processing(input_data, size=320, scale_norm=False):
    if isinstance(size, int):
        size = (size, size)
    input_data = np.squeeze(input_data)
    if np.ndim(input_data) == 2:
        input_data = np.stack([input_data] * 3, axis=-1)
    input_data = cv2.resize(input_data, size)
    input_data = np.expand_dims(input_data, axis=0)[:, :, :, :3]
    if scale_norm:
        input_data = np.float32(input_data)
        input_data = (input_data / 127.5 - 1)
    input_data = np.float32(input_data)
    return input_data


def cnn_output_processing(output_data):
    """output postprocessing after fetching from CNN network
    """
    output_dim_index = -1 if output_data.shape[-1] == 2 else 0  # output dim[-1] == 2 for deeplab, otherwise for SHM
    # output_dim_index = 0
    try:
        output = np.squeeze(output_data[:, :, :, output_dim_index])
    except:
        output = np.squeeze(output_data)
    output = np.squeeze(output)
    return output


# process the hole regions of cloth
def refine_mask_function(original):
    bb_box = [85, 440, 340, 509]
    original[np.where(original > 0)] = 255
    original = (original / 255.0).astype(np.uint8)
    mask_refine = original.copy()
    contours, hierarchy = cv2.findContours(original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logging.info("number of contours:%d" % len(contours))
    cv2.drawContours(original, contours, -1, (0, 255, 255), 2)

    # 找到最大区域并填充
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)
    for i in range(max_idx - 1):
        cv2.fillConvexPoly(original, contours[max_idx - 1], 0)
    cv2.fillConvexPoly(original, contours[max_idx], 1)
    for i in range(bb_box[0], bb_box[1]):
        for k in range(bb_box[2], bb_box[3]):
            if original[k, i] == 1 and mask_refine[k, i] != 1:
                mask_refine[k, i] = 1
    return (mask_refine * 255.0).astype(np.uint8)


class WaifuSeg:
    def __init__(self,
                 model_dir,
                 cnn_inference_size=320,
                 **kwargs):
        # initialization

        self.predict_fn = InferenceWithOnnx(model_dir).predict
        self.cnn_inference_size = cnn_inference_size
        self.kwargs = kwargs
        self._model_dir = model_dir

    def single_inference(self, image: np.ndarray):
        """ predict过程用RGB """
        # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = image
        img_input = img.copy()
        img_input = pre_processing(img_input, size=self.cnn_inference_size)
        img_mask = self.predict_fn(img_input)
        img_mask = cnn_output_processing(img_mask)
        img_mask = cv2.resize(img_mask, (img.shape[1], img.shape[0]))

        # mask去除游离块
        _img, img_mask = remove_small_cnt_in_mask(img, img_mask)
        return np.uint8(img_mask * 255)

    # def batch_inference(self, image_batch):
    #     seg_out_batch = []
    #     for img_pair in image_batch:
    #         img = img_pair['img_content']
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         img_input = img.copy()
    #         img_input = pre_processing(img_input, size=self.cnn_inference_size)
    #         img_mask = self.predict_fn(img_input)
    #         img_mask = cnn_output_processing(img_mask)
    #         img_mask = cv2.resize(img_mask, (img.shape[1], img.shape[0]))
    #         # mask去除游离块
    #         _img, img_mask = remove_small_cnt_in_mask(img, img_mask)
    #         # img_mask = np.ones_like(img_mask)
    #         seg_out_batch.append(np.uint8(img_mask * 255))
    #     return seg_out_batch
