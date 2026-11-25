"""表面缺陷的经典灰度分割。分割函数只接收原图和算法参数。"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Parameters:
    background_size: int
    closing: bool
    min_area: int

    def __post_init__(self):
        if self.background_size < 3 or self.background_size % 2 == 0:
            raise ValueError("背景窗口须为大于等于 3 的奇数")
        if self.min_area < 1:
            raise ValueError("连通域面积下限须为正整数")


def read_gray(path: Path):
    # imdecode/fromfile 同时支持中文目录和文件名。
    data = np.fromfile(path, dtype=np.uint8)
    if data.size:
        image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return image
    raise ValueError(f"无法读取灰度图像 {path}")


def save_png(path: Path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"图像编码失败 {path}")
    encoded.tofile(path)


def baseline(image):
    threshold, bright = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 外圈约 2.5% 的边框估计背景灰度，四个角只计一次。
    width = max(1, round(min(image.shape) * 0.025))
    border = np.ones(image.shape, dtype=bool)
    border[width:-width, width:-width] = False
    return cv2.bitwise_not(bright) if np.median(image[border]) > threshold else bright


def difference_steps(image, background_size):
    smooth = cv2.GaussianBlur(image, (3, 3), 0)
    # 在浮点域做大窗口滤波，再还原到灰度单位。
    background = cv2.GaussianBlur(smooth.astype(np.float32),
                                  (background_size, background_size), 0)
    background = np.rint(background).clip(0, 255).astype(np.uint8)
    response = cv2.absdiff(smooth, background)
    _, binary = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return smooth, background, response, binary


def clean_mask(binary, closing, min_area):
    mask = binary
    if closing:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if min_area == 1:
        return mask
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # 0 号标签是背景，即便面积很大也不能保留。
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels].astype(np.uint8) * 255


def segment(image, params):
    smooth, background, response, binary = difference_steps(image, params.background_size)
    mask = clean_mask(binary, params.closing, params.min_area)
    return {"source": image, "smooth": smooth, "background": background,
            "response": response, "otsu": binary, "prediction": mask}


def metrics(prediction, truth):
    if prediction.shape != truth.shape:
        raise ValueError(f"预测与标注尺寸不同 {prediction.shape}, {truth.shape}")
    pred, actual = prediction != 0, truth != 0
    tp = int(np.count_nonzero(pred & actual))
    fp = int(np.count_nonzero(pred)) - tp
    fn = int(np.count_nonzero(actual)) - tp
    # 两图均无前景，视为完全一致；仅一方有前景，无定义的比率记零。
    empty = float(tp + fp + fn == 0)
    return {"tp": tp, "fp": fp, "fn": fn,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else empty,
            "precision": tp / (tp + fp) if tp + fp else empty,
            "recall": tp / (tp + fn) if tp + fn else empty}
