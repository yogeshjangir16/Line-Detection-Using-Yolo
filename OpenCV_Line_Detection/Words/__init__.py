from collections import defaultdict
from dataclasses import dataclass
from typing import List
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import math

@dataclass
class BoundingBox:
    left: int
    top: int
    width: int
    height: int

@dataclass
class DetectionResult:
    cropped_image: np.ndarray
    bounding_box: BoundingBox

def word_segmentation(image: np.ndarray, filter_size: int, std_dev: float, aspect_ratio: float, area_threshold: int) -> List[DetectionResult]:
    assert image.ndim == 2
    assert image.dtype == np.uint8
    filter_kernel = generate_filter_kernel(filter_size, std_dev, aspect_ratio)
    filtered_image = cv2.filter2D(image, -1, filter_kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    threshold_image = 255 - cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    results = []
    contours = cv2.findContours(threshold_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    for contour in contours:
        if cv2.contourArea(contour) < area_threshold:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cropped = image[y:y + h, x:x + w]
        results.append(DetectionResult(cropped, BoundingBox(x, y, w, h)))
    return results

def generate_filter_kernel(size: int, std_dev: float, aspect_ratio: float) -> np.ndarray:
    assert size % 2
    half_size = size // 2
    x_coords = y_coords = np.linspace(-half_size, half_size, size)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    std_dev_y = std_dev
    std_dev_x = std_dev_y * aspect_ratio
    exp_component = np.exp(-x_grid ** 2 / (2 * std_dev_x) - y_grid ** 2 / (2 * std_dev_y))
    x_component = (x_grid ** 2 - std_dev_x ** 2) / (2 * math.pi * std_dev_x ** 5 * std_dev_y)
    y_component = (y_grid ** 2 - std_dev_y ** 2) / (2 * math.pi * std_dev_y ** 5 * std_dev_x)
    kernel = (x_component + y_component) * exp_component
    kernel = kernel / np.sum(kernel)
    return kernel

def resize_image(image: np.ndarray, target_height: int) -> np.ndarray:
    assert image.ndim in (2, 3)
    assert target_height > 0
    assert image.dtype == np.uint8
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_height = image.shape[0]
    scaling_factor = target_height / original_height
    return cv2.resize(image, dsize=None, fx=scaling_factor, fy=scaling_factor)

def group_lines(detections: List[DetectionResult], distance_threshold: float = 0.7, min_detections: int = 2) -> List[List[DetectionResult]]:
    num_boxes = len(detections)
    distance_matrix = np.ones((num_boxes, num_boxes))
    for i in range(num_boxes):
        for j in range(i, num_boxes):
            a = detections[i].bounding_box
            b = detections[j].bounding_box
            if a.top > b.top + b.height or b.top > a.top + a.height:
                continue
            intersection = min(a.top + a.height, b.top + b.height) - max(a.top, b.top)
            union = a.height + b.height - intersection
            iou = np.clip(intersection / union if union > 0 else 0, 0, 1)
            distance_matrix[i, j] = distance_matrix[j, i] = 1 - iou
    clustering = DBSCAN(eps=distance_threshold, min_samples=min_detections, metric='precomputed').fit(distance_matrix)
    grouped = defaultdict(list)
    for idx, cluster_id in enumerate(clustering.labels_):
        if cluster_id == -1:
            continue
        grouped[cluster_id].append(detections[idx])
    sorted_lines = sorted(grouped.values(), key=lambda group: [det.bounding_box.top + det.bounding_box.height / 2 for det in group])
    return sorted_lines

def order_detections(detections: List[DetectionResult], distance_threshold: float = 0.7, min_detections: int = 2) -> List[List[DetectionResult]]:
    lines = group_lines(detections, distance_threshold, min_detections)
    sorted_results = []
    for line in lines:
        sorted_results += sort_single_line(line)
    return sorted_results

def sort_single_line(detections: List[DetectionResult]) -> List[List[DetectionResult]]:
    return [sorted(detections, key=lambda det: det.bounding_box.left + det.bounding_box.width / 2)]
