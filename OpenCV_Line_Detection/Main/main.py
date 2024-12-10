import argparse
from typing import List
import cv2
import matplotlib.pyplot as plt
from path import Path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from word_detector import word_detection, preprocess_image, arrange_by_lines


def retrieve_image_files(directory: Path) -> List[Path]:
    result = []
    for extension in ['*.png', '*.jpg', '*.bmp']:
        result += Path(directory).files(extension)
    return result


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/page'))
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--sigma', type=float, default=11)
    parser.add_argument('--theta', type=float, default=7)
    parser.add_argument('--min_area', type=int, default=100)
    parser.add_argument('--img_height', type=int, default=50)
    args = parser.parse_args()

    for image_path in retrieve_image_files(args.data):
        print(f'Processing file {image_path}')

        image = preprocess_image(cv2.imread(image_path), args.img_height)
        detection_results = word_detection(image,
                                           kernel_size=args.kernel_size,
                                           sigma=args.sigma,
                                           theta=args.theta,
                                           min_area=args.min_area)

        lines_grouped = arrange_by_lines(detection_results)

        plt.imshow(image, cmap='gray')
        color_count = 7
        color_map = plt.cm.get_cmap('rainbow', color_count)
        for line_id, line in enumerate(lines_grouped):
            for word_id, detection in enumerate(line):
                x_coords = [detection.bbox.left, detection.bbox.left, detection.bbox.left + detection.bbox.width, detection.bbox.left + detection.bbox.width, detection.bbox.left]
                y_coords = [detection.bbox.top, detection.bbox.top + detection.bbox.height, detection.bbox.top + detection.bbox.height, detection.bbox.top, detection.bbox.top]
                plt.plot(x_coords, y_coords, c=color_map(line_id % color_count))
                plt.text(detection.bbox.left, detection.bbox.top, f'{line_id}/{word_id}')

        plt.show()


if __name__ == '__main__':
    run()