import numpy as np

from word_detector import word_detection, preprocess_image, arrange_by_lines, BoundingBox


def synthetic_test():
    ground_truths = [BoundingBox(100, 100, 100, 25), BoundingBox(300, 110, 50, 15), BoundingBox(100, 300, 50, 20)]
    ground_truth_lines = [[ground_truths[0], ground_truths[1]], [ground_truths[2]]]

    image = np.ones([512, 512], np.uint8) * 255
    for gt in ground_truths:
        image[gt.top:gt.top + gt.height, gt.left: gt.left + gt.width] = 0

    image = preprocess_image(image, 512)
    detections = word_detection(image, kernel_size=25, sigma=25, theta=5, min_area=100)

    assert len(detections) == len(ground_truths)

    detected_lines = arrange_by_lines(detections, min_words_per_line=1)

    assert len(detected_lines) == len(ground_truth_lines)

    for detected_line, ground_truth_line in zip(detected_lines, ground_truth_lines):
        assert len(detected_line) == len(ground_truth_line)

        for detected_word, ground_truth_word in zip(detected_line, ground_truth_line):
            threshold = 10
            assert abs(detected_word.bbox.left - ground_truth_word.left) < threshold
            assert abs(detected_word.bbox.top - ground_truth_word.top) < threshold
            assert abs(detected_word.bbox.width - ground_truth_word.width) < threshold
            assert abs(detected_word.bbox.height - ground_truth_word.height) < threshold
