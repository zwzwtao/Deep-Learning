import tensorflow as tf
from yad2k.models.keras_yolo import yolo_boxes_to_corners
from yolo_utils import scale_boxes
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold=0.6):
    """
    Filters YOLO boxes by thresholding on object and class confidence.

    19 x 19 cells with 5 anchor boxes, number of classes is 80

    box_confidence(probabilities of whether it's an object): [b, 19, 19, 5, 1]
    boxes(coordinates of each box): [b, 19, 19, 5, 4]
    box_class_probs: [b, 19, 19, 5, 80]
    """

    # broadcasting of box_confidence: [b, 19, 19, 5, 1] -> [b, 19, 19, 5, 80]
    box_scores = box_confidence * box_class_probs

    # find the class for each anchor box
    # [b, 19, 19, 5, 80] -> [b, 19, 19, 5]
    box_classes = tf.argmax(box_scores, axis=-1)
    box_class_scores = tf.reduce_max(box_scores, axis=-1)

    # [b, 19, 19, 5]
    filtering_mask = box_class_scores >= score_threshold

    # keep boxes which may contain objects
    # [b, 19, 19, 5] -> [num_after_filtering]
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    # [b, 19, 19, 5, 4] -> [num_after_filtering, 4]
    boxes = tf.boolean_mask(boxes, filtering_mask)
    # [b, 19, 19, 5] -> [num_after_filtering]
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes


# box_confidence = tf.random.normal([7, 19, 19, 5, 1], mean=1, stddev=4, seed=1)
# boxes = tf.random.normal([7, 19, 19, 5, 4], mean=1, stddev=4, seed=1)
# box_class_probs = tf.random.normal([7, 19, 19, 5, 80], mean=1, stddev=4, seed=1)
# scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs)
# print(scores.shape, boxes.shape, classes.shape)


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    scores: [num_after_filtering]
    boxes: [num_after_filtering, 4]
    classes: [num_after_filtering]
    max_boxes: max number of boxes needed to bound objects
    """

    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)

    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)

    return scores, boxes, classes


def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
    """
    yolo_output:
        box_confidence: [b, 19, 19, 5, 1]
        box_xy: [b, 19, 19, 5, 2]
        box_wh: [b, 19, 19, 5, 2]
        box_class_probs: [b, 19, 19, 5, 80]
    input shape of image: [b, 608, 608]
    scrore_threshold: the argument 'threshold' in yolo_filter_boxes()
    """

    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # covert to corner coordinates format
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes

# yolo_ouputs = (
#     tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed=1),
#     tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
#     tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
#     tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed=1)
# )
# scores, boxes, classes = yolo_eval(yolo_ouputs)
# print(scores.shape, boxes.shape, classes.shape)
