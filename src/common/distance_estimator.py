def first_distance_estimator(bboxes):
    #
    # :return: estimated distance as a float
    # ^^^ will go in doc string
    """
    Simple distance estimator based on uncalibrated raw pixelcounts from camera
    :param bboxes: bounding boxes of neural network output
    :return: the minimum distance found of any car detected (float)
    """
    hw_ratio = 0.5
    min_dist = None
    min_dist_ratio = None
    score_thresh = 0.5  # could also set on camera directly I think

    for box in bboxes:
        if box.label == "car":
            if box.score > score_thresh:
                ratio = box.height / box.width

                if ratio < hw_ratio:
                    pass
                else:
                    min_dist = 20 - 0.1 * box.width
                    min_dist_ratio = ratio

    return min_dist, min_dist_ratio


def calibrated_img_dist_estimator(bboxes):
    return
