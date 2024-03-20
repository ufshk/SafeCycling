def first_distance_estimator(bboxes):
    """
    Simple distance estimator based on uncalibrated raw pixelcounts from camera
    :param bboxes: bounding boxes of neural network output
    :return: the minimum distance found of any car detected (float)
    """
    car_hw_ratio = 0.5
    person_hw_ratio = 2
    min_dist = 100
    min_dist_ratio = None
    print("hello!")

    # somehow alternatively time number of seconds? instead of number of frames
    # in case nothing is in frame forever... don't want to stay high or medium!
    if bboxes is not None:
        for box in bboxes:
            height = box.bottom - box.top
            width = box.right - box.left
            if box.label == "car":
                ratio = height / width
                if ratio < car_hw_ratio:
                    pass
                else:
                    dist = 20 - 0.1 * width
                    if dist < min_dist:
                        min_dist = 20 - 0.1 * width
                        min_dist_ratio = ratio
            elif box.label == "person":
                ratio = height / width

                dist = 25 - 0.05 * width
                if dist < min_dist:
                    min_dist = dist
                    min_dist_ratio = ratio

    return min_dist, min_dist_ratio


def calibrated_img_dist_estimator(bboxes):
    return
