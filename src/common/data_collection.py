import pandas


def bounding_box_data_collector(bboxes):
    # gathering data for distance estimation by pixelcount
    for bbox in bboxes:
        if bbox.label == "car":
            width = bbox.right - bbox.left
            height = bbox.bottom - bbox.top
            # line = bbox.label + "," + str(width) + "," + str(height) + "," + str(bbox.score)


def write_img(image_data):
    return
