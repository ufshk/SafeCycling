import pandas as pd
from timeit import default_timer as timer


def bounding_box_data_collector(bboxes, objclasses, dataframe):
    # gathering data for distance estimation by pixelcount
    if bboxes is not None:
        for bbox in bboxes:
            if bbox.label in objclasses:
                width = bbox.right - bbox.left
                height = bbox.bottom - bbox.top
                dataframe = pd.concat([pd.DataFrame([[bbox.cid, bbox.label, bbox.left, bbox.right,
                                                      bbox.top, bbox.bottom, width, height, bbox.score]],
                                                    columns=dataframe.columns), dataframe], ignore_index=True)

    return dataframe


def testing_data_collector(bboxes, est_dist, dist, angle_est, vehicle, dataframe, ratio, start=None):
    if bboxes is not None:
        for bbox in bboxes:
            if bbox.label == "car":
                width = bbox.right - bbox.left
                height = bbox.bottom - bbox.top

                elapsed_time = None
                if start is not None:
                    end = timer()
                    elapsed_time = end - start

                dataframe = pd.concat([pd.DataFrame([[bbox.cid, bbox.label, vehicle, bbox.left, bbox.right,
                                                      bbox.top, bbox.bottom, width, height, bbox.score, est_dist, angle_est, dist, ratio,
                                                      elapsed_time]],
                                                    columns=dataframe.columns), dataframe], ignore_index=True)

    return dataframe


def write_img(image_data):
    return
