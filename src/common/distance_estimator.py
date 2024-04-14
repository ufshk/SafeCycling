import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd


def load_model(iteration=2):
    angle_model = None
    if iteration == 2:
        filename = "models/distest/2ndIteration_RegressionModel_Degree2.joblib"
        with open(filename, "rb") as f:
            model = joblib.load(f)
    elif iteration == 3:
        filename = "models/distest/3rdIteration_RegressionModel_Degree2.joblib"
        with open(filename, "rb") as f:
            model = joblib.load(f)

        filename = "models/distest/3rdIteration_AngleModel.joblib"
        with open(filename, "rb") as f:
            angle_model = joblib.load(f)
    return model, angle_model


def symposium_distance_estimator(bboxes):
    """
    Simple distance estimator based on uncalibrated raw images from camera
    :param bboxes: bounding boxes of neural network output
    :return: the minimum distance found of any car or person detected (float)
    """
    car_hw_ratio = 0.5
    # person_hw_ratio = 2
    min_dist = 100
    min_dist_ratio = None
    # print("hello!")

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


def first_iteration_dist_estimator(bboxes):
    """
    Simple distance estimator based on uncalibrated raw images from camera
    :param bboxes: bounding boxes of neural network output
    :return: the minimum distance found of any car or person detected (float)
    """
    car_hw_ratios = {"0": [0.61, 1], "30": [0.49, 0.61], "60": [0, 0.49]}
    # print(car_hw_ratios["0"][0])
    min_dist = 20
    est_angle = None
    min_ratio = 100
    ratio = None
    bbox = None

    if bboxes is not None:
        for box in bboxes:
            # print("Ratio:", car_hw_ratios["0"][0] < ratio <= car_hw_ratios["0"][1])
            # print(box.label)
            if box.label == "car":
                height = box.bottom - box.top
                width = box.right - box.left
                ratio = height / width
                # angle is 0
                if car_hw_ratios["0"][0] <= ratio <= car_hw_ratios["0"][1]:
                    # from lowest q1-1.5*iqr score amongst readings to highest q3+1.5*iqr
                    # if gap between q3+1.5*iqr to next distance then the middle is used
                    if 84 <= width < 96.5:
                        est_dist = 15
                        if est_dist < min_dist:
                            min_dist = est_dist
                            est_angle = 0
                    if 96.5 <= width < 115:
                        est_dist = 12
                        if est_dist < min_dist:
                            min_dist = est_dist
                            est_angle = 0
                    if 115 <= width < 164.5:
                        est_dist = 9
                        if est_dist < min_dist:
                            min_dist = est_dist
                            est_angle = 0
                    if 164.5 <= width < 257:
                        est_dist = 6
                        if est_dist < min_dist:
                            min_dist = est_dist
                            est_angle = 0
                    if width >= 257:
                        est_dist = 3
                        if est_dist < min_dist:
                            min_dist = est_dist
                            est_angle = 0
                #
                # angle is ~30
                if car_hw_ratios["30"][0] <= ratio < car_hw_ratios["30"][1]:
                    # from lowest q1-1.5*iqr score amongst readings to highest q3+1.5*iqr
                    # if gap between q3+1.5*iqr to next distance then the middle is used
                    if 120 <= width < 135:
                        est_dist = 15
                        if est_dist < min_dist:
                            min_dist = est_dist
                            est_angle = 30
                    if 135 <= width < 166:
                        est_dist = 12
                        if est_dist < min_dist:
                            min_dist = est_dist
                            est_angle = 30
                    if 166 <= width < 208.5:
                        est_dist = 9
                        if est_dist < min_dist:
                            min_dist = est_dist
                            est_angle = 30
                    if 208.5 <= width < 289:
                        est_dist = 6
                        if est_dist < min_dist:
                            min_dist = est_dist
                            est_angle = 30
                    if width >= 289:
                        est_dist = 3
                        if est_dist < min_dist:
                            min_dist = est_dist
                            est_angle = 30

                # angle is ~60
                if car_hw_ratios["60"][0] <= ratio < car_hw_ratios["60"][1]:
                    # from lowest q1-1.5*iqr score amongst readings to highest q3+1.5*iqr
                    # if gap between q3+1.5*iqr to next distance then the middle is used
                    if 150 <= width < 175.5:  # van readings were larger... error! maybe see about a predictor?
                        est_dist = 15
                        if est_dist < min_dist:
                            min_dist = est_dist
                            est_angle = 60
                    if 175.5 <= width < 212.5:
                        est_dist = 12
                        if est_dist < min_dist:
                            min_dist = est_dist
                            est_angle = 60
                    if 212.5 <= width < 280:
                        est_dist = 9
                        if est_dist < min_dist:
                            min_dist = est_dist
                            est_angle = 60
                    if 280 <= width < 396:
                        est_dist = 6
                        if est_dist < min_dist:
                            min_dist = est_dist
                            est_angle = 60
                    if width >= 396:
                        est_dist = 3
                        if est_dist < min_dist:
                            min_dist = est_dist
                            est_angle = 60

    return min_dist, est_angle, ratio


def second_iteration_dist_estimator(bboxes, model):
    min_dist = 20

    if bboxes is not None:
        for box in bboxes:
            if box.label == "car":
                poly = PolynomialFeatures(degree=2)
                height = box.bottom - box.top
                width = box.right - box.left
                # print(width)
                X = np.array([height, width]).reshape(1, -1)
                X_poly = poly.fit_transform(X)
                dist = model.predict(X_poly)
                # print(dist)
                if dist < min_dist:
                    min_dist = dist

    return min_dist


def third_iteration_dist_estimator(bboxes, model, angle_model):
    min_dist = 20
    min_angle = None
    ratio = None

    if bboxes is not None:
        for box in bboxes:
            if box.label == "car":
                # apply polynomial feature to height, width
                poly = PolynomialFeatures(degree=2)
                height = box.bottom - box.top
                width = box.right - box.left
                data = {"height": [height], "width": [width]}
                X = pd.DataFrame(data=data)
                X_poly = poly.fit_transform(X)

                # derive angle
                angle = angle_model.predict(X_poly)
                X["angle"] = angle
                # print(angle.shape)

                X_poly_all = poly.fit_transform(X)

                # predict distance
                dist = model.predict(X_poly_all)
                if dist < min_dist:
                    min_dist = dist
                    min_angle = angle
                    ratio = height / width

    return min_dist, min_angle, ratio


def dist_estimator(bboxes, iteration, model, angle_model):
    """
    calls chosen underlying estimator
    :param model:
    :param iteration: which estimator to use
    :param bboxes: bounding boxes from neural network for given frame
    :return: estimated distance of chosen iteration estimator
    """
    est_dist = 20
    est_angle = None
    ratio = None
    if iteration == 1:
        est_dist, est_angle, ratio = first_iteration_dist_estimator(bboxes)
    if iteration == 2:
        est_dist = second_iteration_dist_estimator(bboxes, model)
        # print(est_dist)
    if iteration == 3:
        est_dist, est_angle, ratio = third_iteration_dist_estimator(bboxes, model, angle_model)
        # print(est_dist)
    return est_dist, est_angle, ratio
