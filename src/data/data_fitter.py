import pandas as pd
import os
import glob
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib

matplotlib.style.use("ggplot")

if __name__ == "__main__":
    cwd = os.getcwd()
    all_files = glob.glob(os.path.join(cwd, "collection/raw cleaned/*.csv"))
    li = []

    for filename in all_files:
        path = Path(filename).parts[-1]
        vehicle = path.split("_")[0]
        angle = path.split("_")[-2]
        dist = path.split("_")[-1]
        dist = dist.split(".")[0]
        dist = int(dist.replace('m', ''))
        df = pd.read_csv(filename, index_col=None, header=0)
        df["distance"] = dist
        df["angle"] = angle
        df["vehicle"] = vehicle
        df[["angle"]] = df[["angle"]].astype(float)
        df[["distance"]] = df[["distance"]].astype(float)
        li.append(df)

    final = pd.concat(li, axis=0, ignore_index=True)

    # rmse = [None] * 21
    # angle_rmse = [None] * 21

    # for degree in range(20):
    angle_X = final["height"] / final["width"]
    angle = final[["angle"]].astype(float)
    X = final[["height", "width"]]
    Y = final[["distance"]]
    X = X.dropna()
    print(X)

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    model_2 = LinearRegression()
    model_2.fit(X, Y)
    Y_pred1 = model_2.predict(X)
    rmse = np.sqrt(np.mean((Y - Y_pred1) ** 2))
    print(rmse)
    # filename = '../models/distest/2ndIteration_RegressionModel_Degree2.joblib'

    angle_model = LinearRegression()
    angle_model.fit(angle_X, angle)

    angle_pred = angle_model.predict(X)
    angle_rmse = np.sqrt(np.mean((angle - angle_pred) ** 2))
    print(angle_rmse)

    filename = '../models/distest/3rdIteration_AngleModel.joblib'
    joblib.dump(angle_model, open(filename, "wb"))

    X["angle_pred"] = angle_pred
    # print(X)
    X_full_poly = poly.fit_transform(X)

    model_3 = LinearRegression()
    model_3.fit(X_full_poly, Y)

    filename = '../models/distest/3rdIteration_RegressionModel_Degree2.joblib'
    joblib.dump(model_3, open(filename, "wb"))

    Y_pred = model_3.predict(X_full_poly)
    rmse = np.sqrt(np.mean((Y - Y_pred) ** 2))
    print(rmse)

    plt.scatter(X["width"], X["angle_pred"])
    plt.show()

    # Plotting the results
    # plt.figure(figsize=(12, 5))
    #
    # plt.subplot(3, 1, 1)
    # plt.scatter(X.iloc[:, 1], Y_pred)
    # plt.xlabel('Height')
    # plt.ylabel('Predicted Distance')
    # plt.title('Height vs Predicted Distance')
    #
    # plt.subplot(3, 1, 2)
    # plt.scatter(X.iloc[:, 0], Y_pred)
    # plt.xlabel('Width')
    # plt.ylabel('Predicted Distance')
    # plt.title('Width vs Predicted Distance')
    #
    # plt.subplot(3, 1, 3)
    # plt.scatter(Y, Y_pred)
    # plt.xlabel('True Distance')
    # plt.ylabel('Predicted Distance')
    # plt.title('True Distance vs Predicted Distance')
    #
    # plt.tight_layout()
    # plt.show()

    # print(Y.shape)
    # print(Y_pred.shape)

    # rmse = np.sqrt(np.mean((Y - Y_pred) ** 2))

    # print(X.dtypes)
    # print(Y.dtypes)

    # mse_Y1 = mean_squared_error(Y[:, 0], Y_pred[:, 0])
    # print("MSE of Angle:", mse_Y1)
    #
    # mse_Y2 = mean_squared_error(Y[:, 1], Y_pred[:, 1])
    # print("MSE of Distance:", mse_Y2)

    # order = range(0, 21)
    # plt.style.use("ggplot")
    # plt.figure()
    # plt.title('RMSE vs Polynomial Order')
    # plt.xlabel('Order of Polynomial')
    # plt.ylabel('RMSE')
    # plt.plot(order, rmse)
    # plt.show()
