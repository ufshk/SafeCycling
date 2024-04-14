import pandas as pd
from pathlib import Path
import numpy as np
from scipy import stats
import glob
import os

if __name__ == "__main__":
    cwd = os.getcwd()
    df = pd.DataFrame(columns=["vehicle", "dist", "angle", "mean height", "mean width", "height to width ratio"])
    name = "collection_summary"
    fname = f"~/Documents/SYDE/4B/Capstone/SafeCycling/src/data/collection/summary/{name}.csv"

    all_files = glob.glob(os.path.join(cwd, "collection/raw/*.csv"))
    for file in all_files:
        path = Path(file).parts[-1]
        vehicle = path.split("_")[0]
        angle = path.split("_")[-2]
        dist = path.split("_")[-1]
        dist = dist.split(".")[0]
        dist = int(dist.replace('m', ''))
        # print(path, vehicle, angle, dist)

        read_data = pd.read_csv(file)
        # Select the first quantile
        q1 = read_data['width'].quantile(.25)

        # Select the third quantile
        q3 = read_data['width'].quantile(.75)
        range = q3 - q1

        # Create a mask inbeetween q1 & q3
        mask = read_data['width'].between(q1 - 1.5 * range, q3 + 1.5 * range, inclusive="both")

        # Filtering the initial dataframe with a mask
        iqr = read_data.loc[mask]

        # print(read_data["height"].mean())
        # read_data = np.abs(stats.zscore(read_data["height"])) < 3

        data = {'vehicle': [vehicle], "dist": [dist], "angle": [angle], "mean height": [read_data["height"].mean()],
                "mean width": [read_data["width"].mean()],
                "height to width ratio": [read_data["height"].mean() / read_data["width"].mean()],
                "width to height ratio":  [read_data["width"].mean() / read_data["height"].mean()],
                "q1-1.5*iqr height": [iqr["height"].min()], "q3+1.5*iqr height": [iqr["height"].max()],
                "q1-1.5*iqr width": [iqr["width"].min()], "q3+1.5*iqr width": [iqr["width"].max()]}

        summary = pd.DataFrame(data)
        # print(summary)

        df = pd.concat([summary, df], ignore_index=True)

    df[["angle"]] = df[["angle"]].apply(pd.to_numeric)
    df.to_csv(fname, index=False)

    angles = [0, 30, 60]
    for ang in angles:
        angle_df = df[df["angle"] == ang]
        angle_df = angle_df.sort_values(["vehicle", "dist"], ascending=[True, False])
        angle_df.to_csv(cwd + f"/collection/summary/{ang}_angle_collection_summary.csv", index=False)

    vehicles = ["sedan", "van", "crv"]
    for vehicle in vehicles:
        vehicle_df = df[df["vehicle"] == vehicle]
        vehicle_df = vehicle_df.sort_values(["angle", "dist"], ascending=[True, False])
        vehicle_df.to_csv(cwd + f"/collection/summary/{vehicle}_collection_summary.csv", index=False)

    dists = [3, 6, 9, 12, 15]
    for distance in dists:
        dist_df = df[df["dist"] == distance]
        dist_df = dist_df.sort_values(["vehicle", "angle"], ascending=[True, False])
        dist_df.to_csv(cwd + f"/collection/summary/{distance}m_collection_summary.csv", index=False)




