import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import glob
import os

matplotlib.style.use('ggplot')


def f(row):
    if row['width'] <= 200:
        val = 2
    else:
        val = 1
    return val


if __name__ == "__main__":
    cwd = os.getcwd()
    all_files = glob.glob(os.path.join(cwd, "collection/crv*.csv"))
    df = pd.concat((pd.read_csv(f, index_col=False) for f in all_files))

    # df.drop(["Unnamed: 0"])
    # df.reset_index(drop=True)

    print(df.columns)
    # df.drop(df.columns[0], axis=1, inplace=True)
    # df.to_csv("filtered_crv_data_90_angle.csv")
    df['distance'] = df.apply(f, axis=1)

    x = df["width"]
    y = df["distance"]

    # find line of best fit
    a, b = np.polyfit(x, y, 1)

    ax = plt.figure().gca()
    chart = sns.scatterplot(x="width", y="distance", data=df, hue="label")

    # add line of best fit to plot
    plt.plot(x, a * x + b, color='steelblue', linestyle='--', linewidth=2)

    # add fitted regression equation to plot
    print('y = ' + '{:.2f}'.format(b) + ' + {:.2f}'.format(a) + 'x')

    chart.set_title("Distance to Person by Pixel Count of Width, 1m to 2m", fontdict={'size': 18})
    chart.set_xlabel("Width of Bounding Box in Pixels", fontdict={'size': 14})
    chart.set_ylabel("Distance to Person", fontdict={'size': 14})

    ax.xaxis.get_major_locator().set_params(integer=True)
    plt.savefig("./plots/same_lane_crv.png")
    plt.show()