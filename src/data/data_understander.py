import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.style.use('ggplot')


if __name__ == "__main__":
    distance = "12m"
    angle = "30"
    vehicle = "van"
    df = pd.read_csv(f'collection/{vehicle}_cam_angle_50_car_angle_{angle}_{distance}.csv')
    plotname = f"Pixel Counts of {vehicle.upper()}, {distance}, {angle} Degree Angle"

    print(df.dtypes)

    ax = plt.figure().gca()
    chart = sns.scatterplot(x="width", y="height", data=df, hue="label")
    chart.set_title(plotname, fontdict={'size': 18})
    chart.set_xlabel("Width of Bounding Box in Pixels", fontdict={'size': 14})
    chart.set_ylabel("Height of Bounding Box in Pixels", fontdict={'size': 14})

    ax.xaxis.get_major_locator().set_params(integer=True)
    plotfile = f"{vehicle}_{angle}_deg_{distance}"
    plt.savefig(f"./plots/{plotfile}.png")
    plt.show()


