import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root = "F:/College_UCC/AM6021- Dissertation/Depth Map Numpy Files/"
datapath = "Simulated data/2_15000_0.2_100_2.5/temp/"
df = pd.read_csv(root+datapath+"data_frame.csv").iloc[:, 1:]

cd_value = 3
ntg_value = 10
temp_df = df.loc[df["True_cd"] == cd_value]
results = temp_df.loc[temp_df["True_ntg"] == ntg_value]

means = []
s_devs = []
for cd in range(3, 18, 3):
    for ntg in range(10, 60, 10):
        temp_df = df.loc[df["True_cd"] == cd]
        result = temp_df.loc[temp_df["True_ntg"] == ntg]
        ridge_height = np.array(result.reset_index()["Ridge_Height"])
        means.append(np.mean(ridge_height*100))
        s_devs.append(np.std(ridge_height*100))
error_bars = [1.96 * s_dev for s_dev in s_devs]

# Create a bar plot
colors_section = ['blue'] * 5 + ['red'] * 5
colors = colors_section * int(len(means)/10)


x_ticks = ["10", "20", "30", "40", "50"] * int(len(means)/5)

x = np.arange(len(means))
plt.bar(x, means, yerr=s_devs, capsize=5, align='center', color=colors)
plt.xticks(x, x_ticks)

plt.xlabel('Net to Gross %')
plt.ylabel('Average')
plt.title('                3                6                9                12                15                ')
plt.show()

