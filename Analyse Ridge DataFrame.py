import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


earth_root = "F:/College_UCC/AM6021- Dissertation/Depth Map Numpy Files/Earth data/earth0/"
earth_dataframe = pd.read_csv(earth_root+"data_frame.csv")
earth_ridge_array = np.array(earth_dataframe["Ridge_Width"])


root = "F:/College_UCC/AM6021- Dissertation/Depth Map Numpy Files/test file/temp/"
df = pd.read_csv(root+"data_frame.csv")
df.sort_values('File_Name')
print(df.columns)
# Specify the string you want to check for
specified_string_list = ["3_20_1_1_2_15000_0.4_10", "3_20_1_1_2_15000_0.4_20", "3_20_1_1_2_15000_0.4_50", "3_20_1_1_2_15000_0.4_100", "3_20_1_1_2_15000_0.4_200", "3_20_1_1_2_15000_0.4_500"]
specified_value_list = [10, 20, 50, 100, 200, 500]

plt.title("Ridge Width to edr")
plt.plot([0, 500], [earth_ridge_array.mean()+earth_ridge_array.std(), earth_ridge_array.mean()+earth_ridge_array.std()], alpha=0.5, c="yellow", label="earth standard deviation")
plt.plot([0, 500], [earth_ridge_array.mean()-earth_ridge_array.std(), earth_ridge_array.mean()-earth_ridge_array.std()], alpha=0.5, c="yellow")
plt.plot([0, 500], [earth_ridge_array.mean(), earth_ridge_array.mean()], alpha=1, c="green", label="earth mean ridge height")
plt.errorbar(x=250, y=earth_ridge_array.mean(), yerr=earth_ridge_array.std(), alpha=0.5, c="green")
mean_values = []
for i in range(len(specified_string_list)):
    specified_string = specified_string_list[i]
    # Filter rows where 'File_Name' starts with the specified string
    simulated_ridge_array = np.array(df[df['File_Name'].str.startswith(specified_string)]["Ridge_Width"])
    print(simulated_ridge_array.shape)
    plt.errorbar(x=specified_value_list[i], y=simulated_ridge_array.mean(), yerr=simulated_ridge_array.std(), c="blue", alpha=0.5, capsize=2)
    mean_values.append(simulated_ridge_array.mean())
plt.plot(specified_value_list, mean_values, c="blue")

plt.ylim(0)
plt.legend()
plt.show()
