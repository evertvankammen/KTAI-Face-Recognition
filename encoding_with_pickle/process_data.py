import os
import pandas as pd

input_file = os.path.join("experiment_tolerance_0.53_desired_width_450_internet_pictures", "frames_information")
dataframe = pd.read_pickle(input_file)
print(dataframe)