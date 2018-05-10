import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dataLoader

file_name = 'bdsEventsSample.json'

file_reader = dataLoader.FileReader(file_name)
df = file_reader.load_data(use_already_preprocessed=True, save_preprocessed=True)

ad_request_timestamps = df.loc[df['name'] == 'adRequested'][['clientTimestamp', 'timestamp']]
timestamp_diff = (ad_request_timestamps['clientTimestamp'] - ad_request_timestamps['timestamp'])

timestamp_differences_analysis = lambda t: print(
    'Percentage of timestamp differences greater than {} seconds: {:.3f}%'.format(t, 100 * timestamp_diff[np.abs(timestamp_diff) >= t].size / timestamp_diff.size)
)

timestamp_differences_analysis(10)
timestamp_differences_analysis(60)
timestamp_differences_analysis(600)
timestamp_differences_analysis(3600)

timestamp_diff = timestamp_diff[np.abs(timestamp_diff) < 10]
timestamp_diff.plot.hist(bins=100)
plt.show()

print('Finished!')

