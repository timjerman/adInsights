import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dataLoader


def get_add_engagement(df: pd.DataFrame):
    df_engagement = df[['sessionId', 'name']].copy()
    df_engagement['interacted'] = (df_engagement['name'] == 'interaction') | (df_engagement['name'] == 'firstInteraction')
    session_group_engagement = df_engagement.groupby('sessionId').sum()
    add_engagement_rate = 100 * np.count_nonzero(session_group_engagement['interacted']) / session_group_engagement['interacted'].size
    return add_engagement_rate


def plot_add_requested_histogram(df: pd.DataFrame):
    ad_requested_timestamps = df.loc[df['name'] == 'adRequested'][['clientTimestamp', 'timestamp']]
    ad_requested_timestamps = df.loc[df['name'] == 'adRequested'][['clientTimestamp', 'timestamp']]
    ad_requested_timestamps['clientTimestamp'] = pd.to_datetime(ad_requested_timestamps['clientTimestamp'], unit='s')
    ad_requested_timestamps['timestamp'] = pd.to_datetime(ad_requested_timestamps['timestamp'], unit='s')
    ad_requested_timestamps.groupby(pd.Grouper(key='clientTimestamp', freq='30Min')).count().plot(kind='bar')
    plt.show()






file_name = 'bdsEventsSample.json'

file_reader = dataLoader.FileReader(file_name)
df = file_reader.load_data(use_already_preprocessed=True, save_preprocessed=True)

# Exercise 1: plot histogram of ad request timestamps
plot_add_requested_histogram(df)

# Exercise 2: overall ad engagement rate
add_engagement_rate = get_add_engagement(df)
print('Add engagement rate: {:.2f}%'.format(add_engagement_rate)) # 2.14%



print('Finished!')