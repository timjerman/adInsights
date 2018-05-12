import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dataLoader


def get_overall_add_engagement(df: pd.DataFrame):
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


def plot_interaction_time_histogram(df: pd.DataFrame):
    # first reduce the dataset only to the needed data
    df_start_interaction_time = df[['sessionId', 'name', 'clientTimestamp']].copy()
    df_start_interaction_time = df_start_interaction_time.loc[df_start_interaction_time['name'].isin(['screenShown', 'firstInteraction'])]
    df_start_interaction_time['hasScreenShown'] = (df_start_interaction_time['name'] == 'screenShown').astype('uint8')
    df_start_interaction_time['hasFirstInteraction'] = (df_start_interaction_time['name'] == 'firstInteraction').astype('uint8')

    # find data that has both sessionId and firstInteraction column
    df_start_interaction_time_grouped = df_start_interaction_time[['sessionId', 'hasScreenShown', 'hasFirstInteraction']].copy().groupby('sessionId').sum().reset_index()
    df_start_interaction_time_grouped = df_start_interaction_time_grouped.loc[
        (df_start_interaction_time_grouped['hasScreenShown'] > 0) & (df_start_interaction_time_grouped['hasFirstInteraction'] > 0)
    ]
    df_start_interaction_time = df_start_interaction_time.loc[df_start_interaction_time['sessionId'].isin(df_start_interaction_time_grouped['sessionId'])]
    df_start_interaction_time = df_start_interaction_time.drop(columns=['hasScreenShown', 'hasFirstInteraction'])

    # find the minimum timestamp for screenShown and firstInteraction events for each session separately
    df_min_timestamps = df_start_interaction_time.groupby(['sessionId', 'name']).min().unstack('name')
    df_min_timestamps.columns = ['clientTimestampInteraction', 'clientTimestampScreen']

    # compute the difference between the two timestamps
    timestamp_diff = (df_min_timestamps['clientTimestampInteraction'] - df_min_timestamps['clientTimestampScreen'])
    timestamp_diff[timestamp_diff>-20].plot.hist(bins=50)
    plt.show()


file_name = 'bdsEventsSample.json'

file_reader = dataLoader.FileReader(file_name)
df = file_reader.load_data(use_already_preprocessed=True, save_preprocessed=True)

# Exercise 1: plot histogram of ad request timestamps
plot_add_requested_histogram(df)

# Exercise 2: overall ad engagement rate
add_engagement_rate = get_overall_add_engagement(df)
print('Add engagement rate: {:.2f}%'.format(add_engagement_rate)) # 2.14%

# Exercise 3: start interaction time
plot_interaction_time_histogram(df)


print('Finished!')