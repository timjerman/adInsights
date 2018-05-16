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
    ad_requested_timestamps = df.loc[df['name'] == 'adRequested'][['timestamp', 'sessionId']]
    ad_requested_timestamps['timestamp'] = pd.to_datetime(ad_requested_timestamps['timestamp'], unit='s')
    ad_requested_timestamps_grouped = ad_requested_timestamps.groupby(
        pd.Grouper(key='timestamp', freq='30Min')).count()
    ad_requested_timestamps_grouped.index = ad_requested_timestamps_grouped.index.strftime('%H:%M')
    ax = ad_requested_timestamps_grouped.plot(kind='bar', legend=False, color='#8383d3', rot=50, figsize=(8, 5))
    ax.set_xlabel("Time of day (2015-04-16)")
    ax.set_ylabel("Count")
    ax.set_title('Ad requests over time')
    plt.subplots_adjust(bottom=.15)
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

    # limit to a reasonable range < 1 hour
    timestamp_diff = timestamp_diff[timestamp_diff > 0]
    timestamp_diff = timestamp_diff[timestamp_diff < 3600]

    # timestamp_diff.plot(bins=np.logspace(0, np.log10(3600), 50), kind='hist', loglog=True, xlim=(0, 3600))
    ax = timestamp_diff.plot(bins=np.linspace(0, 20, 50), kind='hist', log=False, xlim=(0, 20), color='#8383d3',
                             figsize=(8, 5))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Count")
    ax.set_title('Time to interaction')
    plt.show()

    print('Median time difference: {:.2f} s'.format(np.median(timestamp_diff)))
    print('Percentage of differences lower than 20 s: {:.1f}%'.format(
        100 * np.count_nonzero(timestamp_diff < 20) / np.size(timestamp_diff)))

    bins = np.logspace(-1, np.log10(3600), 50)
    hist, bin_edges = np.histogram(timestamp_diff, bins=bins)
    hist_cumsum = np.cumsum(hist)
    hist_cumsum = hist_cumsum / np.max(hist_cumsum)

    _, ax = plt.subplots(figsize=(8, 5))
    plt.plot(bins, 100 * np.append([0], hist_cumsum), color='#8383d3')
    plt.xscale('log')
    ax.set_xlim(0.1, 3600)
    ax.set_ylim(0, 100)
    plt.grid(True, which="both", ls="--")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Percentage")
    ax.set_title('Time to interaction (whole range)')
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