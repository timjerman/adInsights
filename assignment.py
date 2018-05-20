import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import ranksums
import datetime as dt

import dataLoader


def get_overall_add_engagement(df: pd.DataFrame):
    df_engagement = df[['sessionId', 'name']].copy()
    df_engagement['interacted'] = (df_engagement['name'] == 'interaction') | (df_engagement['name'] == 'firstInteraction')
    session_group_engagement = df_engagement.groupby('sessionId').sum()
    add_engagement_rate = 100 * np.count_nonzero(session_group_engagement['interacted']) / session_group_engagement['interacted'].size
    return add_engagement_rate


def get_interval_add_engagement(df: pd.DataFrame, interval='10min', attributes=None, absolute=False):
    if attributes is None:
        attributes = {}
    df_engagement = df[['sessionId', 'name', 'timestamp', 'sdk', 'objectClazz']].copy()

    # propagate sdk to all lines of the same session
    session_sdk = df_engagement.loc[df_engagement['sdk'].notnull(), ['sessionId', 'sdk']]
    sdk_dict = dict(zip(session_sdk['sessionId'], session_sdk['sdk']))
    df_engagement['sdk_mapped'] = df_engagement['sessionId'].map(sdk_dict)

    df_engagement['interacted'] = (df_engagement['name'] == 'interaction') | \
                                  (df_engagement['name'] == 'firstInteraction')

    df_engagement['attributes'] = \
        ((df_engagement['sdk_mapped'] == attributes['sdk']) if 'sdk' in attributes else True) & \
        ((df_engagement['objectClazz'] == attributes['objectClazz']) if 'objectClazz' in attributes else True) & \
        ((df_engagement['name'] == attributes['userError']) if 'userError' in attributes else True)

    session_interacted = df_engagement[['sessionId', 'interacted', 'timestamp', 'attributes']].groupby('sessionId').agg(
        {'interacted': 'sum', 'timestamp': 'min', 'attributes': 'sum'})
    session_interacted['attributes_interacted'] = (
            (session_interacted['attributes'] > 0) & (session_interacted['interacted'] > 0)
    ).astype('int')
    session_interacted['interacted'] = (session_interacted['interacted'] > 0).astype('int')
    session_interacted['attributes'] = (session_interacted['attributes'] > 0).astype('int')

    session_interacted['count'] = 1
    session_interacted['timestamp'] = pd.to_datetime(session_interacted['timestamp'], unit='s')
    interval_engagement = session_interacted.groupby([pd.Grouper(key='timestamp', freq=interval)]).sum()
    if len(attributes) == 0:
        interval_engagement['adEngagement'] = 100 * interval_engagement['interacted'] / interval_engagement['count']
    elif absolute:
        interval_engagement['adEngagement'] = 100 * interval_engagement['attributes_interacted'] / interval_engagement[
            'attributes']
    else:
        interval_engagement['adEngagement'] = 100 * interval_engagement['attributes_interacted'] / interval_engagement[
            'count']
    # compute interaction rate using a rolling mean
    #session_interacted['interacted'].rolling(10000).mean()

    return interval_engagement


def plot_interval_add_engagement(add_engagement_interval_rate: pd.DataFrame, absolute=False, save_name=None):
    add_engagement_interval_rate = add_engagement_interval_rate.copy()
    add_engagement_interval_rate = add_engagement_interval_rate.rename(
        columns={'adEngagement': 'Engagement', ('attributes'if absolute else 'count'): 'Count' })
    ax = add_engagement_interval_rate[['Count', 'Engagement']].plot(secondary_y='Engagement', legend=True,
                                                                    color=['#63ea8c', '#8383d3'], linewidth=2.5,
                                                                    figsize=(8, 5))
    ax.set_xlabel('Time of day (2015-04-16)')
    ax.set_ylabel('Count')
    ax.right_ax.set_ylabel('Engagement rate [%]')
    ax.set_title('Ad engagement over time')
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()


def plot_multiple_engagement_rates(index, interaction_count, count, labels, title, save_name=None):

    count = np.array(count)
    interaction_count = np.array(interaction_count)
    interaction_count[count == 0] = 0
    count[count == 0] = 1
    fig, ax = plt.subplots(figsize=(8, 5))
    for arr, lab in zip(100 * np.array(interaction_count) / np.array(count), labels):
        ax.plot(index, arr, label=lab)
    ax.legend(loc='best', framealpha=.5)
    ax.set_xlabel('Time of day (2015-04-16)')
    ax.set_ylabel('Engagement rate [%]')
    ax.set_title(title)
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()


def compute_significance_between_intervals(ad_engagement_rate: pd.DataFrame, timestamp_intervals):
    engagements_per_interval = []
    for i in range(len(timestamp_intervals) - 1):
        engagements_per_interval.append(
            np.array(ad_engagement_rate.loc[(ad_engagement_rate.index < timestamp_intervals[i + 1]) &
                                            (ad_engagement_rate.index >= timestamp_intervals[i]), 'adEngagement'])
        )

    significance_between_intervals = []
    for i in range(len(engagements_per_interval) - 1):
        significance_between_intervals.append(ranksums(engagements_per_interval[i], engagements_per_interval[i + 1]).pvalue)

    return engagements_per_interval, significance_between_intervals


def boxplot_engagement_distribution(ad_engagement_rate: pd.DataFrame, save_name=None):
    timestamp_intervals = pd.date_range(start='2015-4-16 10:00', end='2015-4-16 20:00', freq='60Min')
    engagement_per_interval, _ = compute_significance_between_intervals(
        ad_engagement_rate,
        timestamp_intervals
    )

    _, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(engagement_per_interval, labels=(timestamp_intervals + dt.timedelta(minutes = 30)).strftime('%H:%M')[:-1])
    ax.set_xlabel('Time of day (2015-04-16)')
    ax.set_ylabel('Engagement rate [%]')
    ax.set_title('Hourly ad engagement distribution')
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()


def plot_add_requested_histogram(df: pd.DataFrame, save_name=None):
    ad_requested_timestamps = df.loc[df['name'] == 'adRequested'][['timestamp', 'sessionId']].copy()
    ad_requested_timestamps['timestamp'] = pd.to_datetime(ad_requested_timestamps['timestamp'], unit='s')
    ad_requested_timestamps_grouped = ad_requested_timestamps.groupby(
        pd.Grouper(key='timestamp', freq='30Min')).count()
    ad_requested_timestamps_grouped.index = ad_requested_timestamps_grouped.index.strftime('%H:%M')
    ax = ad_requested_timestamps_grouped.plot(kind='bar', legend=False, color='#8383d3', rot=50, figsize=(8, 5))
    ax.set_xlabel('Time of day (2015-04-16)')
    ax.set_ylabel('Count')
    ax.set_title('Ad requests over time')
    plt.subplots_adjust(bottom=.15)
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()


def plot_interaction_time_histogram(df: pd.DataFrame, save_name=None):
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
    df_start_interaction_time = df_start_interaction_time.drop(['hasScreenShown', 'hasFirstInteraction'], axis=1)

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
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Count')
    ax.set_title('Time to interaction')
    if save_name is not None:
        plt.savefig(save_name[0], bbox_inches='tight')
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
    plt.grid(True, which='both', ls='--')
    ax.set_xlabel('Log10(Time) [s]')
    ax.set_ylabel('Percentage')
    ax.set_title('Time to interaction (whole range)')
    if save_name is not None:
        plt.savefig(save_name[1], bbox_inches='tight')
    plt.show()


file_name = 'bdsEventsSample.json'

file_reader = dataLoader.FileReader(file_name)
df = file_reader.load_data(use_already_preprocessed=True, save_preprocessed=True)

# Exercise 1: plot histogram of ad request timestamps
plot_add_requested_histogram(df, save_name='adRequestedHistogram.png')

# Exercise 2: overall ad engagement rate
add_engagement_rate = get_overall_add_engagement(df)
print('Add engagement rate: {:.2f}%'.format(add_engagement_rate)) # 2.14%

# Exercise 3: start interaction time
plot_interaction_time_histogram(df, save_name=['interactionTimeHistogram.png', 'interactionTimeCumsum.png'])

# Exercise 4.1: interval add engagement rate and statistical significance
ad_engagement_rate_10min = get_interval_add_engagement(df, '10Min')
plot_interval_add_engagement(ad_engagement_rate_10min, save_name='adEngagement10Min.png')
# check significance
ad_engagement_rate_5min = get_interval_add_engagement(df, '5Min')
plot_interval_add_engagement(ad_engagement_rate_5min, save_name='adEngagement5Min.png')

timestamp_intervals = pd.to_datetime(['2015-4-16 10:00', '2015-4-16 11:40', '2015-4-16 16:20', '2015-4-16 20:00'], format='%Y-%m-%d %H:%M')
engagement_per_interval, significance_between_intervals = compute_significance_between_intervals(ad_engagement_rate_5min, timestamp_intervals)

boxplot_engagement_distribution(ad_engagement_rate_5min, save_name='adEngagementBoxPlotDistribution5Min.png')

# Exercise 4.2: interval add engagement rate based on attributes
sdk_unique = df.loc[df['sdk'].notnull(), 'sdk'].unique()
object_clazz_unique = df.loc[df['objectClazz'].notnull(), 'objectClazz'].unique()

# sdk engagement rate
attr_interacted = []
count = []
count_attr = []
for sdk in sdk_unique:
    ad_engagement_rate_10min = get_interval_add_engagement(df, '10Min', {'sdk': sdk}, False)
    attr_interacted.append(ad_engagement_rate_10min['attributes_interacted'])
    count_attr.append(ad_engagement_rate_10min['attributes'])
    count.append(ad_engagement_rate_10min['count'])

plot_multiple_engagement_rates(ad_engagement_rate_10min.index, attr_interacted, count_attr, sdk_unique, 'Absolute SDK engagement over time', save_name='sdkAbsoluteEngagement10Min.png')
plot_multiple_engagement_rates(ad_engagement_rate_10min.index, attr_interacted, count, sdk_unique, 'SDK engagement over time', save_name='sdkEngagement10Min.png')

# objectClazz engagement rate
attr_interacted = []
count = []
count_attr = []
for obj in object_clazz_unique:
    ad_engagement_rate_10min = get_interval_add_engagement(df, '10Min', {'objectClazz': obj}, False)
    attr_interacted.append(ad_engagement_rate_10min['attributes_interacted'])
    count_attr.append(ad_engagement_rate_10min['attributes'])
    count.append(ad_engagement_rate_10min['count'])

plot_multiple_engagement_rates(ad_engagement_rate_10min.index, attr_interacted, count, object_clazz_unique, 'Object engagement over time', save_name='objectEngagement10Min.png')

# error rate over time
ad_engagement_rate_10min_error = get_interval_add_engagement(df, '10Min', {'userError': 'userError'}, False)
plot_multiple_engagement_rates(ad_engagement_rate_10min_error.index, [ad_engagement_rate_10min_error['attributes']],
                               [ad_engagement_rate_10min_error['count']], ['error'], 'Error rate over time',
                               save_name='errorRate10Min.png')

print('Finished!')
