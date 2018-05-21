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

########################### Interactions per object ##############################

df_interaction = df[['sessionId', 'name', 'objectClazz']].copy()

df_interaction = df_interaction.loc[
    df_interaction['name'].isin(['firstInteraction'])]
df_interaction['hasFirstInteraction'] = (df_interaction['name'] == 'firstInteraction').astype(
    'uint8')
df_interaction = df_interaction.loc[df_interaction['hasFirstInteraction'] > 0]
interaction_group = df_interaction.groupby('objectClazz').sum().sort_values('hasFirstInteraction')
ax = interaction_group.plot.barh(legend=False, figsize=(8,5), color='#8383d3')
ax.set_xscale('log')
ax.set_xlabel('Log10(Count)')
ax.set_ylabel('Object class')
ax.set_title('Number of interactions per object')
plt.subplots_adjust(left=.25)
plt.savefig('interactions_objects.png', bbox_inches='tight', dpi=300)
print(interaction_group)


########################### Interactions per SDK ##############################

df_interaction = df[['sessionId', 'name', 'sdk']].copy()

session_sdk = df_interaction.loc[df_interaction['sdk'].notnull(), ['sessionId', 'sdk']]
sdk_dict = dict(zip(session_sdk['sessionId'], session_sdk['sdk']))
df_interaction['sdk_mapped'] = df_interaction['sessionId'].map(sdk_dict)

df_interaction = df_interaction.loc[
    df_interaction['name'].isin(['firstInteraction'])]
df_interaction['hasFirstInteraction'] = (df_interaction['name'] == 'firstInteraction').astype(
    'uint8')
df_interaction = df_interaction.loc[df_interaction['hasFirstInteraction'] > 0]
interaction_group = df_interaction.groupby('sdk_mapped').sum().sort_values('hasFirstInteraction')
ax = interaction_group.plot.barh(legend=False, figsize=(8,5), color='#8383d3')
ax.set_xscale('log')
ax.set_xlabel('Log10(Count)')
ax.set_ylabel('SDK')
ax.set_title('Number of interactions per SDK')
plt.subplots_adjust(left=.25)
plt.savefig('interactions_sdk.png', bbox_inches='tight', dpi=300)
print(interaction_group)



########################### Interaction times per specific object ##############################


df_start_interaction_time = df[['sessionId', 'name', 'clientTimestamp', 'objectClazz']].copy()

df_start_interaction_time = df_start_interaction_time.loc[
    df_start_interaction_time['name'].isin(['screenShown', 'firstInteraction'])]
df_start_interaction_time['hasScreenShown'] = (df_start_interaction_time['name'] == 'screenShown').astype('uint8')
df_start_interaction_time['hasFirstInteraction'] = ((df_start_interaction_time['name'] == 'firstInteraction') & (df_start_interaction_time['objectClazz'] == 'Button')).astype(
    'uint8')

# find data that has both sessionId and firstInteraction column
df_start_interaction_time_grouped = df_start_interaction_time[
    ['sessionId', 'hasScreenShown', 'hasFirstInteraction']].copy().groupby('sessionId').sum().reset_index()
df_start_interaction_time_grouped = df_start_interaction_time_grouped.loc[
    (df_start_interaction_time_grouped['hasScreenShown'] > 0) & (
                df_start_interaction_time_grouped['hasFirstInteraction'] > 0)
    ]
df_start_interaction_time = df_start_interaction_time.loc[
    df_start_interaction_time['sessionId'].isin(df_start_interaction_time_grouped['sessionId'])]
df_start_interaction_time = df_start_interaction_time.drop(['hasScreenShown', 'hasFirstInteraction','objectClazz'], axis=1)

# find the minimum timestamp for screenShown and firstInteraction events for each session separately
df_min_timestamps = df_start_interaction_time.groupby(['sessionId', 'name']).min().unstack('name')
df_min_timestamps.columns = ['clientTimestampInteraction', 'clientTimestampScreen']

# compute the difference between the two timestamps
timestamp_diff = (df_min_timestamps['clientTimestampInteraction'] - df_min_timestamps['clientTimestampScreen'])

# limit to a reasonable range < 1 hour
timestamp_diff = timestamp_diff[timestamp_diff > 0]
timestamp_diff = timestamp_diff[timestamp_diff < 3600]

print(np.median(timestamp_diff))
print(np.percentile(timestamp_diff, 80))


########################### Interaction times per specific SDK ###########################

df_start_interaction_time = df[['sessionId', 'name', 'clientTimestamp', 'sdk']].copy()

session_sdk = df_start_interaction_time.loc[df_start_interaction_time['sdk'].notnull(), ['sessionId', 'sdk']]
sdk_dict = dict(zip(session_sdk['sessionId'], session_sdk['sdk']))
df_start_interaction_time['sdk_mapped'] = df_start_interaction_time['sessionId'].map(sdk_dict)

df_start_interaction_time = df_start_interaction_time.loc[
    df_start_interaction_time['name'].isin(['screenShown', 'firstInteraction'])]
df_start_interaction_time['hasScreenShown'] = (df_start_interaction_time['name'] == 'screenShown').astype('uint8')
df_start_interaction_time['hasFirstInteraction'] = ((df_start_interaction_time['name'] == 'firstInteraction') & (df_start_interaction_time['sdk_mapped'] == 'AdMarvel')).astype(
    'uint8')

# find data that has both sessionId and firstInteraction column
df_start_interaction_time_grouped = df_start_interaction_time[
    ['sessionId', 'hasScreenShown', 'hasFirstInteraction']].copy().groupby('sessionId').sum().reset_index()
df_start_interaction_time_grouped = df_start_interaction_time_grouped.loc[
    (df_start_interaction_time_grouped['hasScreenShown'] > 0) & (
                df_start_interaction_time_grouped['hasFirstInteraction'] > 0)
    ]
df_start_interaction_time = df_start_interaction_time.loc[
    df_start_interaction_time['sessionId'].isin(df_start_interaction_time_grouped['sessionId'])]
df_start_interaction_time = df_start_interaction_time.drop(['hasScreenShown', 'hasFirstInteraction','sdk_mapped','sdk'], axis=1)

# find the minimum timestamp for screenShown and firstInteraction events for each session separately
df_min_timestamps = df_start_interaction_time.groupby(['sessionId', 'name']).min().unstack('name')
df_min_timestamps.columns = ['clientTimestampInteraction', 'clientTimestampScreen']

# compute the difference between the two timestamps
timestamp_diff = (df_min_timestamps['clientTimestampInteraction'] - df_min_timestamps['clientTimestampScreen'])

# limit to a reasonable range < 1 hour
timestamp_diff = timestamp_diff[timestamp_diff > 0]
timestamp_diff = timestamp_diff[timestamp_diff < 3600]

print(np.median(timestamp_diff))
print(np.percentile(timestamp_diff, 80))


print('Finished!')

