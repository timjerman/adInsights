import numpy as np
import pandas as pd

import dataLoader


def get_add_engagement(df: pd.DataFrame):
    df_engagement = df[['sessionId', 'name']].copy()
    df_engagement['interacted'] = (df_engagement['name'] == 'interaction') | (df_engagement['name'] == 'firstInteraction')
    session_group_engagement = df_engagement.groupby('sessionId').sum()
    add_engagement_rate = 100 * np.count_nonzero(session_group_engagement['interacted']) / np.size(session_group_engagement['interacted'])
    return add_engagement_rate









file_name = 'bdsEventsSample.json'

file_reader = dataLoader.FileReader(file_name)
df = file_reader.loadData(use_already_preprocessed=True, save_preprocessed=True)

# Exercise 2: overall ad engagement rate
add_engagement_rate = get_add_engagement(df)
print('Add engagement rate: {:.2f}%'.format(add_engagement_rate)) # 2.14%



print('Finished!')