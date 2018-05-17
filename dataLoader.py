import numpy as np
import pandas as pd
import os
import time
import datetime


class FileReader:

    PROCESSED_SUFFIX = '_preprocessed'
    SORTED_SUFFIX = '_sorted'
    in_file_name = ''
    processed_file_name = ''

    def __init__(self, in_file_name, suffix=PROCESSED_SUFFIX):
        self.in_file_name = in_file_name
        split_path = os.path.splitext(in_file_name)
        self.processed_file_name = split_path[0] + suffix + split_path[1]

    def read_json_line_data(self, in_file_name):

        return pd.read_json(in_file_name, lines=True, convert_dates=False, convert_axes=False)

    def write_json_line_data(self, df: pd.DataFrame, out_file_name):
        """
        :param out_file_name: file name of the file where to save the json
        :param df: data frame to save
        """
        df.to_json(out_file_name, orient='records', lines=True)

    def preprocess_data(self, df: pd.DataFrame):

        df['clientTimestamp'] = df['clientTimestamp'].fillna(value=0)
        # retain only selected values for column 'name'
        df = df.loc[df['name'].isin(['adRequested', 'screenShown', 'interaction', 'firstInteraction', 'userError', 'pageRequested'])].copy()

        # pageRequested doesn't have a client timestamp -> replace it with timestamp
        df.loc[df['name'] == 'pageRequested', 'clientTimestamp'] = df.loc[df['name'] == 'pageRequested', 'timestamp']

        # select only valid clientTimestamps - retain those that are inside of a 24 hour window to the valid server timestamps
        df = df.loc[
            (df['clientTimestamp'] > time.mktime(datetime.datetime(year=2015, month=4, day=15, hour=10).timetuple())) &
            (df['clientTimestamp'] < time.mktime(datetime.datetime(year=2015, month=4, day=17, hour=20).timetuple()))
        ]

        # remove sessions where purpose is not live and it doesn't contain an adRequested event
        # filtering is quite slow
        # df = df.groupby('sessionId').filter(lambda g: any(g['purpose'] == 'live') and any(g['name'] == 'adRequested'))
        # a faster alternative
        df['purposeIsLive'] = (df['purpose'] == 'live').astype('uint8')
        df['nameIsAdRequested'] = (df['name'] == 'adRequested').astype('uint8')
        df['nameIsPageRequested'] = (df['name'] == 'pageRequested').astype('uint8')
        df_grouped = df[['sessionId', 'purposeIsLive', 'nameIsAdRequested', 'nameIsPageRequested']].copy().groupby('sessionId').sum().reset_index()
        df_grouped = df_grouped.loc[(df_grouped['purposeIsLive'] > 0) & (df_grouped['nameIsAdRequested'] > 0) & (df_grouped['nameIsPageRequested'] > 0)]
        df = df.loc[df['sessionId'].isin(df_grouped['sessionId'])]
        df = df.drop(['purposeIsLive', 'nameIsAdRequested', 'nameIsPageRequested'], axis=1)

        # sort values by server timestamp
        df = df.sort_values(by=['timestamp'])

        return df

    def sort_by_timestamp(self, df: pd.DataFrame):
        return df.sort_values(by=['timestamp'])

    def load_and_preprocess_data(self, process_function, use_already_preprocessed=True, save_preprocessed=True):
        if use_already_preprocessed and os.path.exists(self.processed_file_name):
            return self.read_json_line_data(self.processed_file_name)

        df = self.read_json_line_data(self.in_file_name)
        df = process_function(df)

        if save_preprocessed:
            self.write_json_line_data(df, self.processed_file_name)

        return df

    def load_data(self, use_already_preprocessed=True, save_preprocessed=True):
        df = self.load_and_preprocess_data(
            self.preprocess_data,
            use_already_preprocessed=use_already_preprocessed,
            save_preprocessed=save_preprocessed
        )

        return df

    def sort_input_data(self, use_already_preprocessed=True, save_preprocessed=True):
        df = self.load_and_preprocess_data(
            self.sort_by_timestamp,
            use_already_preprocessed=use_already_preprocessed,
            save_preprocessed=save_preprocessed
        )

        return df

if __name__ == '__main__':

    file_name = 'bdsEventsSample.json'

    file_reader = FileReader(file_name)
    df = file_reader.load_data(use_already_preprocessed=True, save_preprocessed=True)


    print('Finished.')