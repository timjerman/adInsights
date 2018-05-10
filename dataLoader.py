import numpy as np
import pandas as pd
import os


class FileReader:

    PROCESSED_SUFFIX = '_preprocessed'
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

        # df = df.dropna(subset=['objectClazz'])
        # tmp = df.groupby('name').count()
        # df = df.loc[df['sdk'].notNull()]# == 'live']
        # df = df.groupby('sessionId').filter(lambda g: any(g['purpose'] == 'live')).filter(lambda g: any(g['name'] == 'adRequested'))

        # retain only selected values for column 'name'
        df = df.loc[df['name'].isin(['adRequested', 'screenShown', 'interaction', 'firstInteraction', 'userError', 'pageRequested'])]

        # select only valid timestamps
        df = df.loc[(df['clientTimestamp'] > 1429092000) & (df['clientTimestamp'] < 1429300800)]

        # remove sessions where purpose is not live and it doesn't contain an adRequested event
        # filtering is quite slow
        # df = df.groupby('sessionId').filter(lambda g: any(g['purpose'] == 'live') and any(g['name'] == 'adRequested'))
        # a faster alternative
        df['purposeIsLive'] = (df['purpose'] == 'live').astype('uint16')
        df['nameIsAdRequested'] = (df['name'] == 'adRequested').astype('uint8')
        df_grouped = df[['sessionId', 'purposeIsLive', 'nameIsAdRequested']].copy().groupby('sessionId').sum().reset_index()
        df_grouped = df_grouped.loc[(df_grouped['purposeIsLive'] > 0) & (df_grouped['nameIsAdRequested'] > 0)]
        df = df.loc[df['sessionId'].isin(df_grouped['sessionId'])]
        df = df.drop(columns=['purposeIsLive', 'nameIsAdRequested'])

        return df

    def load_data(self, save_preprocessed=True, use_already_preprocessed=True):

        if use_already_preprocessed and os.path.exists(self.processed_file_name):
            return self.read_json_line_data(self.processed_file_name)

        df = self.read_json_line_data(self.in_file_name)
        df = self.preprocess_data(df)

        if save_preprocessed:
            self.write_json_line_data(df, self.processed_file_name)

        return df


if __name__ == '__main__':

    file_name = 'bdsEventsSample.json'

    file_reader = FileReader(file_name)
    df = file_reader.load_data(use_already_preprocessed=True, save_preprocessed=True)


    print('Finished.')