from collections import OrderedDict
import json
import numpy as np
import pandas as pd

import dataLoader


class AdSession:

    INTERACTED = 'interacted'
    REQUESTED = 'requested'
    SDK = 'sdk'
    OBJECT = 'objectClazz'
    TIMESTAMP = 'timestamp'
    ERROR = 'userError'
    LIVE = 'live'

    def __init__(self):
        self.data = {
            AdSession.INTERACTED: False,
            AdSession.REQUESTED: False,
            AdSession.SDK: '',
            AdSession.OBJECT: '',
            AdSession.TIMESTAMP: 0,
            AdSession.ERROR: False,
            AdSession.LIVE: False
        }

    def update(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                if not bool(self.data[key]) and value is not None:
                    self.data[key] = value

        return self

    def copy(self):
        new_copy = AdSession().update(**self.data)
        return new_copy


class AdSessionWindow(OrderedDict):
    window_size = 1
    engagement_rate = 0
    error_rate = 0
    engagement_by_sdk = {}
    engagement_by_sdk_relative = {}
    engagement_by_object = {}
    number_of_interactions = 0
    error_count = 0
    number_of_interactions_by_object = {}
    number_of_interactions_by_sdk = {}
    count_by_sdk = {}
    requested_and_live_count = 0

    def __init__(self, window_size=1, *args, **kwargs):
        self.window_size = window_size
        super().__init__(*args, **kwargs)

    @staticmethod
    def divide(x, y):
        if y == 0:
            return 0
        return x/y

    def update_engagement_rate(self):

        self.error_rate = 100 * self.error_count / len(self)
        if self.requested_and_live_count > 0:
            self.engagement_rate = 100 * self.number_of_interactions / self.requested_and_live_count
            self.engagement_by_sdk_relative = {k: 100 * v / self.requested_and_live_count for k, v in self.number_of_interactions_by_sdk.items()}
            self.engagement_by_object = {k: 100 * v / self.requested_and_live_count for k, v in self.number_of_interactions_by_object.items()}

        self.engagement_by_sdk = {k: 100 * AdSessionWindow.divide(v, self.count_by_sdk[k]) for k, v in self.number_of_interactions_by_sdk.items()}

    @staticmethod
    def update_interaction_value(key, interaction_dict, add=True):
        if key not in interaction_dict:
            interaction_dict[key] = 0
        if add:
            interaction_dict[key] += 1
        else:
            interaction_dict[key] -= 1
        return interaction_dict

    def update_session_elements(self, session_id, **kwargs):

        session_before_update = self[session_id].copy()
        self[session_id] = self[session_id].update(**kwargs)
        update_needed = False

        if self[session_id].data[AdSession.REQUESTED] and not session_before_update.data[AdSession.REQUESTED] and self[session_id].data[AdSession.LIVE]:
            self.requested_and_live_count += 1

        if self[session_id].data[AdSession.INTERACTED]:
            if session_before_update.data[AdSession.INTERACTED] != self[session_id].data[AdSession.INTERACTED]:
                self.number_of_interactions += 1

                if self[session_id].data[AdSession.SDK] != '':
                    self.number_of_interactions_by_sdk = AdSessionWindow.update_interaction_value(
                        self[session_id].data[AdSession.SDK],
                        self.number_of_interactions_by_sdk
                    )
                if self[session_id].data[AdSession.OBJECT] != '':
                    self.number_of_interactions_by_object = AdSessionWindow.update_interaction_value(
                        self[session_id].data[AdSession.OBJECT],
                        self.number_of_interactions_by_object
                    )

                update_needed = True
            else:
                if session_before_update.data[AdSession.SDK] == '' and self[session_id].data[AdSession.SDK] != '':
                    self.number_of_interactions_by_sdk = AdSessionWindow.update_interaction_value(
                        self[session_id].data[AdSession.SDK],
                        self.number_of_interactions_by_sdk
                    )
                    update_needed = True

                if session_before_update.data[AdSession.OBJECT] == '' and self[session_id].data[AdSession.OBJECT] != '':
                    self.number_of_interactions_by_object = AdSessionWindow.update_interaction_value(
                        self[session_id].data[AdSession.OBJECT],
                        self.number_of_interactions_by_object
                    )
                    update_needed = True

        if session_before_update.data[AdSession.SDK] == '' and self[session_id].data[AdSession.SDK] != '':
            self.count_by_sdk = AdSessionWindow.update_interaction_value(
                self[session_id].data[AdSession.SDK],
                self.count_by_sdk
            )
            update_needed = True

        if not session_before_update.data[AdSession.ERROR] and self[session_id].data[AdSession.ERROR]:
            self.error_count += 1

        if update_needed:
            self.update_engagement_rate()

    def __setitem__(self, key, value: AdSession):

        OrderedDict.__setitem__(self, key, value)
        if self.window_size > 0:
            if len(self) > self.window_size:
                key_pop, value_pop = self.popitem(False)

                if value_pop.data[AdSession.REQUESTED] and value_pop.data[AdSession.LIVE]:
                    self.requested_and_live_count -= 1
                if value_pop.data[AdSession.SDK] != '':
                    self.count_by_sdk = AdSessionWindow.update_interaction_value(
                        value_pop.data[AdSession.SDK],
                        self.count_by_sdk,
                        add=False
                    )
                if value_pop.data[AdSession.INTERACTED]:
                    self.number_of_interactions -= 1
                    if value_pop.data[AdSession.SDK] != '':
                        self.number_of_interactions_by_sdk = AdSessionWindow.update_interaction_value(
                            value_pop.data[AdSession.SDK],
                            self.number_of_interactions_by_sdk,
                            add=False
                        )
                    if value_pop.data[AdSession.OBJECT] != '':
                        self.number_of_interactions_by_object = AdSessionWindow.update_interaction_value(
                            value_pop.data[AdSession.OBJECT],
                            self.number_of_interactions_by_object,
                            add=False
                        )
                if value_pop.data[AdSession.ERROR]:
                    self.error_count -= 1

                self.update_engagement_rate()


class StreamAdEngagement:

    def __init__(self, window_size):
        self.streamed_data_window = AdSessionWindow(window_size=window_size)

    def analyze_line(self, stream_line_json):

        stream_line = json.loads(stream_line_json)
        session_id = stream_line['sessionId']

        if stream_line['name'] not in ['adRequested', 'interaction', 'firstInteraction', 'pageRequested', 'userError']:
            return None

        if session_id not in self.streamed_data_window:
            self.streamed_data_window[session_id] = AdSession()

        if stream_line['name'] == 'adRequested':
            self.streamed_data_window.update_session_elements(
                session_id,
                requested=True,
                timestamp=stream_line['timestamp'],
                live=stream_line['purpose'] == 'live'
            )
        elif stream_line['name'] == 'interaction' or stream_line['name'] == 'firstInteraction':
            self.streamed_data_window.update_session_elements(
                session_id,
                interacted=True,
                objectClazz=stream_line['objectClazz']
            )
        elif stream_line['name'] == 'pageRequested':
            self.streamed_data_window.update_session_elements(
                session_id,
                sdk=stream_line['sdk']
            )
        elif stream_line['name'] == 'userError':
            self.streamed_data_window.update_session_elements(
                session_id,
                userError=True
            )

        return (
            self.streamed_data_window.engagement_rate,
            self.streamed_data_window.error_rate,
            self.streamed_data_window.engagement_by_sdk,
            self.streamed_data_window.engagement_by_sdk_relative,
            self.streamed_data_window.engagement_by_object
        )


if __name__ == '__main__':

    # load the data that will act as a stream and sort it by timestamp
    file_name = 'bdsEventsSample.json'
    # file_name = 'test.json'
    file_reader = dataLoader.FileReader(file_name, suffix=dataLoader.FileReader.SORTED_SUFFIX)
    file_reader.sort_input_data(use_already_preprocessed=True, save_preprocessed=True)
    file_name = file_reader.processed_file_name

    engagement_rate = []
    error_rate = []
    engagement_rate_sdk = {}
    engagement_rate_object = {}
    engagement_rate_sdk_relative = {}
    line_count = 0

    stream_ad_engagement = StreamAdEngagement(window_size=10000)

    with open(file_name) as stream:
        for stream_line_json in stream:

            stream_engagement_return = stream_ad_engagement.analyze_line(stream_line_json)

            if stream_engagement_return is not None:
                engagement_rate.append(stream_engagement_return[0])
                error_rate.append(stream_engagement_return[1])
                engagement_rate_sdk[line_count] = stream_engagement_return[2]
                engagement_rate_sdk_relative[line_count] = stream_engagement_return[3]
                engagement_rate_object[line_count] = stream_engagement_return[4]

            line_count += 1

    np.array(engagement_rate).dump('adEngagementRate.npy')
    np.array(error_rate).dump('errorRate.npy')
    pd.DataFrame.from_dict(engagement_rate_object).transpose().to_pickle('objectEngagementRate.npy')
    pd.DataFrame.from_dict(engagement_rate_sdk).transpose().to_pickle('sdkEngagementRate.npy')
    pd.DataFrame.from_dict(engagement_rate_sdk_relative).transpose().to_pickle('sdkRelativeEngagementRate.npy')

    # # plots
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # plt.plot(engagement_rate)
    # plt.show()


    print('Finished!')