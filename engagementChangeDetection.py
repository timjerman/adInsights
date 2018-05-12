from collections import OrderedDict
import json
import numpy as np

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

    def update(self, skip_if_exists=True, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                if not bool(self.data[key]) or not skip_if_exists:
                    self.data[key] = value

        return self

    def copy(self):
        new_copy = AdSession().update(**self.data)
        return new_copy



class AdSessionWindow(OrderedDict):
    max_elements = 1
    engagement_rate = 0
    number_of_interactions = 0
    error_count = 0
    number_of_interactions_by_object = {}
    number_of_interactions_by_sdk = {}

    def __init__(self, max_elements=1, *args, **kwargs):
        self.max_elements = max_elements
        super().__init__(*args, **kwargs)

    def update_engagement_rate(self):
        self.engagement_rate = self.number_of_interactions / len(self)

    def update_interaction_value(self, key, interaction_dict, add=True):
        if key not in interaction_dict:
            interaction_dict[key] = 0
        if add:
            interaction_dict[key] += 1
        else:
            interaction_dict[key] -= 1

    def update_session_elements(self, session_id, **kwargs):

        session_before_update = self[session_id].copy()
        self[session_id] = self[session_id].update(skip_if_exists=True, **kwargs)
        update_needed = False

        if self[session_id].data[AdSession.INTERACTED]:
            if session_before_update.data[AdSession.INTERACTED] != self[session_id].data[AdSession.INTERACTED]:
                self.number_of_interactions += 1

                if self[session_id].data[AdSession.SDK] != '':
                    self.update_interaction_value(self[session_id].data[AdSession.SDK], self.number_of_interactions_by_sdk)
                if self[session_id].data[AdSession.OBJECT] != '':
                    self.update_interaction_value(self[session_id].data[AdSession.OBJECT], self.number_of_interactions_by_object)

                update_needed = True
            else:
                if session_before_update.data[AdSession.SDK] == '' and self[session_id].data[AdSession.SDK] != '':
                    self.update_interaction_value(self[session_id].data[AdSession.SDK], self.number_of_interactions_by_sdk)
                    update_needed = True

                if session_before_update.data[AdSession.OBJECT] == '' and self[session_id].data[AdSession.OBJECT] != '':
                    self.update_interaction_value(self[session_id].data[AdSession.OBJECT], self.number_of_interactions_by_object)
                    update_needed = True

        if not session_before_update.data[AdSession.ERROR] and self[session_id].data[AdSession.ERROR]:
            self.error_count += 1

        if update_needed:
            self.update_engagement_rate()


        # self[session_id] = self[session_id].update(kwargs)



    def __setitem__(self, key, value: AdSession):

        OrderedDict.__setitem__(self, key, value)
        if self.max_elements > 0:
            if len(self) > self.max_elements:
                key_pop, value_pop = self.popitem(False)
                if value_pop.data[AdSession.INTERACTED]:
                    self.number_of_interactions -= 1
                    if value_pop.data[AdSession.SDK] != '':
                        self.update_interaction_value(value_pop.data[AdSession.SDK],
                                                      self.number_of_interactions_by_sdk, add=False)
                    if value_pop.data[AdSession.OBJECT] != '':
                        self.update_interaction_value(value_pop.data[AdSession.OBJECT],
                                                      self.number_of_interactions_by_object, add=False)
                if value_pop.data[AdSession.ERROR]:
                    self.error_count -= 1

                self.update_engagement_rate()



if __name__ == '__main__':

    # load the data that will act as a stream and sort it by timestamp
    file_name = 'bdsEventsSample.json'
    # file_name = 'test.json'
    file_reader = dataLoader.FileReader(file_name, suffix=dataLoader.FileReader.SORTED_SUFFIX)
    file_reader.sort_input_data(use_already_preprocessed=True, save_preprocessed=True)
    file_name = file_reader.processed_file_name

    engagement_rate = []
    streamed_data_window = AdSessionWindow(max_elements=10000)

    with open(file_name) as stream:
        for stream_line in stream:

            stream_line = json.loads(stream_line)
            session_id = stream_line['sessionId']

            if stream_line['name'] not in ['adRequested', 'interaction', 'firstInteraction', 'pageRequested', 'userError']:
                continue

            if session_id not in streamed_data_window:
                streamed_data_window[session_id] = AdSession()

            if stream_line['name'] == 'adRequested':
                streamed_data_window.update_session_elements(
                    session_id,
                    requested=True,
                    timestamp=stream_line['timestamp'],
                    live=stream_line['purpose'] == 'live'
                )
            elif stream_line['name'] == 'interaction' or stream_line['name'] == 'firstInteraction':
                streamed_data_window.update_session_elements(
                    session_id,
                    interacted=True,
                    objectClazz=stream_line['objectClazz']
                )
            elif stream_line['name'] == 'pageRequested':
                streamed_data_window.update_session_elements(
                    session_id,
                    sdk=stream_line['sdk']
                )
            elif stream_line['name'] == 'userError':
                streamed_data_window.update_session_elements(
                    session_id,
                    userError=True
                )

            # print(streamed_data_window.engagement_rate)
            engagement_rate.append(streamed_data_window.engagement_rate)

    np.array(engagement_rate).dump('adEngagementRate.npy')

    # # plots
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # plt.plot(engagement_rate)
    # plt.show()


    print('Finished!')