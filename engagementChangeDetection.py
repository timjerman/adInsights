from collections import OrderedDict
import json
import numpy as np

import dataLoader


class AdSession:

    def __init__(self):
        self.data = {
                'interacted': False,
                'requested': False,
                'sdk': '',
                'objectClazz': '',
                'timestamp': 0,
                'error': False,
                'live': False,
                'wasInteractionUpdated': False
            }

    def update(self, **kwargs):
        if kwargs is not None:
            if not self.data['interacted'] and 'interacted' in kwargs.keys():
                self.data['wasInteractionUpdated'] = True
            for key, value in kwargs.items():
                self.data[key] = value

        return self


class AdSessionWindow(OrderedDict):
    max_elements = 1
    engagement_rate = 0
    number_of_interactions = 0

    def update_engagement_rate(self):
        self.engagement_rate = self.number_of_interactions / len(self)

    def __init__(self, *args, max_elements=1, **kwargs):
        self.max_elements = max_elements
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value: AdSession):
        if value.data['wasInteractionUpdated'] and value.data['interacted']:
            value.update(wasInteractionUpdated=False)
            self.number_of_interactions += 1
            self.update_engagement_rate()

        OrderedDict.__setitem__(self, key, value)
        if self.max_elements > 0:
            if len(self) > self.max_elements:
                key_pop, value_pop = self.popitem(False)
                if value_pop.data['interacted']:
                    self.number_of_interactions -= 1
                    self.update_engagement_rate()



if __name__ == '__main__':

    # load the data that will act as a stream and sort it by timestamp
    file_name = 'bdsEventsSample.json'
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
                streamed_data_window[session_id] = streamed_data_window[session_id].update(
                    requested=True,
                    timestamp=stream_line['timestamp'],
                    live=stream_line['purpose'] == 'live'
                )
            elif stream_line['name'] == 'interaction' or stream_line['name'] == 'firstInteraction':
                streamed_data_window[session_id] = streamed_data_window[session_id].update(
                    interacted=True,
                    objectClazz=stream_line['objectClazz']
                )
            elif stream_line['name'] == 'pageRequested':
                streamed_data_window[session_id] = streamed_data_window[session_id].update(sdk=stream_line['sdk'])
            elif stream_line['name'] == 'userError':
                streamed_data_window[session_id] = streamed_data_window[session_id].update(userError=True)

            # print(streamed_data_window.engagement_rate)
            engagement_rate.append(streamed_data_window.engagement_rate)

    np.array(engagement_rate).dump('adEngagementRate.npy')


    print('Finished!')