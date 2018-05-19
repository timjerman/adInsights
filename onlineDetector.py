import pandas as pd

from cusumChangeDetector import CusumChangeDetector
from streamEngagement import StreamAdEngagement
import dataLoader

# load the data that will act as a stream and sort it by timestamp
file_name = 'bdsEventsSample.json'
file_reader = dataLoader.FileReader(file_name, suffix=dataLoader.FileReader.SORTED_SUFFIX)
file_reader.sort_input_data(use_already_preprocessed=True, save_preprocessed=True)
file_name = file_reader.processed_file_name

engagement_window_size = 10000
detector_window_size = 100000

engagement_rate = []
error_rate = []
data_timestamp = []
engagement_rate_sdk = {}
engagement_rate_object = {}
engagement_rate_sdk_absolute = {}
line_count = 0

stream_ad_engagement = StreamAdEngagement(window_size=engagement_window_size)
change_detector = CusumChangeDetector(3, 50, detector_window_size, 'mean')
change_detector_error = CusumChangeDetector(3, 50, detector_window_size, 'mean')
change_detector_sdk = CusumChangeDetector(3, 50, detector_window_size, 'mean')
change_detector_object = CusumChangeDetector(3, 50, detector_window_size, 'mean')
sdk_to_detect = 'MobileWeb'
object_to_detect = 'Hotspot'

with open(file_name) as stream:
    for stream_line_json in stream:

        stream_engagement_return = stream_ad_engagement.analyze_line(stream_line_json)

        if stream_engagement_return is not None:
            data_timestamp.append(stream_engagement_return[0])
            engagement_rate.append(stream_engagement_return[1])
            error_rate.append(stream_engagement_return[2])
            engagement_rate_sdk[line_count] = stream_engagement_return[3]
            engagement_rate_sdk_absolute[line_count] = stream_engagement_return[4]
            engagement_rate_object[line_count] = stream_engagement_return[5]

            # start detection only after the engagement mean stabilizes
            if line_count > 0.5 * engagement_window_size:

                alarm, start_timestamp, relative_change = change_detector.add_data_point(stream_engagement_return[1], stream_engagement_return[0])
                if alarm is not None:
                    print(
                        'Ad Engagement: a change of {:.2f}% to {} detected at timestamp: {} with origin at timestamp: {}'.format(
                            relative_change, stream_engagement_return[1], pd.to_datetime(alarm, unit='s'), pd.to_datetime(start_timestamp, unit='s')
                        )
                    )

                alarm, start_timestamp, relative_change = change_detector_error.add_data_point(stream_engagement_return[2], stream_engagement_return[0])
                if alarm is not None:
                    print(
                        'Error rate: a change of {:.2f}% detected to {} at timestamp: {} with origin at timestamp: {}'.format(
                            relative_change, stream_engagement_return[2], pd.to_datetime(alarm, unit='s'), pd.to_datetime(start_timestamp, unit='s')
                        )
                    )

                if sdk_to_detect in stream_engagement_return[4]:
                    alarm, start_timestamp, relative_change = change_detector_sdk.add_data_point(stream_engagement_return[4][sdk_to_detect], stream_engagement_return[0])
                    if alarm is not None:
                        print(
                            'SDK (MobileWeb) engagement: a change of {:.2f}% to {} detected at timestamp: {} with origin at timestamp: {}'.format(
                                relative_change, stream_engagement_return[4][sdk_to_detect], pd.to_datetime(alarm, unit='s'), pd.to_datetime(start_timestamp, unit='s')
                            )
                        )

                if object_to_detect in stream_engagement_return[5]:
                    alarm, start_timestamp, relative_change = change_detector_sdk.add_data_point(stream_engagement_return[5][object_to_detect], stream_engagement_return[0])
                    if alarm is not None:
                        print(
                            'SDK (MobileWeb) engagement: a change of {:.2f}% to {} detected at timestamp: {} with origin at timestamp: {}'.format(
                                relative_change, stream_engagement_return[5][object_to_detect], pd.to_datetime(alarm, unit='s'), pd.to_datetime(start_timestamp, unit='s')
                            )
                        )

            line_count += 1

print('Finished')
