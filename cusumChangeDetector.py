import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import pandas as pd

class CusumChangeDetector:
    """
    Cumulative sum algorithm (CUSUM) for online change detection
    """

    def __init__(self, threshold_factor=1, min_threshold_steps=0, window_size=1, mode='mean'):
        self.threshold_factor = threshold_factor
        self.min_threshold_steps = min_threshold_steps
        self.window_size = window_size
        self.running_window = deque([], window_size)
        self.control_upper = 0
        self.control_lower = 0
        self.alarm_index = 0
        self.start_index = 0
        self.control_upper_start = 0
        self.control_lower_start = 0
        self.threshold_surpassed_count = 0
        self.mode = mode
        self.previous = [0, 0, 0]  # data point, mean, std of the last computation
        self.stream_mean = None
        self.variance_stream = 0
        self.std = 0
        self.running_window_complete = False
        self.mean_at_start = [0, 0]

    def compute_stream_mean_and_variance(self, added_value, removed_value):
        if self.stream_mean is None:
            self.stream_mean = added_value
            self.variance_stream = 0
        else:
            stream_mean_0 = self.stream_mean
            if self.running_window_complete:
                self.stream_mean = (self.window_size * stream_mean_0 + added_value - removed_value) / self.window_size
                self.variance_stream = self.variance_stream + (added_value + removed_value - self.stream_mean - stream_mean_0) * (added_value - removed_value)
                self.std = np.sqrt(self.variance_stream / (self.window_size - 1))
            else:
                self.stream_mean = (added_value + (len(self.running_window) - 1) * stream_mean_0) / ((len(self.running_window) - 1) + 1)
                self.variance_stream = self.variance_stream + (added_value - stream_mean_0) * (added_value - self.stream_mean)
                self.std = np.sqrt(self.variance_stream / len(self.running_window))

        return self.stream_mean, self.std

    def add_data_point(self, x, index):

        alarm_index = None
        start_index = None
        relative_change = None

        removed_value = self.running_window[0] if len(self.running_window) > 0 else 0
        self.running_window.append(x)
        mean_val, std_val = self.compute_stream_mean_and_variance(x, removed_value)  #np.mean(self.running_window)

        # start detection only after a good portion of the window is filled so as to get some good statistic
        if len(self.running_window) > 0.5 * self.window_size:
            threshold = self.threshold_factor * std_val

            if self.mode == 'mean':
                diff = mean_val - self.previous[1]
            else:
                diff = x - self.previous[0]

            self.control_upper = self.control_upper + diff
            self.control_lower = self.control_lower - diff

            if self.control_upper < 0:  # or x[i] < mean_val:
                self.control_upper = 0
                self.control_upper_start = index
                self.mean_at_start[0] = mean_val
            if self.control_lower < 0:  # or x[i] > mean_val:
                self.control_lower = 0
                self.control_lower_start = index
                self.mean_at_start[1] = mean_val

            if self.control_upper > threshold or self.control_lower > threshold:
                # change detected
                self.threshold_surpassed_count += 1
                # for consistency the change should last at least a few steps
                if self.threshold_surpassed_count > self.min_threshold_steps:
                    alarm_index = index  # alarm index
                    start_index = self.control_upper_start if self.control_upper > threshold else self.control_lower_start
                    relative_change = self.mean_at_start[0] if self.control_upper > threshold else self.mean_at_start[1]
                    relative_change = 100 * (mean_val - relative_change) / (relative_change + 1e-6)
                    self.control_upper, self.control_lower = 0, 0  # reset alarm
                    self.control_upper_start, self.control_lower_start = index, index
                    self.mean_at_start = [mean_val, mean_val]
            else:
                self.threshold_surpassed_count = 0
        else:
            self.control_upper_start = index
            self.control_lower_start = index
            self.mean_at_start = [mean_val, mean_val]

        self.previous = [x, mean_val, std_val]
        if len(self.running_window) == self.window_size:
            self.running_window_complete = True

        return alarm_index, start_index, relative_change


def detect_cusum_offline(data, datat=None, threshold_factor=1, min_threshold_steps=0, window_size=1, mode='mean',
                         show=True, ylabel=None, save_name=None):

    change_detector = CusumChangeDetector(threshold_factor,  min_threshold_steps, window_size, mode)
    alarm_index, start_index = [], []
    data_mean, data_std = [], []

    for i in range(data.size):
        alarm_index_current, start_index_current, relative_change = change_detector.add_data_point(
            data[i],
            i if datat is None else datat[i]
        )

        if alarm_index_current is not None:
            alarm_index.append(alarm_index_current)
            start_index.append(start_index_current)

        data_mean.append(change_detector.previous[1])
        data_std.append(change_detector.previous[2])

    if show:
        plot_detections(data, datat, alarm_index, start_index, np.array(data_mean), 3 * np.array(data_std), ylabel)
        if save_name is not None:
            plt.savefig(save_name, bbox_inches='tight', dpi=300)
        plt.show()


def plot_detections(data, datat, alarm_index, start_index, xmean, xstd, ylabel=None):
    """Plot results of the detect_cusum function"""

    t = datat
    if t is None:
        t = range(data.size)
        alarm_index_t = alarm_index
        start_index_t = start_index
    elif isinstance(t, pd.DatetimeIndex):
        dt_df = pd.DataFrame(index=t, data={'data': data, 'xmean': xmean, 'xstd': xstd})
        dt_df = dt_df.loc[~dt_df.index.duplicated(keep='first')]
        alarm_index_t = [dt_df.index.get_loc(aidx, method='nearest') for aidx in alarm_index]
        start_index_t = [dt_df.index.get_loc(sidx, method='nearest') for sidx in start_index]
        t = dt_df.index
        data = dt_df['data'].values
        xmean = dt_df['xmean'].values
        xstd = dt_df['xstd'].values
    else:
        alarm_index_t = [np.nonzero(np.abs(t - aidx) < 1e-6)[0][0] for aidx in alarm_index]
        start_index_t = [np.nonzero(np.abs(t - sidx) < 1e-6)[0][0] for sidx in start_index]

    _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(t, data, 'b-', lw=2)
    ax.plot(t, xmean, 'r-', lw=2, alpha=0.7)
    ax.fill_between(t, xmean-xstd, xmean+xstd, color='green', alpha=0.3, interpolate=True)
    if len(alarm_index):
        ax.plot(start_index, xmean[start_index_t], '>', mfc='g', mec='g', ms=10, label='Start')
        ax.plot(alarm_index, xmean[alarm_index_t], 'o', mfc='r', mec='y', mew=1, ms=10, label='Alarm')
        ax.legend(loc='best', framealpha=.5, numpoints=1)
    #ax.set_xlim(-.01 * data.size, data.size * 1.01 - 1)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel('Data points', fontsize=14)
    if isinstance(t, pd.DatetimeIndex):
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_xlabel('Time of day', fontsize=14)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=14)
    else:
        ax.set_ylabel('Engagement rate', fontsize=14)
    ymin, ymax = data[np.isfinite(data)].min(), data[np.isfinite(data)].max()
    yrange = ymax - ymin if ymax > ymin else 1
    ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
    ax.set_title('Time series with detected changes [{:d}]'.format(len(start_index)))
    plt.tight_layout()


if __name__ == '__main__':

    import pandas as pd
    import numpy as np

    from cusumChangeDetector import detect_cusum_offline

    data = pd.read_pickle('adEngagementRate.npy')
    # plot x-axis as %H:%M
    data_time = pd.read_pickle('dataTimestamps.npy')
    data_time_dt = pd.to_datetime(data_time, unit='s')
    detect_cusum_offline(data, data_time_dt, 3, 50, 100000, 'mean', True, 'Ad engagement rate', save_name='changeDetectorAdEngagment.png' )
    # # plot x-axis as datapoints
    # detect_cusum_offline(data, None, 3, 50, 100000, 'mean', True)
    # # plot x-axis as unix timestamps
    # data_time = pd.read_pickle('dataTimestamps.npy')
    # detect_cusum_offline(data, data_time, 3, 50, 100000, 'mean', True)

    # Other examples:
    # error rate
    data = pd.read_pickle('errorRate.npy')
    data_time = pd.read_pickle('dataTimestamps.npy')
    data_time_dt = pd.to_datetime(data_time, unit='s')
    detect_cusum_offline(data, data_time_dt, 3, 50, 100000, 'mean', True, 'Error rate', save_name='changeDetectoErrorRate.png' )

    # sdk [MobileWeb] engagement rate
    data = pd.read_pickle('sdkEngagementRate.npy')
    data = data.fillna(0).round(decimals=5)
    data = np.array(data['MobileWeb'])
    data_time = pd.read_pickle('dataTimestamps.npy')
    data_time_dt = pd.to_datetime(data_time, unit='s')
    detect_cusum_offline(data, data_time_dt, 3, 50, 100000, 'mean', True, 'Ad engagement rate [MobileWeb]', save_name='changeDetectorMobileWeb.png' )

    # # sdk [MobileWeb] absolute engagement rate
    data = pd.read_pickle('sdkAbsoluteEngagementRate.npy')
    data = data.fillna(0).round(decimals=5)
    data = np.array(data['MobileWeb'])
    data_time = pd.read_pickle('dataTimestamps.npy')
    data_time_dt = pd.to_datetime(data_time, unit='s')
    detect_cusum_offline(data, data_time_dt, 3, 50, 100000, 'mean', True, 'Ad engagement rate [MobileWeb]', save_name='changeDetectorMobileWebAbsolute.png' )

    print('Finished')
