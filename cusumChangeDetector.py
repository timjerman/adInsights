import numpy as np
from collections import deque
import matplotlib.pyplot as plt


class CusumChangeDetector:
    """
    Cumulative sum algorithm (CUSUM) for online change detection
    """

    def __init__(self, data_0, threshold_factor=1, min_threshold_steps=0, window_size=1, mode='mean'):
        self.threshold_factor = threshold_factor
        self.min_threshold_steps = min_threshold_steps
        self.window_size = window_size
        self.running_window = deque([data_0], window_size)
        self.control_upper = 0
        self.control_lower = 0
        self.alarm_index = 0
        self.start_index = 0
        self.control_upper_start = 0
        self.control_lower_start = 0
        self.threshold_surpassed_count = 0
        self.mode = mode
        self.previous = [data_0, data_0, 0] # data point, mean, std of the last computation
        self.stream_mean = data_0
        self.variance_stream = 0
        self.std = 0
        self.running_window_complete = False

    def compute_stream_mean_and_variance(self, added_value, removed_value):
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

        removed_value = self.running_window[0]
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
            if self.control_lower < 0:  # or x[i] > mean_val:
                self.control_lower = 0
                self.control_lower_start = index

            if self.control_upper > threshold or self.control_lower > threshold:
                # change detected
                self.threshold_surpassed_count += 1
                # for consistency the change should last at least a few steps
                if self.threshold_surpassed_count > self.min_threshold_steps:
                    alarm_index = index  # alarm index
                    start_index = self.control_upper_start if self.control_upper > threshold else self.control_lower_start
                    self.control_upper, self.control_lower = 0, 0  # reset alarm
                    self.control_upper_start, self.control_lower_start = index, index
            else:
                self.threshold_surpassed_count = 0

        self.previous = [x, mean_val, std_val]
        if len(self.running_window) == self.window_size:
            self.running_window_complete = True

        return alarm_index, start_index


def detect_cusum_offline(data, threshold_factor=1, min_threshold_steps=0, window_size=1, mode='mean', show=True):

    change_detector = CusumChangeDetector(data[0], threshold_factor,  min_threshold_steps, window_size, mode)
    alarm_index, start_index = [], []
    data_mean, data_std = [data[0]], [0]

    for i in range(1, data.size):
        alarm_index_current, start_index_current = change_detector.add_data_point(data[i], i)

        if alarm_index_current is not None:
            alarm_index.append(alarm_index_current)
            start_index.append(start_index_current)

        data_mean.append(change_detector.previous[1])
        data_std.append(change_detector.previous[2])

    if show:
        plot_detections(data, threshold_factor, min_threshold_steps, window_size, alarm_index, start_index, np.array(data_mean), 3 * np.array(data_std))


def plot_detections(data, threshold, threshold_steps, window_size, alarm_index, start_index, xmean, xstd):
    """Plot results of the detect_cusum function"""

    _, ax = plt.subplots(figsize=(8, 5))

    t = range(data.size)
    ax.plot(t, data, 'b-', lw=2)
    ax.plot(t, xmean, 'r-', lw=2, alpha=0.7)
    ax.fill_between(t, xmean-xstd, xmean+xstd, color='green', alpha=0.3, interpolate=True)
    if len(alarm_index):
        ax.plot(start_index, xmean[start_index], '>', mfc='g', mec='g', ms=10, label='Start')
        ax.plot(alarm_index, xmean[alarm_index], 'o', mfc='r', mec='y', mew=1, ms=10, label='Alarm')
        ax.legend(loc='best', framealpha=.5, numpoints=1)
    ax.set_xlim(-.01 * data.size, data.size * 1.01 - 1)
    ax.set_xlabel('Data points', fontsize=14)
    ax.set_ylabel('Ad engagement rate', fontsize=14)
    ymin, ymax = data[np.isfinite(data)].min(), data[np.isfinite(data)].max()
    yrange = ymax - ymin if ymax > ymin else 1
    ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
    ax.set_title('Time series and detected changes ' +
                  '(threshold= %.3g * sigma, min len= %d, win size= %d): N changes = %d'
                  % (threshold, threshold_steps, window_size, len(start_index)))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    import pandas as pd
    import numpy as np

    from cusumChangeDetector import detect_cusum_offline

    df = pd.read_pickle('objectEngagementRate.npy')
    df = df.fillna(0).round(decimals=5)
    # data = np.hstack((np.array(df['Hotspot'][11000::100]),np.array(df['Hotspot'][11000::100])[::-1],np.array(df['Hotspot'][11000::100])))
    data = np.array(df['Hotspot'][::10])
    detect_cusum_offline(data, 3, 50, 10000, 'mean', True)

    print('Finished')
