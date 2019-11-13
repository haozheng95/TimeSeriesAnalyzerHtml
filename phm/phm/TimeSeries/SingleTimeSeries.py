import logging

import numpy as np
import statsmodels.api as sm

from phm.phm.TimeSeries.Utils import check_series_type

_logger = logging.getLogger(__name__)


class SingleTimeSeries:
    def __init__(self, time_series, freq=None):
        self.ts = time_series
        self.freq = freq if (freq is not None) else int(time_series.count() / 2)
        self.features = {}

    def _compute_features(self, if_series=True):
        if not check_series_type(self.ts):
            _logger.error("The input is not a DataFrame or Series.")
            return

        self.features['Descriptive statistics summary'] = self.ts.describe()
        self.features['Histogram'] = self.ts.value_counts()
        self.features['Skewness'] = self.ts.skew()
        self.features['Kurtosis'] = self.ts.kurt()
        if if_series:
            self.features['Rolling_mean'] = self.ts.rolling(window=self.freq).mean().dropna(how='any')
            self.features['Rolling_std'] = self.ts.rolling(window=self.freq).std().dropna(how='any')
            res = sm.tsa.seasonal_decompose(self.ts.values, freq=self.freq)
            self.features['Observed'] = res.observed
            self.features['Seasonal'] = res.seasonal
            self.features['Trend'] = res.trend
            self.features['Resid'] = res.resid

    def feature_to_string(self, if_series=True):
        self._compute_features(if_series)
        feature_string = ''
        for name, feature in self.features.items():
            if check_series_type(feature):
                feature = feature.to_json()[1:-1]
            if isinstance(feature, np.ndarray):
                feature = str(feature.tolist())[1:-1]
            feature_string += name + ' # ' + str(feature) + '\n'

        return feature_string

    def feature_to_dict(self, if_series=True):
        self._compute_features(if_series)
        feature_dict = {}
        for name, feature in self.features.items():
            if check_series_type(feature):
                feature = feature.to_json()[1:-1]
            if isinstance(feature, np.ndarray):
                feature = str(feature.tolist())[1:-1]
            feature_dict[name] = str(feature)

        return feature_dict
