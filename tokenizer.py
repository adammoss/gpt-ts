import pandas as pd
import numpy as np


class LCTokenizer:

    def __init__(self, min_flux, max_flux, num_bins, max_delta_time,
                 num_time_bins, bands=None, pad_token=True,
                 min_delta_time=0,
                 band_column='passband', time_column='mjd',
                 parameter_column='flux',
                 parameter_error_column='flux_err',
                 transform=None, inverse_transform=None, min_SN=0):
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform
        if inverse_transform is None:
            self.inverse_transform = lambda x: x
        else:
            self.inverse_transform = inverse_transform
        self.min_flux = self.transform(min_flux)
        self.max_flux = self.transform(max_flux)
        assert num_bins > 2
        self.num_bins = num_bins
        # Reserve 2 tokens for OOD lower and higher
        self.dflux = (self.max_flux - self.min_flux) / (num_bins - 2)
        self.min_delta_time = self.transform(min_delta_time)
        self.max_delta_time = self.transform(max_delta_time)
        self.num_time_bins = num_time_bins
        self.dt = (self.max_delta_time - self.min_delta_time) / num_time_bins
        self.bands = bands
        if bands is not None:
            self.num_bands = len(bands)
            self.vocab_size = num_time_bins + len(bands) * num_bins
        else:
            self.vocab_size = num_time_bins + num_bins
        if pad_token:
            self.vocab_size += 1
        self.pad_token = pad_token
        self.band_column = band_column
        self.time_column = time_column
        self.parameter_column = parameter_column
        self.parameter_error_column = parameter_error_column
        self.min_SN = min_SN

    def flux_token(self, flux):
        flux = self.transform(flux)
        if flux < self.min_flux:
            return 0
        elif flux >= self.max_flux:
            return self.num_bins - 1
        else:
            return int((flux - self.min_flux) // self.dflux) + 1

    def time_token(self, delta_time):
        delta_time = self.transform(delta_time)
        if delta_time < 0:
            return 0
        elif delta_time >= self.max_delta_time:
            return self.num_time_bins - 1
        else:
            return int((delta_time - self.min_delta_time) // self.dt)

    def encode(self, df, augment=False):
        tokens_dict = {}
        last_object_id = None
        # Zip is much faster than .iterrows
        # https://stackoverflow.com/questions/7837722/what-is-the-most-efficient-way-to-loop-through-dataframes-with-pandas
        for row in zip(df['object_id'], df[self.time_column], df[self.band_column], df[self.parameter_column],
                       df[self.parameter_error_column]):
            if row[0] != last_object_id:
                if last_object_id is not None:
                    tokens_dict[last_object_id] = tokens
                tokens = []
                last_time = row[1]
                last_object_id = row[0]
            if self.min_SN is not None and abs(row[3]) / row[4] < self.min_SN:
                continue
            time_token = self.time_token(row[1] - last_time)
            last_time = row[1]
            if time_token > 0:
                if self.pad_token:
                    time_token += 1
                tokens.append(time_token)
            if augment:
                flux_token = self.flux_token(row[3] + row[4] * np.random.normal())
            else:
                flux_token = self.flux_token(row[3])
            if self.bands is not None:
                band_index = self.bands.index(row[2])
                flux_token += band_index * self.num_bins
                if self.pad_token:
                    flux_token += 1
            tokens.append(self.num_time_bins + flux_token)
        tokens_dict[last_object_id] = tokens
        if len(tokens_dict) == 1 and last_object_id is not None:
            return tokens_dict[last_object_id]
        else:
            return tokens_dict

    def decode(self, tokens):
        timestamp = 0
        data = []
        for token in tokens:
            if self.pad_token:
                token = token - 1
            if 0 <= token < self.num_time_bins:
                timestamp += self.inverse_transform(self.min_delta_time + token * self.dt)
            elif token >= self.num_time_bins:
                token = token - self.num_time_bins
                band = token // self.num_bins
                flux = self.inverse_transform(self.min_flux + self.dflux * ((token - 1) % self.num_bins))
                data.append([timestamp, self.bands[band], flux, 0])
        df = pd.DataFrame(data, columns=[self.time_column, self.band_column,
                                         self.parameter_column,
                                         self.parameter_error_column])
        return df
