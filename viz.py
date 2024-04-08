import matplotlib.pyplot as plt
import numpy as np


def plot_lightcurve(dfs, bands=None, time_column='mjd', band_column='passband', parameter_column='flux',
                    parameter_error_column='flux_err', colors=None, min_sn=0):
    if not isinstance(dfs, list):
        dfs = [dfs]
    if not isinstance(parameter_column, list):
        parameter_column = [parameter_column] * len(dfs)
    if not isinstance(parameter_error_column, list):
        parameter_error_column = [parameter_error_column] * len(dfs)
    if not isinstance(min_sn, list):
        min_sn = [min_sn] * len(dfs)
    if bands is None:
        bands = []
        for df in dfs:
            bands += np.unique(df[band_column].values).tolist()
        bands = sorted(list(set(bands)))
    fig, axes = plt.subplots(1, len(dfs), figsize=(5 * len(dfs), 4), sharex=True, sharey=True)
    for i, df in enumerate(dfs):
        for j, band in enumerate(bands):
            if colors is not None:
                color = colors[band]
            else:
                color = None
            df_band = df[df[band_column] == band]
            df_band_filter = df_band[abs(df_band[parameter_column[i]]) / df_band[parameter_error_column[i]] > min_sn[i]]
            timestamp = df_band_filter[time_column]
            flux = df_band_filter[parameter_column[i]]
            if parameter_error_column[i] is not None:
                flux_error = df_band_filter[parameter_error_column[i]]
            else:
                flux_error = []
            if j >= 0:
                if len(dfs) == 1:
                    axes.scatter(timestamp, flux, color=color, s=8)
                else:
                    axes[i].scatter(timestamp, flux, color=color, s=8)
                if len(flux_error) > 0:
                    if len(dfs) == 1:
                        axes.errorbar(timestamp, flux, flux_error, ls='none', color=color)
                    else:
                        axes[i].errorbar(timestamp, flux, flux_error, ls='none', color=color)
        if len(dfs) == 1:
            axes.set_xlabel('Time (days)')
            axes.set_ylabel('Flux')
        else:
            axes[i].set_xlabel('Time (days)')
            axes[i].set_ylabel('Flux')
            axes[i].grid()
    plt.show()
