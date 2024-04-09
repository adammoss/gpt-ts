import george
import scipy.optimize as op
from astropy.stats import biweight_location
import numpy as np
from functools import partial


def fit_2d_gp(df, pb_wavelengths, subtract_background=True, time_column='mjd', band_column="passband",
              parameter_column="flux", parameter_error_column="flux_err", guess_length_scale=20.0,
              sample_interval=1, replace=True):
    """
    :param df: pandas DataFrame containing the object data
    :param pb_wavelengths: dictionary of passband wavelengths for each passband value
    :param subtract_background: boolean indicating whether to subtract the estimated background flux from the object flux
    :param time_column: name of the column containing the time values
    :param band_column: name of the column containing the passband values
    :param parameter_column: name of the column containing the object flux values
    :param parameter_error_column: name of the column containing the object flux error values
    :param guess_length_scale: initial length scale guess for the Matern32Kernel kernel
    :param sample_interval: interval at which to sample the flux values for each passband
    :param replace: boolean indicating whether to replace the original flux values with the resampled flux values
    :return: a tuple containing the modified DataFrame and a tuple of sampled times, flux values, flux errors, and mask information
    """
    obj_data = df.copy()
    bands = np.unique(obj_data[band_column])

    ref_flux_bands = {}

    if subtract_background:

        for band in bands:
            mask = obj_data[band_column] == band
            band_data = obj_data[mask]

            # Use a biweight location to estimate the background
            ref_flux = biweight_location(band_data[parameter_column])

            ref_flux_bands[band] = -ref_flux

            obj_data.loc[mask, parameter_column] -= ref_flux

    obj_times = obj_data[time_column].astype(float)
    obj_flux = obj_data[parameter_column].astype(float)
    obj_flux_error = obj_data[parameter_error_column].astype(float)
    obj_wavelengths = obj_data[band_column].map(pb_wavelengths)
    obj_ref_flux = obj_data[band_column].map(ref_flux_bands)

    def neg_log_like(p):  # Objective function: negative log-likelihood
        gp.set_parameter_vector(p)
        loglike = gp.log_likelihood(obj_flux, quiet=True)
        return -loglike if np.isfinite(loglike) else 1e25

    def grad_neg_log_like(p):  # Gradient of the objective function.
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(obj_flux, quiet=True)

    # Use the highest signal-to-noise observation to estimate the scale. We
    # include an error floor so that in the case of very high
    # signal-to-noise observations we pick the maximum flux value.
    signal_to_noises = np.abs(obj_flux) / np.sqrt(
        obj_flux_error ** 2 + (1e-2 * np.max(obj_flux)) ** 2
    )
    scale = np.abs(obj_flux[signal_to_noises.idxmax()])

    kernel = (0.5 * scale) ** 2 * george.kernels.Matern32Kernel(
        [guess_length_scale ** 2, 6000 ** 2], ndim=2
    )
    kernel.freeze_parameter("k2:metric:log_M_1_1")

    gp = george.GP(kernel)
    default_gp_param = gp.get_parameter_vector()
    x_data = np.vstack([obj_times, obj_wavelengths]).T
    gp.compute(x_data, obj_flux_error)

    bounds = [(0, np.log(1000 ** 2))]
    bounds = [(default_gp_param[0] - 10, default_gp_param[0] + 10)] + bounds
    results = op.minimize(
        neg_log_like,
        gp.get_parameter_vector(),
        jac=grad_neg_log_like,
        method="L-BFGS-B",
        bounds=bounds,
        tol=1e-6,
    )
    if results.success:
        gp.set_parameter_vector(results.x)
    else:
        # Fit failed. Print out a warning, and use the initial guesses for fit
        # parameters.
        obj = obj_data["object_id"][0]
        print("GP fit failed for {}! Using guessed GP parameters.".format(obj))
        gp.set_parameter_vector(default_gp_param)

    gp = partial(gp.predict, obj_flux)

    pred_x_data = np.vstack([obj_times, obj_wavelengths]).T
    resampled_flux, var = gp(pred_x_data, return_var=True)

    obj_data.loc[:, 'resampled_%s' % parameter_column] = resampled_flux
    obj_data.loc[:, 'resampled_%s' % parameter_error_column] = var ** 0.5

    failures = (obj_data['resampled_%s' % parameter_column] - obj_data[parameter_column]) ** 2 / (
            obj_data[parameter_error_column] ** 2 + obj_data['resampled_%s' % parameter_error_column] ** 2) > 10

    obj_data.loc[failures, 'resampled_%s' % parameter_column] = obj_data.loc[failures, parameter_column]
    obj_data.loc[failures, 'resampled_%s' % parameter_error_column] = obj_data.loc[failures, parameter_error_column]

    if replace:
        obj_data.loc[:, parameter_column] = obj_data.loc[:, 'resampled_%s' % parameter_column]
        obj_data.loc[:, parameter_error_column] = obj_data.loc[:, 'resampled_%s' % parameter_error_column]
        obj_data = obj_data.drop(columns=['resampled_%s' % parameter_column, 'resampled_%s' % parameter_error_column])

    sampled_times = np.arange(min(obj_times), max(obj_times), sample_interval)
    sampled_flux = []
    sampled_mask = np.zeros((len(bands), len(sampled_times)), dtype=np.int32)
    for i, band in enumerate(bands):
        mask = obj_data[band_column] == band
        band_times = obj_data[mask][time_column].values
        sampled_mask[i, :] = (np.min(abs(band_times.reshape((-1, 1)) - sampled_times.reshape((1, -1))), axis=0) <
                              sample_interval)
        pred_x_data = np.vstack([sampled_times, np.ones(len(sampled_times)) * pb_wavelengths[band]]).T
        sampled_flux.append(gp(pred_x_data, return_var=True))
    sampled_flux = np.array(sampled_flux)

    return obj_data, (sampled_times, sampled_flux[:, 0, :], sampled_flux[:, 1, :], sampled_mask)
