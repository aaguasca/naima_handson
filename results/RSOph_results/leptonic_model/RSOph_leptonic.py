
# Import necessary libraries
import astropy.units as u
import numpy as np
from astropy.io import ascii
from astropy.table import vstack
import naima
from naima.models import ExponentialCutoffPowerLaw, InverseCompton
import os
from astropy.table import Table

class ICmodel:
    """
    Define a model for the Inverse Compton (IC) emission from an electron distribution,
    following a power-law with exponential cut-off.
    
    The photon energy density is parameterized as a function of the day
    after the nova explosion. The characteristic photon temperature is
    also day-dependent.
    """
    def __init__(self, day_value):
        assert day_value in [1, 2, 3, 4]  # Ensure that the day_value is valid (1 to 4)
        self.day_value = day_value
        # Array of photon temperatures for each day after nova explosion
        Temp_value_array = [10780, 9500, 8460, 7680]
        # Select the temperature for the current day
        self.Temp_value = Temp_value_array[self.day_value - 1]

    def IC(self, pars, data):
        """
        Define particle distribution model, radiative model, and return 
        the flux at the provided energy values.

        Parameters
        ----------
        pars : list
            The model parameters: [amplitude, index, log10(cutoff)].
        data : Table
            Data table containing energy and flux values.

        Returns
        -------
        flux : Quantity
            The computed flux at the provided energy values.
        """
        # Extract parameters for the model: amplitude, index, and cutoff energy
        amplitude = 10**pars[0] / u.eV  # Convert amplitude to proper units (eV^-1)
        alpha = pars[1]  # Spectral index
        e_cutoff = (10 ** pars[2]) * u.GeV  # Exponential cutoff energy

        # Define the Exponential Cutoff Power Law particle distribution model
        ECPL = ExponentialCutoffPowerLaw(amplitude, 130 * u.GeV, alpha, e_cutoff)
        
        # Define the Inverse Compton radiative model using CMB and photosphere photon fields
        IC = InverseCompton(
            ECPL,
            seed_photon_fields=[
                "CMB",  # Cosmic Microwave Background
                ["photosphere", 
                 self.Temp_value * u.K,  # Temperature of the photosphere (dependent on day)
                 0.14 * ((self.Temp_value / 8460) ** 4 / (self.day_value / 3) ** 2) * u.Unit("erg cm-3")]
            ],
        )
        # Compute the flux based on the provided data and distance
        return IC.flux(data, distance=2.45 * u.kpc)


def lnprior(pars):
    """
    Define the prior distribution for the model parameters.
    The amplitude is constrained to be positive.

    Parameters
    ----------
    pars : list
        Model parameters.

    Returns
    -------
    logprob : float
        The logarithmic prior probability for the parameters.
    """
    # Limit amplitude to positive values using a uniform prior
    logprob = naima.uniform_prior(pars[0], 0, np.inf)
    return logprob

if __name__ == '__main__':

    # Set initial model parameters and labels for MCMC sampling
    labels = ["log10(norm)", "index", "log10(cutoff)"]
    day_array = [1, 2, 3, 4]  # Days for which data will be used

    # Read the SED data for each day
    sed_day1 = Table.read('../data/RSOph/RSOph_day1_diff_flux.ecsv', format='ascii.ecsv')
    sed_day2 = Table.read('../data/RSOph/RSOph_day2_diff_flux.ecsv', format='ascii.ecsv')
    sed_day3 = Table.read('../data/RSOph/RSOph_day3_diff_flux.ecsv', format='ascii.ecsv')
    sed_day4 = Table.read('../data/RSOph/RSOph_day4_diff_flux.ecsv', format='ascii.ecsv')

    # Group the daily data into a list for easy access
    daily_data = [sed_day1, sed_day2, sed_day3, sed_day4]

    # Remove any previous results to avoid overwriting issues
    os.system("rm RSOph_results/leptonic_model/fit_naima_results_all*")

    # Loop through each day of data
    for yy, iday in enumerate(day_array):
        
        p0 = np.array((27, 2.5, np.log10(200)))  # Initial guess for model parameters
        day_value = iday  # Set the current day value
        IC_func = ICmodel(day_value)  # Create the IC model for the current day

        print("day", day_value)

        # Run the MCMC sampler to fit the model to the data
        sampler, pos = naima.run_sampler(
            data_table=daily_data[yy],  # The data for the current day
            p0=p0,  # Initial parameters
            labels=labels,  # Parameter labels
            model=IC_func.IC,  # The model to fit (IC emission)
            prior=lnprior,  # The prior distribution
            nwalkers=50,  # Number of walkers in the MCMC chain
            nburn=50,  # Number of burn-in steps
            nrun=40,  # Number of sampling steps
            prefit=True,  # Perform a prefitting step with the initial parameters
            interactive=True,  # Allow interactive adjustment of p0 (only in command line)
        )
        
        # Save the results of the sampling run to a file
        out_root = "RSOph_results/leptonic_model/fit_naima_results_all_day{}".format(day_array[yy])
        naima.save_run(out_root + ".hdf5", sampler)

        # Save diagnostic plots and results table
        naima.save_diagnostic_plots(out_root, sampler, sed=True)
        naima.save_results_table(out_root, sampler)

        print()
        print()
