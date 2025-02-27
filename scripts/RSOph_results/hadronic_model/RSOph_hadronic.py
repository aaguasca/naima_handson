# Magic command to write the content of the cell to a Python file

#imports
import astropy.units as u
import numpy as np 
import naima
from naima.models import ExponentialCutoffPowerLaw
from astropy.table import Table
import os

class PionDecayModel:
    """
    Define a model for the pion decay emission from a proton energy distribution
    following a power-law with exponential cut-off.
    
    The target proton density is parameterized as a function of the day
    after the nova explosion.
    """
    def __init__(self, day_value):
        # Initialize with the day after the nova explosion
        self.day_value = day_value

    def PionDecay(self, pars, data):
        """
        Define the particle distribution model, radiative model, and return model flux
        at the given energy values from data.

        Parameters:
        pars : array
            Parameters for the model (e.g., normalization, index, cutoff).
        data : Table
            The energy data for which flux needs to be calculated.
        
        Returns:
        flux : Quantity
            The calculated flux at the given energy values.
        """
        # Define particle distribution model with exponential cutoff
        ECPL = ExponentialCutoffPowerLaw(
            10 ** pars[0] / u.eV, 130 * u.GeV, pars[1], 10 ** pars[2] * u.GeV
        )

        # Define the pion decay model using the particle distribution and target density
        PP = naima.models.PionDecay(ECPL, nh=6.0e8 * (self.day_value / 3) ** (-3) * u.cm ** -3)

        # Return the flux for the given energy values
        return PP.flux(data, distance=2.45 * u.kpc)

def lnprior(pars):
    """
    Define a prior distribution for the model parameters.
    Limits the amplitude (norm) to positive values.

    Parameters:
    pars : array
        Parameters for the model.

    Returns:
    logprob : float
        The log probability of the parameters.
    """
    # Limit amplitude to positive domain (uniform prior)
    logprob = naima.uniform_prior(pars[0], 0, np.inf)
               
    return logprob


if __name__ == '__main__':
    # Read in the SED data for each day from CSV files
    sed_day1 = Table.read('../data/RSOph/RSOph_day1_sed.ecsv', format='ascii.ecsv')
    sed_day2 = Table.read('../data/RSOph/RSOph_day2_sed.ecsv', format='ascii.ecsv')
    sed_day3 = Table.read('../data/RSOph/RSOph_day3_sed.ecsv', format='ascii.ecsv')
    sed_day4 = Table.read('../data/RSOph/RSOph_day4_sed.ecsv', format='ascii.ecsv')

    # Group the SED data by day
    daily_data = [sed_day1, sed_day2, sed_day3, sed_day4]

    # Set initial parameters and labels for the model
    p0 = np.array((31, 2.5, np.log10(200)))  # Initial guesses for model parameters
    labels = ["log10(norm)", "index", "log10(cutoff)"]  # Labels for the parameters
    day_array = [1, 2, 3, 4]  # Days after the nova explosion

    # Clean up previous results if they exist
    os.system("rm RSOph_results/hadronic_model/fit_naima_results_all*")

    # Loop over the data for each day
    for yy, iday in enumerate(day_array):
        p0 = np.array((31, 2.5, np.log10(200)))  # Reset initial parameters for each day
        day_value = iday  # Set the day value

        print("day", day_value)

        # Create the model function with the appropriate day value
        PionDecay_func = PionDecayModel(iday)
        
        # Run the sampler to fit the model to the data
        sampler, pos = naima.run_sampler(
            data_table=daily_data[yy],  # Use the data for the current day
            p0=p0,  # Initial parameter guesses
            labels=labels,  # Parameter labels
            model=PionDecay_func.PionDecay,  # Model to fit
            prior=lnprior,  # Prior distribution for parameters
            nwalkers=32,  # Number of walkers for the MCMC sampler
            nburn=10,  # Number of burn-in steps
            nrun=40,  # Number of steps to run the sampler
            threads=4,  # Number of threads to use
            prefit=True,  # Whether to perform a prefit
            interactive=False,  # Whether to allow interactive plotting
        )

        # Save the results of the fitting process
        out_root = "RSOph_results/hadronic_model/fit_naima_results_all_day{}".format(day_array[yy])
        naima.save_run(out_root + ".hdf5", sampler)  # Save the sampler output in HDF5 format

        # Save diagnostic plots and the results table
        naima.save_diagnostic_plots(out_root, sampler, sed=True)
        naima.save_results_table(out_root, sampler) 

        print( )
        print( )
