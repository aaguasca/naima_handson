
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
    Define a model for the IC emission from a electron distribution
    following a power-law with exponential cut-off.
    
    A variable photon energy density and characteristic photon temperature. 
    The former is parameterized as a function of the day after the nova explosion.
    """
    def __init__(self, day_value):
        assert day_value in [1,2,3,4]
        self.day_value = day_value
        Temp_value_array = [10780,9500,8460,7680]
        self.Temp_value = Temp_value_array[self.day_value-1]

    def IC(self, pars, data):
        """
        Define particle distribution model, radiative model, and return model flux
        at data energy values
        """
        amplitude = 10**pars[0] / u.eV
        alpha = pars[1]
        e_cutoff = (10 ** pars[2]) * u.GeV

        ECPL = ExponentialCutoffPowerLaw(amplitude, 130 * u.GeV, alpha, e_cutoff)
        IC = InverseCompton(
            ECPL,
            seed_photon_fields=[
                "CMB",
                ["photosphere", 
                self.Temp_value*u.K, 
                0.14*((self.Temp_value/8460)**4/(self.day_value/3)**2)*u.Unit("erg cm-3")]
            ],
        )
        return IC.flux(data, distance=2.45 * u.kpc)


def lnprior(pars):
    # Limit amplitude to positive domain
    logprob = naima.uniform_prior(pars[0], 0, np.inf)
            
    return logprob

if __name__ == '__main__':

    ## Set initial parameters and labels    
    labels = ["log10(norm)", "index", "log10(cutoff)"]
    day_array=[1,2,3,4]

    sed_day1 = Table.read('../data/RSOph/RSOph_day1_sed.ecsv',format='ascii.ecsv')
    sed_day2 = Table.read('../data/RSOph/RSOph_day2_sed.ecsv',format='ascii.ecsv')
    sed_day3 = Table.read('../data/RSOph/RSOph_day3_sed.ecsv',format='ascii.ecsv')
    sed_day4 = Table.read('../data/RSOph/RSOph_day4_sed.ecsv',format='ascii.ecsv')

    daily_data = [sed_day1, sed_day2, sed_day3, sed_day4]

    os.system("rm RSOph_results/leptonic_model/fit_naima_results_all*")

    for yy,iday in enumerate(day_array):
        
        p0 = np.array((27, 2.5, np.log10(200)))
        day_value=iday
        IC_func=ICmodel(day_value)

        print("day",day_value)

        # Run sampler    
        sampler, pos = naima.run_sampler(
            data_table=daily_data[yy],
            p0=p0,
            labels=labels,
            model=IC_func.IC,
            prior=lnprior,
            nwalkers=50,
            nburn=50,
            nrun=40,
            # threads=4,
            prefit=True, # needed to start in good initial parameters
            interactive=True, #adjust p0 interatively with pop-up window (it only works from command line)
        );
        ## Save run results
        out_root = "RSOph_results/leptonic_model/fit_naima_results_all_day{}".format(day_array[yy])
        naima.save_run(out_root + ".hdf5", sampler)

        ## Save diagnostic plots and results table
        naima.save_diagnostic_plots(out_root, sampler, sed=True)
        naima.save_results_table(out_root, sampler) 
        print( )
        print( )
