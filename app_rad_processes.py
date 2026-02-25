import streamlit as st
import numpy as np
import astropy.units as u
from astropy.io import ascii
from astropy.constants import c, h
import naima
import matplotlib.pyplot as plt
import os
from astropy.constants import c, sigma_sb, k_B, h


# --- Functions for the slider ---
st.set_page_config(page_title="Multiwavelength Radiation Models", layout="wide")

def dist_axis():
    """
    Define the range of distances in kpc for the slider.
    """
    dist_values_kpc = np.geomspace(0.1, 10, 20)
    ini_dist_values_kpc = dist_values_kpc[int(len(dist_values_kpc)/2)]
    return dist_values_kpc, ini_dist_values_kpc

def p_index_axis():
    """
    Define the range of spectral indices for the slider.
    """
    p_values = np.arange(15,30+1)/10
    p_ini_value = 2
    assert p_ini_value in p_values
    return p_values, p_ini_value

def e_cutoff_axis_GeV():
    """
    Define the range of cutoff energies in GeV for the slider.
    """
    ecut_values_GeV = np.geomspace(10, 1e4, 25)
    ini_ecut_values_GeV = ecut_values_GeV[int(len(ecut_values_GeV)/2)]
    return ecut_values_GeV, ini_ecut_values_GeV

def e_cutoff_axis_TeV():
    """
    Define the range of cutoff energies in TeV for the slider.
    """
    ecut_values_TeV = np.geomspace(10, 1e5, 25)
    ini_ecut_values_TeV = ecut_values_TeV[int(len(ecut_values_TeV)/2)]
    return ecut_values_TeV, ini_ecut_values_TeV

def density_pion_decay_axis():
    """
    Define the range of target densities in cm^-3 in the pion decay model for the slider.
    """
    density_values_cm3 = np.geomspace(10, 1e10, 10)
    ini_density_value_cm3 = 1e8
    assert ini_density_value_cm3 in density_values_cm3
    return density_values_cm3, ini_density_value_cm3

def density_brem_axis():
    """
    Define the range of target densities in cm^-3 in the brem model for the slider.
    """    
    density_values_cm3 = np.geomspace(10, 1e10, 10)
    ini_density_value_cm3 = 1e7
    assert ini_density_value_cm3 in density_values_cm3
    return density_values_cm3, ini_density_value_cm3

def mag_field_sync_axis():
    """
    Define the range of magnetic fields in uGauss in the synchrotron model for the slider.
    """
    values_B_uGauss = np.geomspace(0.1, 1e4, 51)
    ini_value_B_uGauss = 10
    assert ini_value_B_uGauss in values_B_uGauss
    return values_B_uGauss, ini_value_B_uGauss

def mag_field_ssc_axis():
    """
    Define the range of magnetic fields in uGauss in the synchrotron-self-Compton model for the slider.
    """
    values_B_uGauss = np.geomspace(0.1, 1e4, 51)
    ini_value_B_uGauss = 10
    assert ini_value_B_uGauss in values_B_uGauss
    return values_B_uGauss, ini_value_B_uGauss

def photon_field_axis():
    """
    Define the photon fields in the inverse Compton model for the slider. The parameters
    of each photon seed are defined in the get_inverse_compton_sed function.
    """
    photon_field_list = ["CMB", "FIR", "NIR", "Hot star"]
    ini_photon_field = photon_field_list[0]
    return photon_field_list, ini_photon_field

def reset_all_parameters():
    """
    Reset all parameters of the models to their default values.
    """
    _, default_dist = dist_axis()
    _, default_ecut = e_cutoff_axis_GeV()
    _, default_p = p_index_axis()
    _, default_dens_pion = density_pion_decay_axis()
    _, default_dens_brem = density_brem_axis()
    _, default_B_sync = mag_field_sync_axis()
    _, default_B_ssc = mag_field_ssc_axis()
    _, default_photon_field = photon_field_axis()
    
    if st.session_state.active_tab == "Pion Decay":
        st.session_state["dist_pion"] = default_dist
        st.session_state["e_cutoff_pion"] = default_ecut
        st.session_state["p_index_pion"] = default_p
        st.session_state["dens_pion"] = default_dens_pion
    elif st.session_state.active_tab == "Bremsstrahlung":
        st.session_state["dist_brem"] = default_dist
        st.session_state["e_cutoff_brem"] = default_ecut
        st.session_state["p_index_brem"] = default_p
        st.session_state["dens_brem"] = default_dens_brem
    elif st.session_state.active_tab == "Synchrotron":
        st.session_state["dist_sync"] = default_dist
        st.session_state["e_cutoff_sync"] = default_ecut
        st.session_state["p_index_sync"] = default_p
        st.session_state["B_sync"] = default_B_sync
    elif st.session_state.active_tab == "Inverse Compton":
        st.session_state["dist_ic"] = default_dist
        st.session_state["e_cutoff_ic"] = default_ecut
        st.session_state["p_index_ic"] = default_p
        st.session_state["photon_field_ic"] = default_photon_field
    elif st.session_state.active_tab == "Synchrotron-self Compton":
        st.session_state["dist_ssc"] = default_dist
        st.session_state["e_cutoff_ssc"] = default_ecut
        st.session_state["p_index_ssc"] = default_p
        st.session_state["B_ssc"] = default_B_ssc

# --- Functions for the radiative models ---
@st.cache_data
def get_pion_decay_sed(distance_kpc, p, e_cutoff_GeV, n_target_cm3):
    """
    Compute the spectral energy distribution (SED) for neutral pion decay given a 
    certain distance, spectral index p, cutoff energy, and target density.

    Parameters
    ----------
    distance_kpc : float
        Distance to the source in kpc.
    p : float
        Spectral index of the power-law model.
    e_cutoff_GeV : float
        Cutoff energy in GeV.
    n_target_cm3 : float
        Target density in cm^-3.

    Returns
    -------
    energy_range : np.ndarray
        Energy range in eV.
    sed : np.ndarray
        Spectral energy distribution in erg cm^-2 s^-1.
    wp : float
        Total energy in protons above 1 GeV in erg.
    """
    energy_range = np.logspace(8, 13, 50) * u.eV
    distance = distance_kpc * u.kpc  # Assign kpc unit to the distance_kpc variable, hereafter distance
    e_cutoff = e_cutoff_GeV * u.GeV  # Assign GeV unit to the e_cutoff_GeV variable, hereafter e_cutoff
    n_target = n_target_cm3 * u.cm ** -3  # Convert target density to proper unit
    part_dist = naima.models.ExponentialCutoffPowerLaw(
        amplitude=1e36 / u.eV,  # Set a normalization value for the spectrum
        e_0=1 * u.GeV,  # Reference energy of 1 GeV
        alpha=p,  # Spectral index of particles (determines shape of the power-law model)
        e_cutoff=e_cutoff  # Apply the user-defined cutoff energy
    )
    # Create a pion decay radiation model using the particle distribution and target density
    rad_models = naima.models.PionDecay(particle_distribution=part_dist, nh=n_target)
    
    # Compute the spectral energy distribution (SED) for the given energy range and distance
    sed = rad_models.sed(photon_energy=energy_range, distance=distance)
    
    # Compute the total energy in protons above 1 GeV
    wp = rad_models.compute_Wp(1*u.GeV).to(u.erg).value
    
    return energy_range.value, sed.to(u.Unit("erg cm-2 s-1")).value, wp

@st.cache_data
def get_bremsstrahlung_sed(distance_kpc, p, e_cutoff_GeV, n_medium_cm3):
    """
    Compute the spectral energy distribution (SED) for bremsstrahlung given a 
    certain distance, spectral index p, cutoff energy, and medium density.

    Parameters
    ----------
    distance_kpc : float
        Distance to the source in kpc.
    p : float
        Spectral index of the power-law model.
    e_cutoff_GeV : float
        Cutoff energy in GeV.
    n_medium_cm3 : float
        Target density in cm^-3.

    Returns
    -------
    energy_range : np.ndarray
        Energy range in eV.
    sed : np.ndarray
        Spectral energy distribution in erg cm^-2 s^-1.
    we : float
        Total energy in electrons above 1 GeV in erg.
    """

    energy_range = np.logspace(8, 13, 50) * u.eV
    # Convert input values to astropy units
    distance = distance_kpc * u.kpc
    e_cutoff = e_cutoff_GeV * u.GeV
    n_medium = n_medium_cm3 * u.cm ** -3
    # Define particle distribution with the given cutoff energy
    part_dist = naima.models.ExponentialCutoffPowerLaw(
        amplitude=1e31 / u.eV,
        e_0=100 * u.GeV,
        alpha=p,
        e_cutoff=e_cutoff
    )
    rad_models = naima.models.Bremsstrahlung(part_dist, n0=n_medium)
    sed = rad_models.sed(energy_range, distance=distance)
    we = rad_models.compute_We(1*u.GeV).to(u.erg).value
    return energy_range.value, sed.to(u.Unit("erg cm-2 s-1")).value, we

@st.cache_data
def get_synchrotron_sed(distance_kpc, p, e_cutoff_GeV, B_uGauss):
    """
    Compute the spectral energy distribution (SED) for synchrotron radiation given a 
    certain distance, spectral index p, cutoff energy, and magnetic field.

    Parameters
    ----------
    distance_kpc : float
        Distance to the source in kpc.
    p : float
        Spectral index of the power-law model.
    e_cutoff_GeV : float
        Cutoff energy in GeV.
    B_uGauss : float
        Magnetic field in uG.

    Returns
    -------
    energy_range : np.ndarray
        Energy range in eV.
    sed : np.ndarray
        Spectral energy distribution in erg cm^-2 s^-1.
    we : float
        Total energy in electrons above 1 GeV in erg.
    """
    distance = distance_kpc * u.kpc
    e_cutoff = e_cutoff_GeV * u.GeV
    magnetic_field = B_uGauss * u.uG
    energy_range = np.logspace(-9, 13, 100) * u.eV
    part_dist = naima.models.ExponentialCutoffPowerLaw(
        amplitude=1e36 / u.eV,
        e_0=100 * u.GeV,
        alpha=p,
        e_cutoff=e_cutoff
    )
    rad_models = naima.models.Synchrotron(part_dist, B=magnetic_field)
    sed = rad_models.sed(energy_range, distance=distance)
    we = rad_models.compute_We(1*u.GeV).to(u.erg).value
    return energy_range.value, sed.to(u.Unit("erg cm-2 s-1")).value, we

@st.cache_data
def get_inverse_compton_sed(distance_kpc, p, e_cutoff_GeV, photon_field):
    """
    Compute the spectral energy distribution (SED) for inverse Compton given a 
    certain distance, spectral index p, cutoff energy, and photon field.

    Parameters
    ----------
    distance_kpc : float
        Distance to the source in kpc.
    p : float
        Spectral index of the power-law model.
    e_cutoff_GeV : float
        Cutoff energy in GeV.
    photon_field : float
        Photon field.

    Returns
    -------
    energy_range : np.ndarray
        Energy range in eV.
    sed : np.ndarray
        Spectral energy distribution in erg cm^-2 s^-1.
    we : float
        Total energy in electrons above 1 GeV in erg.
    """
    energy_range = np.logspace(8, 13, 100) * u.eV
    distance = distance_kpc * u.kpc
    e_cutoff = e_cutoff_GeV * u.GeV
    photon_seed = photon_field  # No conversion needed
    # Define particle distribution with the given cutoff energy
    part_dist = naima.models.ExponentialCutoffPowerLaw(
        amplitude=1e36 / u.eV,
        e_0=100 * u.GeV,
        alpha=p,
        e_cutoff=e_cutoff
    )
    # Define additional label for the plot
    if photon_field == "CMB":
        additional_label = "\n($T=2.72$ K, $U_{{\\rm ph}}=0.261$ eV/cm³)"
    elif photon_field == "FIR":
        additional_label = "\n($T=30$ K, $U_{{\\rm ph}}=0.5$ eV/cm³)"
    elif photon_field == "NIR":
        additional_label = "\n($T=3000$ K, $U_{{\\rm ph}}=1$ eV/cm³)"
    elif photon_field == "Hot star":
        Temp=30000 * u.K
        # characteristic_temp=(Temp*k_B).to("eV")
        star_radius=10 * u.Rsun
        radius_process=5000 * u.Rsun
        L=4 * np.pi * sigma_sb * star_radius**2 * Temp**4
        Uph=(L/(4*np.pi*c*radius_process**2)).to("erg cm-3")
        photon_seed=['Hot star', Temp, Uph] 
        additional_label = f"\n($T={Temp.to_value(u.K):1.0f}$ K"+\
                           f", $U_{{\\rm ph}}={Uph.to_value('erg cm-3'):1.3f}$ erg/cm³)"
    else:
        additional_label = ""

    # Create radiation model for Inverse Compton
    rad_models = naima.models.InverseCompton(part_dist, seed_photon_fields=[photon_seed])
    sed = rad_models.sed(energy_range, distance=distance)
    we = rad_models.compute_We(1*u.GeV).to(u.erg).value
    return energy_range.value, sed.to(u.Unit("erg cm-2 s-1")).value, we, additional_label

@st.cache_data
def get_synchrotron_self_compton_sed(distance_kpc, p, e_cutoff_GeV, B_uGauss):
    """
    Compute the spectral energy distribution (SED) for synchrotron-self-compton given a 
    certain distance, spectral index p, cutoff energy, and magnetic field.

    Parameters
    ----------
    distance_kpc : float
        Distance to the source in kpc.
    p : float
        Spectral index of the power-law.
    e_cutoff_GeV : float
        Cutoff energy in GeV.
    B_uGauss : float
        Magnetic field in uG.

    Returns
    -------
    energy_range : np.ndarray
        Energy range in eV.
    sed : np.ndarray
        Spectral energy distribution in erg cm^-2 s^-1.
    we : float
        Total energy in electrons above 1 GeV in erg.
    """

    # Define the energy range for the spectrum (from 10^-10 eV to 10^14 eV with 100 points)
    spectrum_energy = np.logspace(-10, 14, 100) * u.eV

    # Convert input values to appropriate astropy units
    distance = distance_kpc * u.kpc
    e_cutoff = e_cutoff_GeV * u.GeV
    B = B_uGauss * u.uG
    
    # Define a particle distribution model using an exponential cutoff power law
    part_dist = naima.models.ExponentialCutoffPowerLaw(
        amplitude=1e36 / u.eV,
        e_0=100 * u.GeV,
        alpha=p,
        e_cutoff=e_cutoff
    )
    
    # Compute Synchrotron emission based on the particle distribution and magnetic field
    SYN = naima.models.Synchrotron(part_dist, B=B)
    
    # Define energy array for synchrotron seed photon field
    Esy = np.logspace(-6, 6, 100) * u.eV  # Energy range for synchrotron photons
    Lsy = SYN.flux(Esy, distance=0 * u.cm)  # Compute synchrotron flux at the source

    # Define source radius and compute photon density
    R = 0.01 * u.pc  # Source radius set to 0.01 parsecs
    # The factor 2.24 accounts for geometrical considerations of a uniform spherical emitter
    phn_sy = Lsy / (4 * np.pi * R**2 * c) * 2.24  # Compute seed photon energy density

    # Compute Inverse Compton emission including multiple seed photon fields
    IC = naima.models.InverseCompton(
        part_dist, seed_photon_fields=[['SSC', Esy, phn_sy]]#, 'CMB', 'FIR', 'NIR']
    )

    # Compute SEDs for both Synchrotron and Inverse Compton
    sed_IC = IC.sed(spectrum_energy, distance=distance)
    sed_SYN = SYN.sed(spectrum_energy, distance=distance)
    we = IC.compute_We(1*u.GeV).to(u.erg).value

    return spectrum_energy.value, sed_SYN.to(u.Unit("erg cm-2 s-1")).value, sed_IC.to(u.Unit("erg cm-2 s-1")).value, we


# --- Functions for the Crab Nebula fit by hand ---
@st.cache_data
def load_crab_data():
    """
    Load the Crab Nebula data from the ECSV file.

    Returns
    -------
    data : astropy.table.Table
        Table containing the Crab Nebula data.
    """
    # Asegúrate de que la ruta al archivo sea correcta en tu sistema
    try:
        return ascii.read("./data/CrabNebula/CrabNebula_spectrum.ecsv")
    except:
        # Fallback por si el archivo no está en la ruta
        st.error("Crab Nebula data file not found. Please check the file path.")
        return None

@st.cache_data
def get_Crab_synch_ic_sed(dist, e_cut_tev, b_ugauss):
    """
    Compute the spectral energy distribution (SED) for synchrotron-self-compton given a 
    certain distance, cutoff energy, and magnetic field.

    Parameters
    ----------
    distance_kpc : float
        Distance to the source in kpc.
    e_cut_tev : float
        Cutoff energy in TeV.
    B_uGauss : float
        Magnetic field in uG.

    Returns
    -------
    energy_range : np.ndarray
        Energy range in eV.
    sed_syn : np.ndarray
        Synchrotron component of the spectral energy distribution in erg cm^-2 s^-1.
    sed_ic : np.ndarray
        Inverse Compton component of the spectral energy distribution in erg cm^-2 s^-1.
    """
    spectrum_energy = np.logspace(-10, 14, 100) * u.eV

    # Definir unidades
    distance = dist * u.kpc
    e_cutoff = e_cut_tev * u.TeV
    B = b_ugauss * u.uG
    
    part_dist = naima.models.ExponentialCutoffBrokenPowerLaw(
        amplitude=3.699e36 / u.eV,  # Amplitude (normalization) for the spectrum
        e_0=1 * u.TeV,  # Reference energy (in TeV)
        e_break=0.265 * u.TeV,  # Break energy in the broken power law
        alpha_1=1.5,  # Spectral index before the break
        alpha_2=3.233,  # Spectral index after the break
        e_cutoff=e_cutoff,  # Apply the user-defined cutoff energy
        beta=2.0,  # Break smoothness factor
    )
    
    # Compute Synchrotron emission based on the particle distribution and magnetic field
    SYN = naima.models.Synchrotron(part_dist, B=B, Eemax=50 * u.PeV, Eemin=0.1 * u.GeV)
    
    # Define an energy array for the synchrotron seed photon field (synchrotron photons)
    Esy = np.logspace(-6, 6, 100) * u.eV
    Lsy = SYN.flux(Esy, distance=0 * u.cm)

    # Define the source radius and calculate the photon energy density for synchrotron seed photons
    R = 2 * u.pc
    phn_sy = Lsy / (4 * np.pi * R**2 * c) * 2.24 

    # Compute Inverse Compton emission considering various seed photon fields (CMB, FIR, NIR, SSC)
    IC = naima.models.InverseCompton(
        part_dist,     
        seed_photon_fields=[
            "CMB",
            ["FIR", 70 * u.K, 0.5 * u.eV / u.cm**3],
            ["NIR", 5000 * u.K, 1 * u.eV / u.cm**3],
            ["SSC", Esy, phn_sy],
        ],
    )

    # Compute the Spectral Energy Distributions (SED) for Synchrotron and Inverse Compton
    spectrum_energy = np.logspace(-10, 14, 100) * u.eV
    sed_IC = IC.sed(spectrum_energy, distance=distance)
    sed_SYN = SYN.sed(spectrum_energy, distance=distance)
    
    return spectrum_energy.value, sed_SYN.value, sed_IC.value


# --- General Sidebar Structure ---

# URL oficial del logo de la UB (puedes usar esta o una ruta local "logo_ub.png")
logo_ub_url = "https://upload.wikimedia.org/wikipedia/commons/e/e6/Logo_Universitat_de_Barcelona.png"
logo_ub_url = "https://www.iqtc.ub.edu/wp-content/themes/css-grid/img/logo_ub.png"
logo_ub_url = "https://www.ub.edu/portal/documents/6298690/7342921/Logo+Petit.png/c61f1b2f-ca0e-f78f-5f63-a43ffe93ce1f?t=1580137823945"

col1, col2, col3 = st.sidebar.columns([0.1, 1, 0.1])
with col2:
    st.image(logo_ub_url, use_container_width=True)
# Add Facultat de Física
# st.sidebar.markdown("<h3 style='text-align: center;'>Facultat de Física</h3>", unsafe_allow_html=True)

st.sidebar.divider()

st.sidebar.header("Control Panel")

# 1. Model Selection. Choose between different radiative processes or fit the Crab Nebula spectrum by eye.
tab_selection = st.sidebar.selectbox(
    "Select Option:",
    ["Radiative Processes", "Crab Nebula fit"],
    help="Select if you want to display different radiative processes or fit the Crab Nebula spectrum by eye."
)

st.sidebar.divider() # Línea visual divisoria


# --- Main Content Area ---
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 5px;">
        <span style="color: #666; font-size: 0.9em;">powered by</span>
        <strong style="font-size: 1.1em; color: #304C7A;">naima</strong>
        <a href="https://naima.readthedocs.io/en/latest/" target="_blank" style="display: flex; align-items: center; text-decoration: none;">
            <svg height="24" width="24" viewBox="0 0 16 16" style="fill: #24292e;">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
            </svg>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)


if tab_selection == "Radiative Processes":

    # 2. Initialize the active tab if it doesn't exist
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Bremsstrahlung"


    # --- Main area: Buttons for the radiative processes ---
    st.title("Radiative Processes")
    st.markdown("Select the radiative process to display the spectral energy distribution (SED).")


    # 1. Inject CSS to equalize the height and behavior of the buttons
    st.markdown(
        """
        <style>
        /* Force all buttons to have the same minimum height */
        /* and that the text adjusts if necessary */
        div.stButton > button {
            min-height: 60px;           /* Adjust this value if you want more or less height */
            height: auto;
            padding-top: 10px;
            padding-bottom: 10px;
            line-height: 1.2;
        }
        
        /* Ensure long text does not break the layout */
        div.stButton > button p {
            font-size: 14px;            /* Uniform font size */
            word-wrap: break-word;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
            
    # Create five columns that will function as buttons
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("Pion Decay", use_container_width=True):
            st.session_state.active_tab = "Pion Decay"
    with col2:
        if st.button("Bremsstrahlung", use_container_width=True):
            st.session_state.active_tab = "Bremsstrahlung"    
    with col3:
        if st.button("Synchrotron", use_container_width=True):
            st.session_state.active_tab = "Synchrotron"    
    with col4:
        if st.button("Inverse Compton", use_container_width=True):
            st.session_state.active_tab = "Inverse Compton"    
    with col5:
        if st.button("Synchrotron-self Compton", use_container_width=True):
            st.session_state.active_tab = "Synchrotron-self Compton"    

    # 3. Dividing line to separate the buttons from the content
    st.divider()

    # --- Sidebar content of the radiative process ---
    # Display the active tab
    st.sidebar.info(f"Modifying: {st.session_state.active_tab}")

    st.sidebar.header("Free parameters to change")

    # --- Button 1: Pion Decay ---
    if st.session_state.active_tab == "Pion Decay":

        # Botón de Reset específico para esta sección
        st.sidebar.button("⟳ Reset Parameters", on_click=reset_all_parameters, use_container_width=True)

        # Distance
        dist_values_kpc, ini_dist_values_kpc = dist_axis()
        distance_kpc = st.sidebar.select_slider(
            "Distance, $d~(\\rm{kpc})$", 
            options=dist_values_kpc,
            value=ini_dist_values_kpc, # Initial value
            format_func=lambda x: f"{x:.1f}", # Para que los números se vean limpios
            key="dist_pion",
            help="Distance to the emitting source in kiloparsecs (kpc)."
        )

        # Spectral Index
        p_values, p_ini_value = p_index_axis()
        p_index = st.sidebar.select_slider(
            "Spectral Index, $p$", 
            options=p_values,
            value=p_ini_value, # Initial value
            format_func=lambda x: f"{x:.1f}", # Para que los números se vean limpios
            key="p_index_pion",
            help="Spectral index of the particle energy distribution."
        )    

        # Cutoff
        ecut_values_GeV, ini_ecut_values_GeV = e_cutoff_axis_GeV()
        e_cutoff_GeV = st.sidebar.select_slider(
            "$E_{\\rm cutoff}~(\\rm{GeV})$", 
            options=ecut_values_GeV,
            value=ini_ecut_values_GeV,
            format_func=lambda x: f"{x:.0f}", # Para que los números se vean limpios
            key="e_cutoff_pion",
            help="Cutoff energy of the particle energy distribution (maximum energy of the relativistic particles) in gigaelectronvolts (GeV)."
        )

        # Density
        density_values_cm3, ini_density_values_cm3 = density_pion_decay_axis()
        n_target_cm3 = st.sidebar.select_slider(
            "Target density, $n_{\\rm target}$ $\\left(\\rm{cm}^{-3}\\right)$", 
            options=density_values_cm3,
            value=ini_density_values_cm3, # Valor inicial (el primero de la lista)
            format_func=lambda x: f"{x:.2e}", # Para que los números se vean limpios
            key="dens_pion",
            help="Target number density for proton-proton interactions in particles per cubic centimeter (cm⁻³)."
        )

        st.sidebar.header("Fixed parameters")
        st.sidebar.caption(
            "Energy distribution of the relativistic protons:\n"
            "- Model: Exponential cutoff power law\n"
            "- Amplitude: $10^{{36}}$ protons $\\rm{{eV}}^{{-1}}$ @ $1~\\rm{GeV}$"
        )

        # st.sidebar.divider()
        # st.sidebar.write(f"**Current Values:**")
        # st.sidebar.caption(f"Dist: {distance_kpc:.2f} kpc | p: {p_index:.1f}")
        # st.sidebar.caption(f"E_cut: {e_cutoff_GeV:.0f} GeV | n: {n_target_cm3:.1e}")

        st.subheader("Proton-Proton Interactions (neutral pion decay)")
        # st.write(f"Valor seleccionado: {n_target_cm3:.4f}")


        e_v, s_v, we = get_pion_decay_sed(distance_kpc, p_index, e_cutoff_GeV, n_target_cm3)
        
        fig1, ax1 = plt.subplots(figsize=(9, 5))
        ax1.loglog(e_v, s_v, color='tab:blue', lw=2, 
            label=f"$d = {distance_kpc:.2f}$ kpc,\n"+\
            f"$p = {p_index:.1f}$,\n"+\
            f"$E_{{\\rm cutoff}} = {e_cutoff_GeV:.0f}$ GeV,\n"+\
            f"$n_{{\\rm target}} = {n_target_cm3:.1e}$ cm$^{{-3}}$"
        )

        ax1.set_title(f"Energy in protons ($E > 1~\\rm{{GeV}}$): {we:1.1e} erg")
        ax1.set_xlabel("Energy, $E$ (eV)")
        ax1.set_ylabel(r"$E^2 d\phi/dE$ (erg cm$^{-2}$ s$^{-1}$)")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.set_ylim(1e-13, 1e-9)
        ax1.legend(title="Pion Decay")
        st.pyplot(fig1)

        with st.expander("Diagram"):
            # Ruta de la imagen
            img_path = "data/images/npi_decay.png"
            # Verificamos si el archivo existe para evitar errores
            if os.path.exists(img_path):
                st.image(
                    img_path, 
                    caption="Diagram of proton-proton interactions and subsequent neutral pion decay.", 
                    width=500#use_container_width=True
                )
                st.markdown(
                    """
                    Source: C. Grupen, Astroparticle Physics, Undergraduate Texts in Physics, Springer (2020), 10.1007/978-3-030-27339-2
                    """
                )                
            else:
                st.error(f"Image not found at {img_path}")            

        with st.expander("Typical number densities in astrophysical sources"):
            st.markdown(
                """
                For reference, below you can find some typical number densities in astrophysical sources (from Metzger et al. 2016, MNRAS, 457, 1786):

                - Intergalactic medium : n ~ $10^{-5}$&ndash;$10^{-4}~\\rm{cm}^{-3}$

                - Supernova (SN) remnants : n ~ $1~\\rm{cm}^{-3}$

                - Interplanetary medium : n ~ $1$&ndash;$10~\\rm{cm}^{-3}$

                - SNR-cloud interactions: n ~ $10$&ndash;$10^{3}~\\rm{cm}^{-3}$

                - Regions in which there are massive young stellar objects: n ~ $10^{3}$&ndash;$10^{6}~\\rm{cm}^{-3}$ (from Bosch-Ramon et al. 2010, A&A, 511, A8)

                - Type IIn SNe: n ~ $10^{6}$&ndash;$10^{10}~\\rm{cm}^{-3}$

                - Novae: n ~ $10^{8}$&ndash;$10^{11}~\\rm{cm}^{-3}$
                """
            )        

    # --- Button 2: Bremsstrahlung ---
    if st.session_state.active_tab == "Bremsstrahlung":

        # Botón de Reset específico para esta sección
        st.sidebar.button("⟳ Reset Parameters", on_click=reset_all_parameters, use_container_width=True)

        # Distance
        dist_values_kpc, ini_dist_values_kpc = dist_axis()
        distance_kpc = st.sidebar.select_slider(
            "Distance, $d~(\\rm{kpc})$", 
            options=dist_values_kpc,
            value=ini_dist_values_kpc, # Initial value
            format_func=lambda x: f"{x:.1f}", # Para que los números se vean limpios
            key="dist_brem",
            help="Distance to the emitting source in kiloparsecs (kpc)."
        )

        # Spectral Index
        p_values, p_ini_value = p_index_axis()
        p_index = st.sidebar.select_slider(
            "Spectral Index, $p$", 
            options=p_values,
            value=p_ini_value, # Initial value
            format_func=lambda x: f"{x:.1f}", # Para que los números se vean limpios
            key="p_index_brem",
            help="Spectral index of the particle energy distribution."
        )    

        # Cutoff
        ecut_values_GeV, ini_ecut_values_GeV = e_cutoff_axis_GeV()
        e_cutoff_GeV = st.sidebar.select_slider(
            "$E_{\\rm cutoff}~(\\rm{GeV})$", 
            options=ecut_values_GeV,
            value=ini_ecut_values_GeV,
            format_func=lambda x: f"{x:.0f}", # Para que los números se vean limpios
            key="e_cutoff_brem",
            help="Cutoff energy of the particle energy distribution (maximum energy of the relativistic particles) in gigaelectronvolts (GeV)."
        )

        # Density
        values_dens_brems, ini_value_dens_brems= density_brem_axis()
        n_medium_cm3 = st.sidebar.select_slider(
            "Medium density, $n_{\\rm medium}$ $\\left(\\rm{cm}^{-3}\\right)$", 
            options=values_dens_brems,
            value=ini_value_dens_brems,
            format_func=lambda x: f"{x:.2e}",
            key="dens_brems",
            help="Medium number density for Bremsstrahlung in particles per cubic centimeter (cm⁻³."
        )

        st.sidebar.header("Fixed parameters")
        st.sidebar.caption(
            "Energy distribution of the relativistic electrons:\n "
            "- Model: Exponential cutoff power law\n "
            "- Amplitude: $10^{{31}}$ electrons $\\rm{{eV}}^{{-1}}$ @ $100~\\rm{GeV}$"
        )

        # st.sidebar.write(f"**Current Values:**")
        # st.sidebar.caption(f"Dist: {distance_kpc:.2f} kpc | p: {p_index:.1f}")
        # st.sidebar.caption(f"E_cut: {e_cutoff_GeV:.0f} GeV | n: {n_medium_cm3:.1e}")

        st.subheader("Bremsstrahlung radiation")
        e_v2, s_v2, we = get_bremsstrahlung_sed(distance_kpc, p_index, e_cutoff_GeV, n_medium_cm3)
        
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        ax2.loglog(e_v2, s_v2, color='tab:red', lw=2, label=f"$d = {distance_kpc:.2f}$ kpc,\n"+\
            f"$p = {p_index:.1f}$,\n"+\
            f"$E_{{\\rm cutoff}} = {e_cutoff_GeV:.0f}$ GeV,\n"+\
            f"$n_{{\\rm medium}} = {n_medium_cm3:.1e}$ cm$^{{-3}}$"
        )

        ax2.set_title(f"Energy in electrons ($E > 1~\\rm{{GeV}}$): {we:1.1e} erg")
        ax2.set_xlabel("Energy, $E$ (eV)")
        ax2.set_ylabel(r"$E^2 d\phi/dE$ (erg cm$^{-2}$ s$^{-1}$)")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.set_ylim(1e-13, 1e-9)
        ax2.legend(title="Bremsstrahlung")
        st.pyplot(fig2)

        with st.expander("Diagram"):
            # Ruta de la imagen
            img_path = "data/images/bremsstrahlung.png"
            # Verificamos si el archivo existe para evitar errores
            if os.path.exists(img_path):
                st.image(
                    img_path, 
                    caption="Diagram of electron undergoing bremsstrahlung on a nucleus.", 
                    width=500#use_container_width=True
                )
                st.markdown(
                    """
                    Source: C. Grupen, Astroparticle Physics, Undergraduate Texts in Physics, Springer (2020), 10.1007/978-3-030-27339-2
                    """
                )
            else:
                st.error(f"Image not found at {img_path}")            

        st.info("Naima considers a fully ionised medium with solar abundances. Electron-ion bremsstrahlung is the only process that is considered above $10~ \\rm MeV$.")


    # --- Button 3: Synchrotron ---
    if st.session_state.active_tab == "Synchrotron":

        # Botón de Reset específico para esta sección
        st.sidebar.button("⟳ Reset Parameters", on_click=reset_all_parameters, use_container_width=True)

        # Distance
        dist_values_kpc, ini_dist_values_kpc = dist_axis()
        distance_kpc = st.sidebar.select_slider(
            "Distance, $d~(\\rm{kpc})$", 
            options=dist_values_kpc,
            value=ini_dist_values_kpc, # Initial value
            format_func=lambda x: f"{x:.1f}", # Para que los números se vean limpios
            key="dist_sync",
            help="Distance to the emitting source in kiloparsecs (kpc)."
        )

        # Spectral Index
        p_values, p_ini_value = p_index_axis()
        p_index = st.sidebar.select_slider(
            "Spectral Index, $p$", 
            options=p_values,
            value=p_ini_value, # Initial value
            format_func=lambda x: f"{x:.1f}", # Para que los números se vean limpios
            key="p_index_sync",
            help="Spectral index of the particle energy distribution."
        )    

        # Cutoff
        ecut_values_GeV, ini_ecut_values_GeV = e_cutoff_axis_GeV()
        e_cutoff_GeV = st.sidebar.select_slider(
            "$E_{\\rm cutoff}~(\\rm{GeV})$", 
            options=ecut_values_GeV,
            value=ini_ecut_values_GeV,
            format_func=lambda x: f"{x:.0f}", # Para que los números se vean limpios
            key="e_cutoff_sync",
            help="Cutoff energy of the particle energy distribution (maximum energy of the relativistic particles) in gigaelectronvolts (GeV)."
        )

        # Magnetic field
        values_B_uGauss, ini_value_B_uGauss = mag_field_sync_axis()
        B_uGauss = st.sidebar.select_slider(
            "Magnetic Field, $B$ $\\left(\\rm{µG}\\right)$", 
            options=values_B_uGauss,
            value=ini_value_B_uGauss,
            format_func=lambda x: f"{x:.2e}",
            key="B_sync",
            help="Magnetic field strength for synchrrotron radiation in microgauss (uG)."
        )

        st.sidebar.header("Fixed parameters")
        
        st.sidebar.caption(
            "Energy distribution of the relativistic electrons:\n "
            "- Model: Exponential cutoff power law\n "
            "- Amplitude: $10^{{36}}$ electrons $\\rm{{eV}}^{{-1}}$ @ $100~\\rm{GeV}$"
        )

        # st.sidebar.divider()
        # st.sidebar.write(f"**Current Values:**")
        # st.sidebar.caption(f"Dist: {distance_kpc:.2f} kpc | p: {p_index:.1f}")
        # st.sidebar.caption(f"E_cut: {e_cutoff_GeV:.0f} GeV | B: {B_uGauss:.1e}")


        st.subheader("Synchrotron radiation")
        e_v3, s_v3, we = get_synchrotron_sed(distance_kpc, p_index, e_cutoff_GeV, B_uGauss)
        
        fig3, ax3 = plt.subplots(figsize=(9, 5))
        ax3.loglog(e_v3, s_v3, color='tab:green', lw=2, label=f"$d = {distance_kpc:.2f}$ kpc,\n"+\
            f"$p = {p_index:.1f}$,\n"+\
            f"$E_{{\\rm cutoff}} = {e_cutoff_GeV:.0f}$ GeV,\n"+\
            f"$B = {B_uGauss:1.1f}$ µG"
        )
        ax3.set_title(f"Energy in electrons ($E > 1~\\rm{{GeV}}$): {we:1.1e} erg")
        ax3.set_xlabel("Energy, $E$ (eV)")
        ax3.set_ylabel(r"$E^2 d\phi/dE$ (erg cm$^{-2}$ s$^{-1}$)")
        ax3.grid(True, which="both", alpha=0.3)
        ax3.set_ylim(1e-13, 1e-9)
        ax3.legend(title="Synchrotron")

        # Create a secondary x-axis for frequency
        ax3_2 = ax3.twiny()
        ax3_2.set_xscale("log")
        # Convert energy to frequency using ν = E / h
        spectrum_freq_sync = (e_v3*u.eV / h).to(u.Hz)
        ax3_2.set_xlim(spectrum_freq_sync[0].value, spectrum_freq_sync[-1].value)
        ax3_2.set_xlabel("Frequency (Hz)")
        st.pyplot(fig3)

        with st.expander("Diagram"):
            # Ruta de la imagen
            img_path = "data/images/synchrotron.png"
            # Verificamos si el archivo existe para evitar errores
            if os.path.exists(img_path):
                st.image(
                    img_path, 
                    caption="Diagram of electron undergoing synchrotron radiation in a magnetic field.", 
                    width=500#use_container_width=True
                )
                st.markdown(
                    """
                    Source: C. Grupen, Astroparticle Physics, Undergraduate Texts in Physics, Springer (2020), 10.1007/978-3-030-27339-2
                    """
                )            
            else:
                st.error(f"Image not found at {img_path}")            

        st.info("""
            - Can you identify the absorbed component of the synchrotron emission?
            - What is the spectral index of the emission in the optically thin energies? What about in the optically thick?
        """)



    # --- Button 4: Inverse Compton ---
    if st.session_state.active_tab == "Inverse Compton":

        # Botón de Reset específico para esta sección
        st.sidebar.button("⟳ Reset Parameters", on_click=reset_all_parameters, use_container_width=True)

        # Distance
        dist_values_kpc, ini_dist_values_kpc = dist_axis()
        distance_kpc = st.sidebar.select_slider(
            "Distance, $d~(\\rm{kpc})$", 
            options=dist_values_kpc,
            value=ini_dist_values_kpc, # Initial value
            format_func=lambda x: f"{x:.1f}", # Para que los números se vean limpios
            key="dist_ic",
            help="Distance to the emitting source in kiloparsecs (kpc)."
        )

        # Spectral Index
        p_values, p_ini_value = p_index_axis()
        p_index = st.sidebar.select_slider(
            "Spectral Index, $p$", 
            options=p_values,
            value=p_ini_value, # Initial value
            format_func=lambda x: f"{x:.1f}", # Para que los números se vean limpios
            key="p_index_ic",
            help="Spectral index of the particle energy distribution."
        )    

        # Cutoff
        ecut_values_GeV, ini_ecut_values_GeV = e_cutoff_axis_GeV()
        e_cutoff_GeV = st.sidebar.select_slider(
            "$E_{\\rm cutoff}~(\\rm{GeV})$", 
            options=ecut_values_GeV,
            value=ini_ecut_values_GeV,
            format_func=lambda x: f"{x:.0f}", # Para que los números se vean limpios
            key="e_cutoff_ic",
            help="Cutoff energy of the particle energy distribution (maximum energy of the relativistic particles) in gigaelectronvolts (GeV)."
        )

        # Photon field
        photon_field = st.sidebar.selectbox(
            label="Photon Seed:",
            options=['CMB', 'FIR', 'NIR', 'Hot star'],
            index=0,  # Corresponde a 'CMB' (el valor por defecto)
            help="Select the type of photon field for Inverse Compton scattering"
        )

        st.sidebar.header("Fixed parameters")
        st.sidebar.caption(
            "Energy distribution of the relativistic electrons:\n "
            "- Model: Exponential cutoff power law\n "
            "- Amplitude: $10^{{36}}$ electrons $\\rm{{eV}}^{{-1}}$ @ $100~\\rm{GeV}$"
        )

        # st.sidebar.divider()
        # st.sidebar.write(f"**Current Values:**")
        # st.sidebar.caption(f"Dist: {distance_kpc:.2f} kpc | p: {p_index:.1f}")
        # st.sidebar.caption(f"E_cut: {e_cutoff_GeV:.0f} GeV | Photon field: {photon_field}")


        st.subheader("Inverse Compton scattering")
        e_v4, s_v4, we, additional_label = get_inverse_compton_sed(distance_kpc, p_index, e_cutoff_GeV, photon_field)
        
        fig4, ax4 = plt.subplots(figsize=(9, 5))
        ax4.loglog(e_v4, s_v4, color='tab:orange', lw=2, 
            label=f"$d = {distance_kpc:.2f}$ kpc,\n"+\
            f"$E_{{\\rm cutoff}} = {e_cutoff_GeV:.0f}$ GeV,\n"+\
            f"Photon Seed = {photon_field}{additional_label}"
        )

        ax4.set_title(f"Energy in electrons ($E > 1~\\rm{{GeV}}$): {we:1.1e} erg")
        ax4.set_xlabel("Energy, $E$ (eV)")
        ax4.set_ylabel(r"$E^2 d\phi/dE$ (erg cm$^{-2}$ s$^{-1}$)")
        ax4.grid(True, which="both", alpha=0.3)
        ax4.set_ylim(1e-13, 1e-9)
        ax4.legend(title="Inverse Compton")
        st.pyplot(fig4)

        with st.expander("Diagram"):
            # Ruta de la imagen
            img_path = "data/images/ic.png"
            # Verificamos si el archivo existe para evitar errores
            if os.path.exists(img_path):
                st.image(
                    img_path, 
                    caption="Diagram of an electron scattering off a low-energy photon and transfering energy to it.", 
                    width=500#use_container_width=True
                )
                st.markdown(
                    """
                    Source: C. Grupen, Astroparticle Physics, Undergraduate Texts in Physics, Springer (2020), 10.1007/978-3-030-27339-2
                    """
                )
            else:
                st.error(f"Image not found at {img_path}")            

        with st.expander("Description of the available photon seeds"):
            st.markdown(
                """
                - CMB: Simulate the cosmic microwave background, i.e., isotropic photons with $T=2.72~ \\rm K$ and photon energy density $U_{\rm ph}=0.261~ \\rm eV/cm^{3}$

                - FIR: Simulate the far-infrared dust emission, i.e., isotropic photons with $T=30~ \\rm K$ and photon energy density $U_{\\rm ph}=0.5~ \\rm eV/cm^{3}$

                - NIR: Simulate the near-infrared stellar emission, i.e., isotropic photons with $T=3,000~ \\rm K$ and photon energy density $U_{\\rm ph}=1~ \\rm eV/cm^{3}$

                - Hot star: Simulate the photons from a hot star (O-B) next to a compact object in a binary system. Specific details: photons with $T=30,000~ \\rm K$ and photon energy density at $5000~R_\\odot$, i.e., $U_{\\rm ph}=6.1\\times 10^{-3} ~ \\rm erg/cm^{3}$ $\\left(=3.81\\times 10^{9} ~ \\rm eV/cm^{3}\\right)$
                """
            )        


    # --- Button 5: Synchrotron-self Compton ---
    if st.session_state.active_tab == "Synchrotron-self Compton":

        # Botón de Reset específico para esta sección
        st.sidebar.button("⟳ Reset Parameters", on_click=reset_all_parameters, use_container_width=True)

        # Distance
        dist_values_kpc, ini_dist_values_kpc = dist_axis()
        distance_kpc = st.sidebar.select_slider(
            "Distance, $d~(\\rm{kpc})$", 
            options=dist_values_kpc,
            value=ini_dist_values_kpc, # Initial value
            format_func=lambda x: f"{x:.1f}", # Para que los números se vean limpios
            key="dist_ssc",
            help="Distance to the emitting source in kiloparsecs (kpc)."
        )

        # Spectral Index
        p_values, p_ini_value = p_index_axis()
        p_index = st.sidebar.select_slider(
            "Spectral Index, $p$", 
            options=p_values,
            value=p_ini_value, # Initial value
            format_func=lambda x: f"{x:.1f}", # Para que los números se vean limpios
            key="p_index_ssc",
            help="Spectral index of the particle energy distribution."
        )    

        # Cutoff
        ecut_values_GeV, ini_ecut_values_GeV = e_cutoff_axis_GeV()
        e_cutoff_GeV = st.sidebar.select_slider(
            "$E_{\\rm cutoff}~(\\rm{GeV})$", 
            options=ecut_values_GeV,
            value=ini_ecut_values_GeV,
            format_func=lambda x: f"{x:.0f}", # Para que los números se vean limpios
            key="e_cutoff_ssc",
            help="Cutoff energy of the particle energy distribution (maximum energy of the relativistic particles) in gigaelectronvolts (GeV)."
        )

        # Magnetic field
        values_B_uGauss, ini_value_B_uGauss = mag_field_ssc_axis()
        B_uGauss = st.sidebar.select_slider(
            "Magnetic Field, $B$ $\\left(\\rm{µG}\\right)$", 
            options=values_B_uGauss,
            value=ini_value_B_uGauss,
            format_func=lambda x: f"{x:.2e}",
            key="B_ssc",
            help="Magnetic field strength for synchrrotron radiation in microgauss (uG)."
        )

        st.sidebar.header("Fixed parameters")
        st.sidebar.caption(
            "Energy distribution of the relativistic electrons:\n "
            "- Model: Exponential cutoff power law\n "
            "- Amplitude: $10^{{36}}$ electrons $\\rm{{eV}}^{{-1}}$ @ $100~\\rm{GeV}$"
        )

        # st.sidebar.divider()
        # st.sidebar.write(f"**Current Values:**")
        # st.sidebar.caption(f"Dist: {distance_kpc:.2f} kpc | p: {p_index:.1f}")
        # st.sidebar.caption(f"E_cut: {e_cutoff_GeV:.0f} GeV | n: {dens_val:.1e}")


        st.subheader("Synchrotron-self Compton")
        e_v5, s_v5_sync, s_v5_ic, we_v5 = get_synchrotron_self_compton_sed(distance_kpc, p_index, e_cutoff_GeV, B_uGauss)
        
        fig5, ax5 = plt.subplots(figsize=(9, 5))
        ax5.loglog(e_v5, s_v5_sync, color='tab:green', lw=2, label="Synchrotron")
        ax5.loglog(e_v5, s_v5_ic, color='tab:orange', lw=2, label="Inverse Compton")
        ax5.set_title(f"Energy in electrons ($E > 1~\\rm{{GeV}}$): {we_v5:1.1e} erg")
        ax5.set_xlabel("Energy, $E$ (eV)")
        ax5.set_ylabel(r"$E^2 d\phi/dE$ (erg cm$^{-2}$ s$^{-1}$)")
        ax5.grid(True, which="both", alpha=0.3)
        ax5.set_ylim(1e-13, 1e-6)
        ax5.legend(
            title=f"$d = {distance_kpc:.2f}$ kpc,\n"
            f"$p = {p_index:.1f}$,\n"
            f"$E_{{\\rm cutoff}} = {e_cutoff_GeV:.0f}$ GeV,\n"
            f"$B = {B_uGauss:.1f}$ µG",
            loc="best"
        )

        # Create a secondary x-axis for frequency
        ax5_2 = ax5.twiny()
        ax5_2.set_xscale("log")
        # Convert energy to frequency using ν = E / h
        spectrum_freq_SSC = (e_v5*u.eV / h).to(u.Hz)
        ax5_2.set_xlim(spectrum_freq_SSC[0].value, spectrum_freq_SSC[-1].value)
        ax5_2.set_xlabel("Frequency (Hz)")

        st.pyplot(fig5)

    
else:

    st.sidebar.header("Free parameters to change")

    # Botón de Reset específico para esta sección
    st.sidebar.button("⟳ Reset Parameters", on_click=reset_all_parameters, use_container_width=True)

    # Distance
    dist_values_kpc, ini_dist_values_kpc = dist_axis()
    distance_kpc = st.sidebar.select_slider(
        "Distance, $d~(\\rm{kpc})$", 
        options=dist_values_kpc,
        value=ini_dist_values_kpc, # Initial value
        format_func=lambda x: f"{x:.1f}", # Para que los números se vean limpios
        key="dist_crab",
        help="Distance to the emitting source in kiloparsecs (kpc)."
    )

    # Cutoff
    ecut_values_TeV, ini_ecut_values_TeV = e_cutoff_axis_TeV()
    e_cutoff_TeV = st.sidebar.select_slider(
        "$E_{\\rm cutoff}~(\\rm{TeV})$", 
        options=ecut_values_TeV,
        value=ini_ecut_values_TeV,
        format_func=lambda x: f"{x:.0f}", # Para que los números se vean limpios
        key="e_cutoff_crab",
        help="Cutoff energy of the particle energy distribution (maximum energy of the relativistic particles) in teraelectronvolts (TeV)."
    )

    # Magnetic field
    values_B_uGauss, ini_value_B_uGauss = mag_field_ssc_axis()
    B_uGauss = st.sidebar.select_slider(
        "Magnetic Field, $B$ $\\left(\\rm{µG}\\right)$", 
        options=values_B_uGauss,
        value=ini_value_B_uGauss,
        format_func=lambda x: f"{x:.2e}",
        key="B_crab",
        help="Magnetic field strength for synchrrotron radiation in microgauss (uG)."
    )

    st.title("Crab Nebula Multiwavelength fit")
    st.markdown("Let's try to fit by hand the Crab Nebula spectrum across the electromagnetic spectrum using a Self-Synchrotron Compton model. Can you obtain a good fit?")

    st.sidebar.header("Fixed parameters")
    st.sidebar.caption(
        "Energy distribution of the relativistic electrons:\n "
        "- Model: Broken power law\n "
        "- Amplitude: $3.699 \\times 10^{{36}}$ electrons $\\rm{{eV}}^{{-1}}$ @ $1~\\rm{TeV}$\n "
        "- Energy break: $0.265~\\rm{TeV}$\n "
        "- Spectral index before the break: $1.5$\n "
        "- Spectral index after the break: $3.233$\n "
        "- Break smoothness factor: $2.0$"
    )

    crab_data = load_crab_data()
    spectrum_energy, s_syn, s_ic = get_Crab_synch_ic_sed(distance_kpc, e_cutoff_TeV, B_uGauss)

    # Convert energy to frequency using ν = E / h (Planck relation)
    spectrum_freq = (spectrum_energy*u.eV / h).to(u.Hz)

    # Convert frequency to wavelength using λ = c / ν
    spectrum_wavelength = (c / spectrum_freq).to(u.m)

    fig, ax1 = plt.subplots(figsize=(10, 7))
    
    # Graficar datos experimentales si existen
    if crab_data:
        naima.plot_data(crab_data, e_unit=u.eV, figure=fig)
        
    # Graficar Modelos
    ax1.loglog(spectrum_energy, s_syn, label="Synchrotron", color='darkorange', lw=2)
    ax1.loglog(spectrum_energy, s_ic, label="Inverse Compton", color='royalblue', lw=2)

    # Configuración de ejes
    ax1.set_xlabel("Energy, $E$ (eV)")
    ax1.set_ylabel("$E^2$ d$\\phi$/d$E$ (erg cm$^{-2}$ s$^{-1}$)")
    ax1.set_ylim(1e-15, 1e-7)
    ax1.legend(
        title=f"Distance = {distance_kpc:.2f} kpc,\n"
        f"$E_{{\\rm cutoff}} = {e_cutoff_TeV:.0f}$ TeV,\n"
        f"$B = {B_uGauss:.1f}$ µG",
        loc="best"
    )
    ax1.grid(True, which="both", alpha=0.2)

    # Create a secondary x-axis for frequency (in Hz)
    ax2 = ax1.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(spectrum_freq[0].value, spectrum_freq[-1].value)
    ax2.set_xlabel("Frequency (Hz)")

    # Create a third x-axis for wavelength (in meters)
    ax3 = ax1.twiny()
    ax3.set_xscale("log")
    ax3.set_xlim(spectrum_wavelength[0].value, spectrum_wavelength[-1].value)
    ax3.spines['top'].set_position(('outward', 40))
    ax3.set_xlabel("Wavelength (m)")

    st.pyplot(fig)

    st.info("The Inverse Compton model includes CMB, FIR, NIR, and Self-Synchrotron Compton (SSC) seed fields.")


