# naima_handson

This repository has been prepared to give a hands-on session on the high-energy astrophysics subject of the Astrophysics, Particle Physics and Cosmology master's degree at the Universitat de Barcelona. 
This hands-on session is about the computation of non-thermal radiation and MCMC fitting with [naima](https://naima.readthedocs.io/en/latest/index.html). 

## Repository Structure

- `data/`: Spectral energy distribution (SED) data files (.ecsv) and images used in Jupyter Notebooks.
- `notebooks/`: Jupyter Notebooks about non-thermal radiative processes and RSOph fitting.
- `results/`: Output files from fitting procedures.
- `app_rad_processes.py`: Source code for the Streamlit application about non-thermal radiative processes.

## Features

- **Non-thermal Radiation Models**: Interactive interface to compute the SED of pion decay, bremsstrahlung, synchrotron, inverse Compton, and synchrotron-self Compton radiative processes from a population of relativistic particles.
- **MCMC Fitting**: Step-by-step guide for fitting a radiative model to observational gamma-ray data from the nova RS Ophiuchi.

## Streamlit App

You can execute the radiative process Jupyter Notebook (`notebooks/radiative_process.ipynb`) as an app through Streamlit.

Note: The Jupyter Notebook `notebooks/fit_RSOph.ipynb` is not included in the Streamlit app. You can find it in the `notebooks` folder.

### Online Access
If the app is active, you can access it here:
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://naimahandson-lrfbrj5pcxxcj3zt7ex3eu.streamlit.app/)

### Local Execution
If the online app does not work, you can execute it locally. First, you must follow the installation instructions below. Then, execute the following command:

```bash
conda activate naima_handson
streamlit run app_rad_processes.py
```

## Installation

1. **Prerequisites**: Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or [Anaconda](https://www.anaconda.com/distribution/).

2. **Preparation**

Open a terminal. Make sure you have conda activated. If not, use the following: 
    ```bash
    conda activate
    ```

3. **Clone the repository**
   ```bash
   git clone https://github.com/aaguasca/naima_handson.git
   cd naima_handson
   ```
Or download it without git as follows: 

- On the main page of the repository, in the top right of the webpage, press CODE (green button) and click the "Download ZIP" option at the bottom of the pop-up options. 

- Move the zip file to the desired location and decompress it. 

- Open a terminal and `cd` to the naima_handson directory. 

4. **Environment Setup**:
Create and activate the environment using the provided configuration file:
    ```bash
    conda env create -f environment.yml.bak
    conda activate naima_handson
    ```

## Usage

The scripts in this repository have been tested using Jupyter Notebook. I recommend executing the scripts with it (Jupyter Notebook is already installed in the environment). However, you can use your preferred code interface under your responsibility.

1. **Activate Environment**:
   ```bash
   conda activate naima_handson
   ```

2. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```

3. **Open the Jupyter Notebooks**:
   - `notebooks/radiative_process.ipynb`: Explore non-thermal radiation models.
   - `notebooks/fit_RSOph.ipynb`: MCMC fitting application for RS Ophiuchi.

## Citation

If you use `naima` in your research, please follow the steps in [https://github.com/zblz/naima?tab=readme-ov-file#attribution](https://github.com/zblz/naima?tab=readme-ov-file#attribution) to cite it.