# naima_handson

This repository has been prepared to give a hands-on session on the high-energy astrophysics subject of the Astrophysics, Particle Physics and Cosmology master's degree at the Universitat de Barcelona. 
This hands-on session is about the computation of non-thermal radiation and MCMC fitting with [naima](https://naima.readthedocs.io/en/latest/index.html). 

## Repository structure

The structure of the repository is divided into the "data", "notebooks" and "results" folders.
- data: store 1) data files (.ecsv) that contain the spectral energy distribution data points that will be used to run the scripts in the notebooks folder and 2) images displayed in the notebooks.
- notebooks: store Jupyter Notebook scripts to compute the non-thermal radiation and MCMC fitting.
- results: store the results of the MCMC fitting.


## Streamlit app

You can execute the radiative process Jupyter notebook (notebooks/radiative_process.ipynb) as an app through Streamlit. You just need to press the button below (if the app is running):
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://naimahandson-dpjzkrkdynzmd5tid7chkb.streamlit.app/)

If it does not work, you can execute it locally. First, you must follow the installation instructions below. Then, you can execute the following command:

```
conda activate naima_handson
streamlit run app_rad_processes.py
```

Note: The notebook fit_RSOph.ipynb is not included in the Streamlit app. You can find it in the notebooks folder.

## Install

1. You will need to install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or [anaconda](https://www.anaconda.com/distribution/#download-section) first. 

2. Preparation

Open a terminal. Make sure you have conda activated. If not, use the following: 
```
conda activate
```

3. Clone the repository
```
git clone https://github.com/aaguasca/naima_handson.git
cd naima_handson
```
Or download it without git as follows: 

- On the main page of the repository, in the top right of the webpage, press CODE (green button) and click the "Download ZIP" option at the bottom of the pop-up options. 

- Move the zip file to the desired location and decompress it. 

- Open a terminal and `cd` to the naima_handson directory. 

4. Create and activate the naima environment. Use the following:
```
conda env create -f environment.yml.bak
conda activate naima_handson
```

## How to execute the code

The scripts in this repository have been tested using jupyter notebook. I recommend executing the scripts with it (jupyter notebook is already installed in the environment). However, you can use your preferred code interface under your responsibility.

To open jupyter notebook from the terminal, use the following:
```
conda activate naima_handson
jupyter notebook
```
