## naima_handson

This repository has been prepared to give a hands-on session at the high-energy astrophysics subject of the Astrophysics, Particle Physics and Cosmology master's degree at the Universitat de Barcelona. 
This hands-on session is about the computation of non-thermal radiation and MCMC fitting with [naima](https://naima.readthedocs.io/en/latest/index.html). 

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://naima-handson.streamlit.app/)

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
conda env create -f environment.yml
conda activate naima_handson
pip install naima
```

5. Enjoy

The scripts in this repository have been tested using jupyter notebook. I recommend executing the scripts with it (jupyter notebook is already installed in the environment). However, you can use your preferred code interface under your responsibility.

To open jupyter notebook from the terminal, use the following:
```
conda activate naima_handson
jupyter notebook
```

## Repository structure

The sctructure of the repository is divided into the "data" and "scripts" folders.
- scripts: store Jupyter Notebook scripts to compute the non-thermal radiation and MCMC fitting.
- data: store data files (.ecsv) that contain the spectral energy distribution data points that will be used to run the scripts in the scripts folder.
