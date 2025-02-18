## naima_handson

This repository has been prepared to give a hands-on session at the high-energy astrophysics subject of the Astrophysics, Particle Physics and Cosmology master's degree at the Universitat de Barcelona. 
This hands-on session is about the computation of non-thermal radiation and MCMC fitting with [naima](https://naima.readthedocs.io/en/latest/index.html). 

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
conda activate naima
pip install naima
```

5. Enjoy
Open your preferred code interface to execute the code.

## Repository structure

The sctructure of the repository is divided into the "data" and "scripts" folder.
- scripts: store Jupyter Notebook scripts to compute the non-thermal radiation and MCMC fitting.
- data: store data files (.dat or .ecsv) that contain the spectral energy distribution data points that will be used to run the scripts in the scripts folder.
