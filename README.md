# Multiple Melanoma
Daan Knoors - 11/11/22

## Objective
The purpose of this data challenge is to develop a machine learning model for predicting the 
risk of fast dying or relapsing of newly diagnosed MM patients.

## Data
Three data files:
- **expression**: gene expression data: https://www.synapse.org/#!Synapse:syn10573789
- **clinical**: clinical data + follow-up for newly diagnosed MM patients: https://www.synapse.org/#!Synapse:syn9926878
- **dictionary**: explanation of the clinical and label annotations: https://www.synapse.org/#!Synapse:syn9744732

Other datasets and info can be found on the Multiple Myeloma Dream Challenge Site: https://www.synapse.org/#!Synapse:syn6187098/wiki/401884


## Installation
Create environment using preferred environment manager. For example using conda:
> conda env create -f environment.yml
> conda activate mm_highrisk

After activating the environment you can open jupyter notebooks using jupyter lab by running the following in terminal:
> jupyter lab

## Project structure
mm_high_risk
|-- data/ 												: place data files here
|
|-- docs/												: documentation of project
|	|-- references/										: literature references
|
|-- models/											    : trained classifiers by name and date trained
|
|-- notebooks/											: jupyter notebooks
|	|-- 000_exploratory_data_analysis.ipynb				: EDA of provided data files
|	|-- 001_train_model.ipynb							: train classifiers and evaluate performance
|
|-- src/											    : python modules
|	|-- __init__.py                                     
|	|-- config.py                                       : configuration settings and variable constants
|	|-- gene.py                                       	: extract gene descriptions
|	|-- model.py                                        : train classifier
|	|-- preprocess.py									: preprocess data prior to modeling
|	|-- utils.py                                        : utility functions
|	|-- visual.py	                                    : visualizations
|
|-- environment.yml										: python environment
|-- README.md