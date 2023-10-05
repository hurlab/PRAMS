# PRAMS

This repo contains scripts used in the paper [Machine Learning-Based Predictive Modeling of Postpartum Depression](https://www.mdpi.com/2077-0383/9/9/2899). We trained machine learning models to predict postpartum depression, and further analysed our models to find features that are important in the development of this phenotype. 

#### software
1. create a conda environment: `conda env create -f software/environment.yaml`
2. open the `r` console
3. install additional packages: `Rscript software/packages.R`

### scripts and usage

[modules](scripts/modules/): contains helper functions used in this project; if you intend to reproduce these results, you don't need to run this script.

0. helpers.R

[preprocess](scripts/preprocess): contains scripts used to preprocess the data

1. preprocessing.R
2. balancing_dataset.R
3. collect_and_split_data.R

[train](scripts/train):

4. train_set_one.R        
5. train_set_two.R
6. train_set_three.R      

[analyse](scripts/analyse):

7. analyse.R

### Notes
- there is a `feature_details.csv` file that I need to find