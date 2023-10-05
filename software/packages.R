#!/usr/bin/env Rscript

# Author: Temi
# Date:
# Description: script to install R packages needed


# packages

devtools::install_github("RomeroBarata/bimba")
remotes::install_github("cran/DMwR")
install.packages(c('doParallel', 'gbm', 'DMwR', 'kernlab', 
    'naivebayes', 'adabag', 'mctest', 'corpcor', 
    'ROSE', 'FSelector'), dependencies=T)