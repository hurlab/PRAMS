# Author: Temi
# Date:
# Description: This script was used split the pre-processed PRAMS data into 3 data sets
# depending on what using smote returns; smote was used to balance the data

# Notes: caret was used to pre-process this data

setwd('/Users/temi/Dropbox/UND_Temidayo-Adeluwa/prams/PRAMS/PRAMS/scripts/')
data_dir <- '/Users/temi/Dropbox/UND_Temidayo-Adeluwa/prams/PRAMS/data'
source('./modules/helpers.R') # some helper functions

set.seed(1)

library(pROC)
library(caret)
library(tidyverse)

prams_dt <- read.csv(file.path(data_dir, 'complete_prams_with_IDs.csv'), header=T, stringsAsFactors = T)
use_ids <- read.csv(file.path(data_dir, 'each_unique_IDs_non.weight.csv'), header=T, stringsAsFactors = T) # these are features we used after ...

take_out <- c('STRS_TT3', 'MOM_BMIG', 'MOM_BMIG_QX_REV', 'INFQ_AGE', 'PNC_WKS', 'YY4_LMP', 'MH_PPINT', 'MH_PPDPR')
weighting_variables <- c('INQX', 'NEST_YR', 'SAMCNT', 'STRATUMC', 'SUD_NEST', 'TOTCNT', 'WTANAL', 'WTONE', 'WTTHREE', 'WTTWO', 'TYPE')
prams_dt <- prams_dt[, !names(prams_dt) %in% c(take_out, weighting_variables)]
ids <- prams_dt$ID
prams_dt <- prams_dt[!names(prams_dt) %in% c('X', 'ID')]

enc_prams <- createDummyFeatures(prams_dt, 'PP_DEPRESS') # using a helper function
enc_prams <- as.data.frame(cbind(ID=ids, enc_prams))
enc_dep <- dplyr::filter(enc_prams, PP_DEPRESS=='Depressed')

# === depending on the feature set, I prepare 3 data sets to train on ===
df.one <- enc_prams %>%
    filter(ID %in% use_ids$first_ID)
df.one <- as.data.frame(rbind(df.one, enc_dep))
df.one <- df.one[sample(nrow(df.one)), ]

df.two <- enc_prams %>%
    filter(ID %in% use_ids$second_ID)
df.two <- as.data.frame(rbind(df.two, enc_dep))
df.two <- df.two[sample(nrow(df.two)), ]

df.three <- enc_prams %>%
    filter(ID %in% use_ids$third_ID)
df.three <- as.data.frame(rbind(df.three, enc_dep))
df.three <- df.three[sample(nrow(df.three)), ]

# df.one <- createDummyFeatures(df.one, 'PP_DEPRESS')
# df.two <- createDummyFeatures(df.two, 'PP_DEPRESS')
# df.three <- createDummyFeatures(df.three, 'PP_DEPRESS')

write.csv(df.one, file=file.path(data_dir, 'df_one.csv'), row.names=F)
write.csv(df.two, file=file.path(data_dir, 'df_two.csv'), row.names=F)
write.csv(df.three, file=file.path(data_dir, 'df_three.csv'), row.names=F)


sessionInfo()