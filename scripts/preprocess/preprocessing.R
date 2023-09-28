# Author: Temi
# Date:
# Description: This script was used to pre-process the PRAMS data

library(tidyverse)
library(caret)
library(data.table)
library(plyr)

project_dir <- "/Users/temi/Dropbox/UND_Temidayo-Adeluwa/prams/PRAMS"
setwd(project_dir)

# load the data
all_prams <- read.delim(file.path(project_dir, 'data', 'PRAMS2012-2013.txt'), header=T, 
                          stringsAsFactors = F, sep='\t', skip=0)

# keep the ID column and remove the STATE and BATCH columns
# ensure all other columns are `numeric`
ID <- all_prams$ID
all_prams$ID <- NULL
all_prams$STATE <- NULL
all_prams$BATCH <- NULL

# Coerce the values into numeric and map the binary outcome to healthy or depressed
all_prams <- as.data.frame(cbind(ID, as.data.frame(sapply(all_prams, as.numeric))))
all_prams$PP_DEPRESS <- as.factor(mapvalues(all_prams$PP_DEPRESS, from=c(1, 2), to=c('Healthy', 'Depressed')))

# after coercing, some columns have nA values, 
# remove columns with at least 10,000 missing values
na.cols <- apply(all_prams, 2, function(x){
    sum(is.na(x))
})
na.cols <- names(na.cols[na.cols > 10000])
all_prams <- all_prams[, !names(all_prams) %in% na.cols]

# select complete rows and ensure the binary outcomes are mapped
comp_prams <- as.data.frame(all_prams[complete.cases(all_prams), ])
target <- as.factor(mapvalues(comp_prams$PP_DEPRESS, from=c(1, 2), to=c('Healthy', 'Depressed')))
comp_prams$PP_DEPRESS <- NULL

ID <- comp_prams$ID
comp_prams$ID <- NULL

# Some features have little to no variance in them
# remove near zero variances using a method from caret
nzv <- nearZeroVar(comp_prams[, !names(comp_prams) %in% c("PP_DEPRESS")], names=T, saveMetrics = T)
nzv <- row.names(nzv[nzv$nzv,])
comp_prams <- comp_prams[, !names(comp_prams) %in% c(nzv)]

# apply(comp_prams, 2, function(x){
#   typeof(x)
# })

# remove highly correlated variables too 
# here I used a cutoff of 0.95
prams.corr <- cor(comp_prams[, !names(comp_prams) %in% c("PP_DEPRESS", 'ID')])
no.corr <- sum(abs(prams.corr[upper.tri(prams.corr)]) >= 0.99)
highly.cor <- findCorrelation(prams.corr, cutoff = 0.95, names=T)
comp_prams <- comp_prams[, !names(comp_prams) %in% highly.cor]

# add back the ID and target columns and save the pre-processed data
complete_prams_with_IDs <- as.data.frame(cbind(ID, PP_DEPRESS=target, comp_prams))
write.csv(complete_prams_with_IDs, file='./data/complete_prams_with_IDs.csv', row.names = T)

#summary(cor(comp_prams[, !names(comp_prams) %in% c("PP_DEPRESS")])[upper.tri(cor(comp_prams[, !names(comp_prams) %in% c("PP_DEPRESS")]))])


sessionInfo()