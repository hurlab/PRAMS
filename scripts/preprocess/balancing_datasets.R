# Author: Temi
# Date:
# Description: This script was used to prepare balanced datasets

project_dir <- "/Users/temi/Dropbox/UND_Temidayo-Adeluwa/prams/PRAMS"
setwd(project_dir)

library(caret)
library(tidyverse)
# install bimba ==> devtools::install_github("RomeroBarata/bimba")
library(bimba)

prams <- read.csv('./data/complete_prams_with_IDs.csv', header=T, stringsAsFactors = F)

# still need to figure out why I removed these features
prams <- prams[, !names(prams) %in% c("MH_PPINT", "MH_PPDPR", 'INQX', 'NEST_YR', 'SAMCNT', 'STRATUMC', 'SUD_NEST', 'TOTCNT', 'WTANAL', 'WTONE', 'WTTHREE', 'WTTWO', 'TYPE')]

#================sub-equal dataset===============
dep.class <- subset(prams, PP_DEPRESS=='Depressed')
hea.class <- subset(prams, PP_DEPRESS=='Healthy')

# set seed for reproducibility
set.seed(1)

hea.class.first <- sample_n(hea.class, 2*nrow(dep.class))
hea.class.second <- sample_n(subset(hea.class, !ID %in% hea.class.first$ID), 2*nrow(dep.class))
hea.class.third <- sample_n(subset(hea.class, !ID %in% hea.class.first$ID & !ID %in% hea.class.second$ID), 2*nrow(dep.class))

# save the column IDs
each.IDs <- data.frame(first_ID=hea.class.first$ID, 
                       second_ID=hea.class.second$ID,
                       third_ID=hea.class.third$ID)
write.csv(each.IDs, file='./data/each_unique_IDs_non.weight.csv')


datasets.list <- list(first_dataset=hea.class.first,
                      second_dataset=hea.class.second,
                      third_dataset=hea.class.third)

bound.datasets <- lapply(datasets.list, function(x){
    set.seed(1)
    rbind(x, dep.class)[sample(nrow(rbind(x, dep.class))), ]
})

# preprocessed dataset; use Yeo Johnson to normalize the datasets
# while doing that, remember to make the X column the rownames
ent.dataset <- list()
for (i in 1:length(bound.datasets)) {
    row.names(bound.datasets[[i]]) <- bound.datasets[[i]]$ID
    bound.datasets[[i]]$ID <- NULL
    bound.datasets[[i]]$X <- NULL
    cs <- preProcess(as.data.frame(bound.datasets[[i]]), method=c('center', 'scale', 'YeoJohnson'))
    temp.prams <- predict(cs, as.data.frame(bound.datasets[[i]]))
    ent.dataset[[names(bound.datasets)[i]]] <- as.data.frame(cbind(temp.prams[, !names(temp.prams) %in% 'PP_DEPRESS'], PP_DEPRESS=as.factor(temp.prams$PP_DEPRESS)))
}

# smote the datasets using BIMBA.
# Bimba ensure equal distributions of the classes
smoted.dataset <- list()
for (i in 1:length(ent.dataset)){
    set.seed(1)
    smoted.dataset[[names(ent.dataset)[i]]] <- bimba::SMOTE(ent.dataset[[i]], perc_min = 50, k = 5)
    print(table(smoted.dataset[[i]]$PP_DEPRESS))
}

for(i in 1:length(smoted.dataset)){
    write.csv(smoted.dataset[[i]], file=paste('./data/non_weight_sub-balanced_', names(smoted.dataset)[i], '.csv', sep=''), row.names = F)
}

#save.image('./objects/cleaning_objects.RData')

sessionInfo()