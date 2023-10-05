# Author: Temi
# Date:
# Description: This script contains functions used to collect metrics and create plots from '../objects'

setwd('/Users/temi/Dropbox/UND_Temidayo-Adeluwa/prams/PRAMS/PRAMS/scripts/')
data_dir <- '/Users/temi/Dropbox/UND_Temidayo-Adeluwa/prams/PRAMS/data'
objects_dir <- '/Users/temi/Dropbox/UND_Temidayo-Adeluwa/prams/PRAMS/objects'
results_dir <- '/Users/temi/Dropbox/UND_Temidayo-Adeluwa/prams/PRAMS/results'
if(!dir.exists(results_dir){
    dir.create(results_dir)
})

source('./modules/helpers.R') # some helper functions

library(pROC)
library(caret)
library(tidyverse)
library(gbm)

# load in the data
# one.relief.list
load(file.path(objects_dir, 'set_one_fits.RData'))
# two.relief.list
load(file.path(objects_dir, 'set_two_fits.RData'))
# three.relief.list
load(file.path(objects_dir, 'set_three_fits.RData'))

# ???
feature_desc <- read.csv(file.path(data_dir, 'feature_details.csv'), header=T)

# 
set1.list <- list(`LR_Set1`=set_one.list$LR_Set1,
                  `RF_Set1`=set_one.list$RF_Set1,
                  `RPART_Set1`=set_one.list$RPART_Set1,
                  `NNET_Set1`=set_one.list$NNET_Set1,
                  `NB_Set1`=set_one.list$NB_Set1,
                  `GBM_Set1`=set_one.list$GBM_Set1,
                  `AdaBoost_Set1`=set_one.list$AdaBoost_Set1,
                  `kNN_Set1`=set_one.list$kNN_Set1,
                  `SVM_Set1`=set_one.list$SVM_Set1)

set2.list <- list(`LR_Set2`=set_two.list$LR_Set2,
                  `RF_Set2`=set_two.list$RF_Set2,
                  `RPART_Set2`=set_two.list$RPART_Set2,
                  `NNET_Set2`=set_two.list$NNET_Set2,
                  `NB_Set2`=set_two.list$NB_Set2,
                  `GBM_Set2`=set_two.list$GBM_Set2,
                  `AdaBoost_Set2`=set_two.list$AdaBoost_Set2,
                  `kNN_Set2`=set_two.list$kNN_Set2,
                  `SVM_Set2`=set_two.list$SVM_Set2)

set3.list <- list(`LR_Set3`=set_three.list$LR_Set3,
                  `RF_Set3`=set_three.list$RF_Set3,
                  `RPART_Set3`=set_three.list$RPART_Set3,
                  `NNET_Set3`=set_three.list$NNET_Set3,
                  `NB_Set3`=set_three.list$NB_Set3,
                  `GBM_Set3`=set_three.list$GBM_Set3,
                  `AdaBoost_Set3`=set_three.list$AdaBoost_Set3,
                  `kNN_Set3`=set_three.list$kNN_Set3,
                  `SVM_Set3`=set_three.list$SVM_Set3)

all.list <- c(set1.list, set2.list, set3.list)

n.models <- c('LR', 'RF', 'RPART', 'NNET', 'NB', 
              'GBM', 'AdaBoost', 'kNN', 'SVM')

set1.top.four <- list(`RF_Set1`=set_one.list$RF_Set1,
                  `GBM_Set1`=set_one.list$GBM_Set1,
                  `AdaBoost_Set1`=set_one.list$AdaBoost_Set1,
                  `SVM_Set1`=set_one.list$SVM_Set1)

set2.top.four <- list(`RF_Set2`=set_two.list$RF_Set2,
                      `GBM_Set2`=set_two.list$GBM_Set2,
                      `AdaBoost_Set2`=set_two.list$AdaBoost_Set2,
                      `SVM_Set2`=set_two.list$SVM_Set2)

set1.freq.table <- freqTableFromDataframe(analyseTopFeatures(set1.top.four, 20))
set2.freq.table <- freqTableFromDataframe(analyseTopFeatures(set2.top.four, 20))

write.csv(rearrangeFreqTable(set1.freq.table), file=file.path(results_dir, 'freq_table_set1.csv'), row.names = F)
write.csv(rearrangeFreqTable(set2.freq.table), file=file.path(results_dir, 'freq_table_set2.csv'), row.names = F)

columns <- c('Model', 'ROC', 'Sensitivity', 'Specificity', 'Accuracy')

set1.metrics <- gatherMetrics(set1.list) %>%
    dplyr::select(columns) %>%
    tidyr::separate(Model, into=c('keep', 'discard'), sep='_') %>% 
    dplyr::select(-c('discard')) %>% 
    dplyr::rename(Model=keep)

set2.metrics <- gatherMetrics(set2.list) %>%
    dplyr::select(columns) %>%
    tidyr::separate(Model, into=c('keep', 'discard'), sep='_') %>% 
    dplyr::select(-c('discard')) %>% 
    dplyr::rename(Model=keep)

set3.metrics <- gatherMetrics(set3.list) %>%
    dplyr::select(columns) %>%
    tidyr::separate(Model, into=c('keep', 'discard'), sep='_') %>% 
    dplyr::select(-c('discard')) %>% 
    dplyr::rename(Model=keep)

write.csv(set1.metrics, file=file.path(resultsdir, 'set1_metrics.csv'), row.names = F)
write.csv(set2.metrics, file=file.path(resultsdir, 'set2_metrics.csv'), row.names = F)
write.csv(set3.metrics, file=file.path(resultsdir, 'set3_metrics.csv'), row.names = F)

#
temp.roc <- gatherPlotROC(set1.list, case='Depressed', control='Healthy',
                                random.select = 'min')
pdf(file=file.path(results_dir, 'roc_plot_set1.pdf'), width=11, height=9)
temp.roc
dev.off()

temp.roc <- gatherPlotROC(set2.list, case='Depressed', control='Healthy',
                                random.select = 'min')
pdf(file=file.path(results_dir, 'roc_plot_set2.pdf'), width=11, height=9)
temp.roc
dev.off()

temp.roc <- gatherPlotROC(set3.list, case='Depressed', control='Healthy',
                                random.select = 'min')
pdf(file=file.path(results_dir, 'roc_plot_set3.pdf'), width=11, height=9)
temp.roc
dev.off()

# collect average metrics across all the models ===========================
# works on all.list
all.list <- c(set1.list, set2.list, set3.list)

sort.all.list <- lapply(n.models, function(r){
    all.list[grepl(r, names(all.list))]
})
names(sort.all.list) <- n.models

# gather.metrics(sort.all.list$LR)
all.metrics <- lapply(sort.all.list, gatherMetrics)

sum.all.metrics <- lapply(all.metrics, function(e){
    e %>%
        summarise(mean.ROC=mean(ROC), mean.Sens=mean(Sensitivity), 
                  mean.Spec=mean(Specificity),mean.Acc=mean(`Balanced Accuracy`), 
                  mean.Prec=mean(Precision), mean.F1=mean(F1)) 
})

sum.all.metrics.df <- do.call(rbind, sum.all.metrics) %>% 
    round(3) %>%
    rownames_to_column(var='Model') %>%
    arrange(desc(mean.ROC)) %>%
    dplyr::rename(AUC=mean.ROC, Sensitivity=mean.Sens, Specificity=mean.Spec,
                  Accuracy=mean.Acc, Precision=mean.Prec, F1=mean.F1)

write.csv(sum.all.metrics.df, file=file.path(results_dir, 'average_metrics.csv'), row.names = F)
