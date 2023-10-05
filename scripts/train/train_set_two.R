# Author: Temi
# Date:
# Description: This script was used to train on df_two.csv


# set these depending on your cluster
setwd('/Users/temi/Dropbox/UND_Temidayo-Adeluwa/prams/PRAMS/PRAMS/scripts/')
data_dir <- '/Users/temi/Dropbox/UND_Temidayo-Adeluwa/prams/PRAMS/data'
objects_dir <- '/Users/temi/Dropbox/UND_Temidayo-Adeluwa/prams/PRAMS/objects'
if(!dir.exists(objects_dir){
    dir.create(objects_dir)
})

source('./modules/helpers.R') # some helper functions

set.seed(1)

library(caret)
library(foreign)
library(tidyverse)
library(parallel)
library(doParallel)
library(future)
library(ggplot2)
library(plyr)
library(dplyr)
library(gbm)
library(DMwR)
library(kernlab)
library(nnet)
library(pROC)
library(naivebayes)
library(gridExtra)
#library(mboost)
library(MASS)
library(rpart)
library(adabag)
library(mctest)
library(corpcor)
library(ROSE)
library(bimba)
#library(FCBF)
# library(FSelector)
# library(RoughSets)

set.two <- read.csv(file.path(data_dir, 'df_two.csv'), stringsAsFactors = T, header=T)

# preprocess
# csyj <- caret::preProcess(x=set.two[, !names(set.two) %in% 'PP_DEPRESS'], 
#                           method=c('center', 'scale', 'YeoJohnson'))
# set.two <- predict(csyj, set.two)

# shuffle the data
set.two <- set.two[sample(nrow(set.two)), ]

# smote
set.two <- as.data.frame(cbind(set.two[, !names(set.two) %in% "PP_DEPRESS"], PP_DEPRESS=as.factor(set.two$PP_DEPRESS)))
set.seed(1)
set.two <- bimba::SMOTE(set.two, perc_min = 50, k = 5)

# predictors
predictors <- names(set.two)[!names(set.two) %in% 'PP_DEPRESS']

# general control
general.ctrl <- trainControl(method='repeatedcv', number=10, repeats=5,
                             allowParallel = T, summaryFunction = twoClassSummary,
                             classProbs = T, savePredictions = T)

cl <- makeCluster(120)
registerDoParallel(cl)

print('=== Training with relief features on set.two ===')

set.seed(1)
two.relief.naive_bayes <- caret::train(x=set.two[, names(set.two) %in% predictors],
                                       y=as.factor(set.two$PP_DEPRESS),
                                       trControl = general.ctrl,
                                       method='naive_bayes', metric='ROC', tuneLength = 20)
print(two.relief.naive_bayes)

set.seed(1)
two.relief.rf <- caret::train(x=set.two[, names(set.two) %in% predictors],
                              y=as.factor(set.two$PP_DEPRESS),
                              trControl = general.ctrl,
                              method='rf', metric='ROC', tuneLength=10)
print(two.relief.rf)

set.seed(1)
two.relief.gbm <- caret::train(x=set.two[, names(set.two) %in% predictors],
                               y=as.factor(set.two$PP_DEPRESS), trControl = general.ctrl,
                               method='gbm', metric='ROC', tuneGrid=expand.grid(interaction.depth=c(2:4),
                                                                                n.trees=seq(140, 145, by=1),
                                                                                shrinkage=0.1,
                                                                                n.minobsinnode=c(40:50)))
print(two.relief.gbm)

set.seed(1)
two.relief.glm <- caret::train(x=set.two[, names(set.two) %in% predictors],
                               y=as.factor(set.two$PP_DEPRESS),
                               trControl = general.ctrl,
                               family=binomial(link = 'logit'),
                               maxit=200, method='glm',
                               metric='ROC')
print(two.relief.glm)

set.seed(1)
two.relief.rpart <- caret::train(x=set.two[, names(set.two) %in% predictors],
                                 y=as.factor(set.two$PP_DEPRESS),
                                 trControl = general.ctrl, method='rpart',
                                 metric='ROC', tuneGrid=data.frame(cp=seq(0.002, 0.003, length=1000)))
print(two.relief.rpart)

# tuneGrid=expand.grid(size=1,
#                      decay=seq(10, 39, length=2000))

set.seed(1)
two.relief.nnet <- caret::train(x=set.two[, names(set.two) %in% predictors],
                                y=as.factor(set.two$PP_DEPRESS),
                                trControl = general.ctrl, maxit=600,
                                method='nnet', metric='ROC')
print(two.relief.nnet)

set.seed(1)
two.relief.svm <- caret::train(x=set.two[, names(set.two) %in% predictors],
                               y=as.factor(set.two$PP_DEPRESS),
                               trControl = general.ctrl,
                               method='svmRadial', metric='ROC', tuneGrid=expand.grid(C=seq(60, 80, by=2),
                                                                                      sigma=c(0.006629303)))
print(two.relief.svm)

set.seed(1)
two.relief.knn <- caret::train(x=set.two[, names(set.two) %in% predictors],
                               y=as.factor(set.two$PP_DEPRESS),
                               trControl = general.ctrl,
                               method='knn', metric='ROC', tuneGrid=data.frame(k=seq(1, 30, by=2)))
print(two.relief.knn)

set.seed(1)
two.relief.adaboost <- caret::train(x=set.two[, names(set.two) %in% predictors],
                                    y=as.factor(set.two$PP_DEPRESS),
                                    trControl = general.ctrl,
                                    method='AdaBoost.M1', metric='ROC', tuneGrid=expand.grid(maxdepth=c(2),
                                                                                             mfinal=seq(500, 800, by=25),
                                                                                             coeflearn=c('Breiman')))
print(two.relief.adaboost)


stopCluster(cl)

set_two.list <- list(`LR_Set2`=two.relief.glm,
                     `RF_Set2`=two.relief.rf,
                     `RPART_Set2`=two.relief.rpart,
                     `NNET_Set2`=two.relief.nnet,
                     `NB_Set2`=two.relief.naive_bayes,
                     `GBM_Set2`=two.relief.gbm,
                     `AdaBoost_Set2`=two.relief.adaboost,
                     `kNN_Set2`=two.relief.knn,
                     `SVM_Set2`=two.relief.svm)

save(set_two.list, file=file.path(objects_dir, 'set_two_fits.RData'))