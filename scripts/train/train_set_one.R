# Author: Temi
# Date:
# Description: This script was used to train on df_one.csv


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
library(foreign)
library(gbm)
library(parallel)
library(doParallel)
library(DMwR)
library(kernlab)
library(nnet)
library(future)
library(pROC)
library(naivebayes)
library(gridExtra)
library(mboost)
library(MASS)
library(rpart)
library(adabag)
library(mctest)
library(corpcor)
library(ROSE)
library(bimba)
library(FCBF)
library(FSelector)
library(RoughSets)

set.one <- read.csv(file.path(data_dir, 'df_one.csv'), stringsAsFactors = T, header=T)

# # preprocess
# csyj <- caret::preProcess(x=set.one[, !names(set.one) %in% 'PP_DEPRESS'], 
#                           method=c('center', 'scale', 'YeoJohnson'))
# set.one <- predict(csyj, set.one)


# shuffle the data
set.one <- set.one[sample(nrow(set.one)), ]

# smote
set.one <- as.data.frame(cbind(set.one[, !names(set.one) %in% "PP_DEPRESS"], PP_DEPRESS=as.factor(set.one$PP_DEPRESS)))
set.one <- bimba::SMOTE(set.one, perc_min = 50, k = 5)

# predictors
predictors <- names(set.one)[!names(set.one) %in% 'PP_DEPRESS']

# general control for caret
general.ctrl <- trainControl(method='repeatedcv', number=10, repeats=5,
                             allowParallel = T, summaryFunction = twoClassSummary,
                             classProbs = T, savePredictions = T)

cl <- makeCluster(120) # using 120 cores
registerDoParallel(cl)

print('=== Training with relief features on set.one ===')

set.seed(1)
one.relief.naive_bayes <- caret::train(x=set.one[, names(set.one) %in% predictors],
                                       y=as.factor(set.one$PP_DEPRESS),
                                       trControl = general.ctrl,
                                       method='naive_bayes', metric='ROC', tuneLength = 20)
print(one.relief.naive_bayes)

set.seed(1)
one.relief.rf <- caret::train(x=set.one[, names(set.one) %in% predictors],
                              y=as.factor(set.one$PP_DEPRESS),
                              trControl = general.ctrl,
                              method='rf', metric='ROC', tuneLength=10)
print(one.relief.rf)

set.seed(1)
one.relief.gbm <- caret::train(x=set.one[, names(set.one) %in% predictors],
                               y=as.factor(set.one$PP_DEPRESS), trControl = general.ctrl,
                               method='gbm', metric='ROC', tuneGrid=expand.grid(interaction.depth=c(2:4),
                                                                                n.trees=seq(140, 145, by=1),
                                                                                shrinkage=0.1,
                                                                                n.minobsinnode=c(40:50)))
print(one.relief.gbm)

set.seed(1)
one.relief.glm <- caret::train(x=set.one[, names(set.one) %in% predictors],
                               y=as.factor(set.one$PP_DEPRESS),
                               trControl = general.ctrl,
                               family=binomial(link = 'logit'),
                               maxit=200, method='glm',
                               metric='ROC')
print(one.relief.glm)

set.seed(1)
one.relief.rpart <- caret::train(x=set.one[, names(set.one) %in% predictors],
                                 y=as.factor(set.one$PP_DEPRESS),
                                 trControl = general.ctrl, method='rpart',
                                 metric='ROC', tuneGrid=data.frame(cp=seq(0.002, 0.003, length=1000)))
print(one.relief.rpart)

# tuneGrid=expand.grid(size=1,
#                      decay=seq(10, 39, length=2000))

set.seed(1)
one.relief.nnet <- caret::train(x=set.one[, names(set.one) %in% predictors],
                                y=as.factor(set.one$PP_DEPRESS),
                                trControl = general.ctrl, maxit=600,
                                method='nnet', metric='ROC')
print(one.relief.nnet)

set.seed(1)
one.relief.svm <- caret::train(x=set.one[, names(set.one) %in% predictors],
                               y=as.factor(set.one$PP_DEPRESS),
                               trControl = general.ctrl,
                               method='svmRadial', metric='ROC', tuneGrid=expand.grid(C=seq(60, 80, by=2),
                                                                                      sigma=c(0.006629303)))
print(one.relief.svm)

set.seed(1)
one.relief.knn <- caret::train(x=set.one[, names(set.one) %in% predictors],
                               y=as.factor(set.one$PP_DEPRESS),
                               trControl = general.ctrl,
                               method='knn', metric='ROC', tuneGrid=data.frame(k=seq(1, 30, by=2)))
print(one.relief.knn)

set.seed(1)
one.relief.adaboost <- caret::train(x=set.one[, names(set.one) %in% predictors],
                                    y=as.factor(set.one$PP_DEPRESS),
                                    trControl = general.ctrl,
                                    method='AdaBoost.M1', metric='ROC', tuneGrid=expand.grid(maxdepth=c(2),
                                                                                             mfinal=seq(500, 800, by=25),
                                                                                             coeflearn=c('Breiman')))
print(one.relief.adaboost)


stopCluster(cl)

set_one.list <- list(`LR_Set1`=one.relief.glm,
                  `RF_Set1`=one.relief.rf,
                  `RPART_Set1`=one.relief.rpart,
                  `NNET_Set1`=one.relief.nnet,
                  `NB_Set1`=one.relief.naive_bayes,
                  `GBM_Set1`=one.relief.gbm,
                  `AdaBoost_Set1`=one.relief.adaboost,
                  `kNN_Set1`=one.relief.knn,
                  `SVM_Set1`=one.relief.svm)

save(set_one.list, file=file.path(objects_dir, 'set_one_fits.RData'))