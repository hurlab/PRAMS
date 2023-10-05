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

set.three <- read.csv(file.path(data_dir, 'df_three.csv'), stringsAsFactors = T, header=T)

# preprocess
# csyj <- caret::preProcess(x=set.three[, !names(set.three) %in% 'PP_DEPRESS'], 
#                           method=c('center', 'scale', 'YeoJohnson'))
# set.three <- predict(csyj, set.three)

# shuffle the data
set.three <- set.three[sample(nrow(set.three)), ]

# smote
set.three <- as.data.frame(cbind(set.three[, !names(set.three) %in% "PP_DEPRESS"], PP_DEPRESS=as.factor(set.three$PP_DEPRESS)))
set.seed(1)
set.three <- bimba::SMOTE(set.three, perc_min = 50, k = 5)

# predictors
predictors <- names(set.three)[!names(set.three) %in% 'PP_DEPRESS']

# general control
general.ctrl <- trainControl(method='repeatedcv', number=10, repeats=5,
                             allowParallel = T, summaryFunction = twoClassSummary,
                             classProbs = T, savePredictions = T)

cl <- makeCluster(120)
registerDoParallel(cl)

print('=== Training with relief features on set.three ===')

set.seed(1)
three.relief.naive_bayes <- caret::train(x=set.three[, names(set.three) %in% predictors],
                                       y=as.factor(set.three$PP_DEPRESS),
                                       trControl = general.ctrl,
                                       method='naive_bayes', metric='ROC', tuneLength = 20)
print(three.relief.naive_bayes)

set.seed(1)
three.relief.rf <- caret::train(x=set.three[, names(set.three) %in% predictors],
                              y=as.factor(set.three$PP_DEPRESS),
                              trControl = general.ctrl,
                              method='rf', metric='ROC', tuneLength=10)
print(three.relief.rf)

set.seed(1)
three.relief.gbm <- caret::train(x=set.three[, names(set.three) %in% predictors],
                               y=as.factor(set.three$PP_DEPRESS), trControl = general.ctrl,
                               method='gbm', metric='ROC', tuneGrid=expand.grid(interaction.depth=c(2:4),
                                                                                n.trees=seq(140, 145, by=1),
                                                                                shrinkage=0.1,
                                                                                n.minobsinnode=c(40:50)))
print(three.relief.gbm)

set.seed(1)
three.relief.glm <- caret::train(x=set.three[, names(set.three) %in% predictors],
                               y=as.factor(set.three$PP_DEPRESS),
                               trControl = general.ctrl,
                               family=binomial(link = 'logit'),
                               maxit=200, method='glm',
                               metric='ROC')
print(three.relief.glm)

set.seed(1)
three.relief.rpart <- caret::train(x=set.three[, names(set.three) %in% predictors],
                                 y=as.factor(set.three$PP_DEPRESS),
                                 trControl = general.ctrl, method='rpart',
                                 metric='ROC', tuneGrid=data.frame(cp=seq(0.002, 0.003, length=1000)))
print(three.relief.rpart)

# tuneGrid=expand.grid(size=1,
#                      decay=seq(10, 39, length=2000))

set.seed(1)
three.relief.nnet <- caret::train(x=set.three[, names(set.three) %in% predictors],
                                y=as.factor(set.three$PP_DEPRESS),
                                trControl = general.ctrl, maxit=600,
                                method='nnet', metric='ROC')
print(three.relief.nnet)

set.seed(1)
three.relief.svm <- caret::train(x=set.three[, names(set.three) %in% predictors],
                               y=as.factor(set.three$PP_DEPRESS),
                               trControl = general.ctrl,
                               method='svmRadial', metric='ROC', tuneGrid=expand.grid(C=seq(60, 80, by=2),
                                                                                      sigma=c(0.006629303)))
print(three.relief.svm)

set.seed(1)
three.relief.knn <- caret::train(x=set.three[, names(set.three) %in% predictors],
                               y=as.factor(set.three$PP_DEPRESS),
                               trControl = general.ctrl,
                               method='knn', metric='ROC', tuneGrid=data.frame(k=seq(1, 30, by=2)))
print(three.relief.knn)

set.seed(1)
three.relief.adaboost <- caret::train(x=set.three[, names(set.three) %in% predictors],
                                    y=as.factor(set.three$PP_DEPRESS),
                                    trControl = general.ctrl,
                                    method='AdaBoost.M1', metric='ROC', tuneGrid=expand.grid(maxdepth=c(2),
                                                                                             mfinal=seq(500, 800, by=25),
                                                                                             coeflearn=c('Breiman')))
print(three.relief.adaboost)


stopCluster(cl)

set_three.list <- list(`LR_Set3`=three.relief.glm,
                     `RF_Set3`=three.relief.rf,
                     `RPART_Set3`=three.relief.rpart,
                     `NNET_Set3`=three.relief.nnet,
                     `NB_Set3`=three.relief.naive_bayes,
                     `GBM_Set3`=three.relief.gbm,
                     `AdaBoost_Set3`=three.relief.adaboost,
                     `kNN_Set3`=three.relief.knn,
                     `SVM_Set3`=three.relief.svm)

save(set_three.list, file=file.path(objects_dir, 'set_three_fits.RData'))


