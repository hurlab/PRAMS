# Author: Temi
# Date:
# Description: This script contains functions used to collect metrics and create plots from '../objects'

setwd('/Users/temi/Dropbox/UND_Temidayo-Adeluwa/prams/PRAMS/PRAMS/scripts/')
objects_dir <- '/Users/temi/Dropbox/UND_Temidayo-Adeluwa/prams/PRAMS/objects'
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

feature_desc <- read.csv('../../data/feature_details.csv', header=T)

set1.freq.table <- freqTableFromDataframe(analyseTopFeatures(set1.top.four, 20))
set2.freq.table <- freqTableFromDataframe(analyseTopFeatures(set2.top.four, 20))

rearrangeFreqTable <- function(df){
    
    one.half <- subset(df, grepl('\\.2', df$Features)) %>% 
        separate(Features, into=c('keep', 'discard'), sep='\\.') %>% 
        dplyr::select(-c('discard')) %>%
        dplyr::rename(Features=keep)
        
    two.half <- subset(df, !grepl('\\.2', df$Features))
    
    df <- as.data.frame(rbind(one.half, two.half))
    
    var_desc <- subset(feature_desc, feature_desc$variable %in% df$Features)
    
    names(var_desc)[1] <- 'Features'
    
    freq.tab.labels <- merge(df, var_desc, by='Features', all=T)
    
    freq.tab.labels <- freq.tab.labels[!duplicated(freq.tab.labels$Features), ] %>%
        arrange(desc(Frequency), RF_Set1_rank)
    
    freq.tab.labels

}

write.csv(rearrangeFreqTable(set1.freq.table), file='../results/freq_table_set1.csv', row.names = F)

write.csv(freq.tab.labels, file='../results/freq_table_set2.csv', row.names = F)


gatherMetrics <- function(model.list) {
    
    # To-do: 
    # 1. Specify what metrics are available to collect.
    # ...So that you can specify what metrics you want to to collect
    #
    # 2. 
    
    #' Gather metrics from a list of Caret train objects
    #' 
    #' @description This function collates the AUC, Specificity, Accuracy
    #' Sensitivity, Precision, Recall, and F1 of a list of Caret models.
    #' 
    #' @param model.list A list of caret train objects
    #' 
    #' @usage temp.gather.metrics(model.list)
    #' 
    #' @note IMPORTANT: Ensure that while training, you set 'savePredictions' to TRUE
    #' so that predictions within resamplings are saved. 
    #' 
    #' @return A dataframe of metrics.
    #' 
    #' As long as the list of models are caret trained objects, you should be fine
    #' 
    #' 
    #'
    #'
    #' 
    #' named list of train objects returned by caret.
    #' 
    
    resampled <- resamples(model.list)
    summary.resamples <- summary(resampled)
    
    df.resampled <- as.data.frame(summary.resamples$statistics)
    
    df.resampled <- df.resampled %>%
        dplyr::select('ROC.Mean', 'Sens.Mean', 'Spec.Mean') %>%
        dplyr::rename(ROC=ROC.Mean, Sensitivity=Sens.Mean, Specificity=Spec.Mean) %>%
        round(3) %>% 
        tibble::rownames_to_column() %>% 
        dplyr::rename(Model=rowname)
    
    n <- paste0(names(model.list), '~ROC')
    ROC.SD <- list()
    for (i in 1:length(n)) {
        ROC.SD[[names(model.list)[i]]] <- as.numeric(round(sd(summary.resamples$values[[i]], na.rm=T), 3))
    }
    sd.df <- as.data.frame(do.call(rbind, ROC.SD)) 
    names(sd.df) <- 'ROC.SD'
    sd.df <- sd.df %>%
        tibble::rownames_to_column(var='Model') 
    
    others <- lapply(model.list, function(y){
        caret::confusionMatrix(y$pred$pred, y$pred$obs, mode='prec_recall')
    })
    
    other.metrics <- lapply(others, function(t){
        t(as.data.frame(t$byClass)) %>% 
            as.data.frame() %>% 
            dplyr::select(`Balanced Accuracy`, Precision, Recall, F1) %>% 
            dplyr::rename(Accuracy=`Balanced Accuracy`) %>%
            round(3)
    })
    rem.metrics <- as.data.frame(do.call(rbind, other.metrics)) %>%
        tibble::rownames_to_column(var = 'Model')
    
    # merge
    temp <- merge(df.resampled, rem.metrics, by='Model', all=T)
    result <- merge(temp, sd.df, by='Model', all=T)
    colss <- c('Model', 'ROC', 'ROC.SD', 'Sensitivity', 'Specificity', 'Accuracy', 'Precision', 'Recall', 'F1')
    result <- result[, colss]
    
    result %>%
        arrange(desc(ROC))
}
columns <- c('Model', 'ROC', 'Sensitivity', 'Specificity', 'Accuracy')

set1.metrics <- gatherMetrics(set1.list) %>%
    dplyr::select(columns) %>%
    separate(Model, into=c('keep', 'discard'), sep='_') %>% 
    dplyr::select(-c('discard')) %>% 
    dplyr::rename(Model=keep)

set2.metrics <- gatherMetrics(set2.list) %>%
    dplyr::select(columns) %>%
    separate(Model, into=c('keep', 'discard'), sep='_') %>% 
    dplyr::select(-c('discard')) %>% 
    dplyr::rename(Model=keep)

set3.metrics <- gatherMetrics(set3.list) %>%
    dplyr::select(columns) %>%
    separate(Model, into=c('keep', 'discard'), sep='_') %>% 
    dplyr::select(-c('discard')) %>% 
    dplyr::rename(Model=keep)

write.csv(set1.metrics, file='../results/set1_metrics.csv', row.names = F)
write.csv(set2.metrics, file='../results/set2_metrics.csv', row.names = F)
write.csv(set3.metrics, file='../results/set3_metrics.csv', row.names = F)


gatherPlotROC <- function(model.list, case='', control='', title='', algo.names=n.models, random.select=NULL) {
    library(tidyverse)
    library(parallel)
    
    #' Given a list of caret models, this function builds the ROC plots for each model into a single figure:
    #' It is important to give the case and control names, which are the names of levels of the classes, 
    #' and if you want, give the plot a title. Else, there is a default title name
    
    temp.roc <- parallel::mclapply(model.list, function(x){
        pROC::roc(predictor=x$pred[[case]], 
                  response=x$pred$obs, levels=c(control, case))
    }, mc.cores=12)
    
    collect <- function(roc.object){
        FPR <- rev(1-roc.object[['specificities']])
        TPR <- rev(roc.object[['sensitivities']])
        bound <- cbind(TPR, FPR)
        return(bound)
    }
    
    temp.out <- list()
    for (i in 1:length(temp.roc)){
        values <- data.frame(collect(temp.roc[[i]]))
        model <- rep(paste(names(temp.roc[i]), round(temp.roc[[i]]$auc, 3), sep=': '),
                     nrow(values))
        auc <- rep(round(temp.roc[[i]]$auc, 3),
                   nrow(values))
        temp.out[[i]] <- data.frame(cbind(values, model, auc))
    }
    
    comb <- as.data.frame(do.call(rbind, temp.out))
    
    split.comb <- parallel::mclapply(n.models, function(r){
        comb %>%
            filter(grepl(r, model))
    }, mc.cores = 12)
    
    names(split.comb) <- n.models
    
    if (random.select == 'min'){
        temp.comb.two <- parallel::mclapply(split.comb, function(r){
            sample_n(r, min(unlist(lapply(split.comb, nrow))))}, mc.cores = 12)
    } else if (is.double(random.select)) {
        # use this only when you know what you are doing
        if (random.select > min(unlist(lapply(split.comb, nrow)))) {
            print('Error: the random select number is higher than the minimum rows available')
            break
        } else {
            temp.comb.two <- mclapply(split.comb, function(r){
                sample_n(r, random.select)}, mc.cores = 12)
        }
    } else {
        temp.comb.two <- split.comb
    }
    
    names(temp.comb.two) <- n.models
    
    comb.two <- as.data.frame(do.call(rbind, temp.comb.two))
    
    # change the names in the model column
    spec.model <- comb.two %>%
        separate(model, into=c('keep', 'discard'), sep='_') %>%
        pull(keep)
    model_auc <- paste(spec.model, comb.two$auc, sep=': ')
    
    comb.two$what_model <- spec.model
    comb.two$model_auc <- model_auc
    
    roc.plot <- comb.two %>% 
        dplyr::mutate(model = reorder(model_auc, auc)) %>%
        ggplot2::ggplot(aes(x=as.numeric(FPR), y=as.numeric(TPR), col=model)) + geom_line() + theme_bw() +
        geom_abline(intercept = 0, slope = 1, alpha=0.8, col='grey') + ylim(0, 1) + xlim(0, 1) +
        theme(legend.text=element_text(size=14),
              legend.title = element_text(color='black', size=15),
              axis.title.x=element_text(size=15, face='plain'),
              axis.title.y=element_text(size=15, face='plain'),
              plot.title = element_text(face='plain', size=20),
              axis.line = element_line(colour ='black', size=1)) +
        coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
        labs(col='Model: AUC', x='False Positive Rate', y='True Positive Rate', title=ifelse(title=='', '', title))
    
    return(roc.plot)
    # return(comb)
    # return(comb.two)
    
}
temp.roc <- gatherPlotROC(set1.list, case='Depressed', control='Healthy',
                                random.select = 'min')
pdf(file='../results/roc_plot_set1.pdf', width=11, height=9)
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

write.csv(sum.all.metrics.df, file='../results/average_metrics.csv', row.names = F)
