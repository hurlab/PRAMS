# Author
# Date:
# Description

createDummyFeatures <- function(df=df.prams, target.name='PP_DEPRESS'){
    # one-hot encode catogerical variables with just two categories. Treat the others as continuous variables.
    cats <- sort(apply(df, 2, function(x){
        length(unique(x))
    }))
    
    names.cats <- names(cats)[cats > 1 & cats < 3 & !names(cats) %in% c(target.name)]
    
    for(e in names.cats) {
        df[[e]] <- as.factor(as.character(df[[e]]))
    }
    
    cat.prams <- as.data.frame(cbind(df[, names(df) %in% names.cats], Class=as.factor(df[[target.name]])))
    # normalize categorical dummy variables
    dv <- caret::dummyVars(Class ~ ., data=cat.prams, fullRank=T)
    cat.prams <- predict(dv, cat.prams)
    cat.prams <- as.data.frame(cbind(as.data.frame(cat.prams), Class=as.factor(df[[target.name]])))
    
    set.seed(1)
    cs <- caret::preProcess(x=cat.prams[, !names(cat.prams) %in% 'Class'],
                            method=c('center', 'scale'))
    cat.prams <- predict(cs, cat.prams)
    names(cat.prams)[names(cat.prams)=='Class'] <- target.name
    
    # normalize the continuous features
    cont.prams <- df[, !names(df) %in% c(names.cats, target.name)]
    #cont.prams <- cont.prams[, c(2:ncol(cont.prams), 1)]
    set.seed(1)
    csyj <- caret::preProcess(x=cont.prams[, !names(cont.prams) %in% 'PP_DEPRESS'],
                              method=c('center', 'scale', 'YeoJohnson'))
    cont.prams <- predict(csyj, cont.prams)
    
    # remove linear combinations of features
    #lin.combos <- findLinearCombos(cat.prams)
    #cat.prams <- cat.prams[, -lin.combos$remove]
    
    # predict cat
    output <- as.data.frame(cbind(cat.prams, cont.prams))
    
    # preprocess
    # set.seed(1)
    # csyj <- caret::preProcess(x=output[, !names(output) %in% 'PP_DEPRESS'], 
    #                           method=c('center', 'scale', 'YeoJohnson'))
    # output <- predict(csyj, output)
    
    return(output)
}


analyseTopFeatures <- function(names.list, top.n){
    
    #' Given a list of models, this function returns a data frame of the 
    #' top n features for each model
    
    common.list <- list()
    for (i in 1:length(names.list)) {
        e.imp <- as.data.frame(caret::varImp(names.list[[i]])$importance)
        e.imp$feature <- row.names(e.imp)
        if ('Overall' %in% names(e.imp)) {
            e.imp <- e.imp[with(e.imp, order(Overall, decreasing = T)), ]
        } else {
            e.imp <- e.imp[with(e.imp, order(Depressed, decreasing = T)), ]
            e.imp$Positive <- NULL
            names(e.imp) <- 'Overall'
        }
        common.list[[names(names.list)[i]]] <- row.names(e.imp)[1:top.n]
    }
    return(as.data.frame(common.list))
}


freqTableFromDataframe <- function(df) {
    
    #' Create a frequency table 
    #' 
    #' Given a data frame returned by analyseTopFeatures, this function
    #' ranks the features returned by each model and also creates a frequency table of 
    #' the top n most important features
    
    all.features <- unique(unlist(df))
    freq.features <- sapply(all.features, function(x){
        length(df[df==x])
    })
    freq.features <- as.data.frame(sort(freq.features, decreasing = T))
    names(freq.features) <- 'Frequency'
    freq.features <- freq.features %>% 
        tibble::rownames_to_column() 
    names(freq.features)[1] <- 'Features' 
    
    rank.features <- list()    
    for (i in 1:ncol(df)){
        named.num.i <- setNames(rep(1:length(df[[i]])), df[[i]])
        res <- rep('NA', nrow(freq.features))
        for(j in 1:length(named.num.i)){
            res <- replace(res, which(names(named.num.i)[j]  == freq.features$Feature), named.num.i[names(named.num.i)==names(named.num.i)[j]][[1]])
        }
        rank.features[[paste(names(df)[i], '_rank', sep='')]] <- res
    }
    
    temp <- as.data.frame(do.call(cbind, rank.features))
    return(cbind(freq.features, temp))
}

rearrangeFreqTable <- function(df, features_desc){
    
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