


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
    as.data.frame(common.list)
}



freqTableFromDataframe <- function(df) {
    
    #' Create a frequency table 
    #' 
    #' Given a data frame returned by analyse.top.features, this function
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