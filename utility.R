## count elements in a class
countElems <- function(class, table) {
    nrow(table[class])
}

## returns the embedding of a word from the matrix of embeddings of all words
func <- function(word, embd) {

    e1 <- tryCatch({
        embd[word, ]
    }, error = function(cond) { }
    )
    if(is.null(e1)) {
        e1 <- rep(NA, 50)
    }
    return(e1)
}


## builds embeddings of product names by adding or averaging the embeddings of the words which compose the name
buildEmbedding <- function(pname, emb, type = "ADD") {
    tokens <- unlist(str_split(pname, " "))
    embds <- sapply(tokens, func, emb)
    if( type == "ADD") {
        word_vector <- rowSums(embds, na.rm = TRUE)
    }
    else if(type == "MEAN") {
        word_vector <- rowMeans(embds, na.rm = TRUE)
    }
    else
        word_vector <- NULL

    return(word_vector)
}

## build embedding of product names for fasttext method (an average of the vectors divided by theirs L2 norm)
buildFastTextEmbedding <- function(pname, emb) {
    tokens <- unlist(str_split(pname, " "))
    tokens <- c(tokens, "</s>")
    embds <- sapply(tokens, func, emb)
    L2<-c()
    counter <- 0
    sumv <- 0
    for(i in 1:ncol(embds)) {
        L2<- norm(unlist(embds[1:50,i]), "2")
        if(L2 > 0 ) {
            counter <- counter + 1
            sumv <- sumv + unlist(embds[1:50,i])
        }
    }
    return(sumv / counter)
}

#### check if we have OOV words                     ####

#### a function to create a set of words from product names ####
createSetOfWords <- function(prod_names) {
    words <- list()
    for( i in 1:length(prod_names)) {
        row <- prod_names[i]
        tokens <- str_split(row, " ")
        words <- append(words, unlist(tokens))

    }
    words <- unique(words)
    return(words)
}

#### a function to check if a given word is in a set of words   ####
checkWord <- function(word1, wordset) {
    if(!(word1 %in% wordset))
        return(word1)
}

#### a function to check if the words in test set are in the training set. It returns the out of vocabulary words ####
checkOOV <- function(testset, trainset) {
    testwords <- createSetOfWords(testset$nume)
    trainwords <- createSetOfWords(trainset$nume)

    oov <- lapply(testwords, checkWord, trainwords)
    oov <- oov[!sapply(oov,is.null)]

    return(oov)
}

w <- c(0.02593761,
       0.007360673,
       0.007360673,
       0.011566772,
       0.326323169,
       0.268839818,
       0.231335436,
      0.004907115,
      0.015772871,
      0.012618297,
      0.020679986,
      0.015772871,
      0.030844725,
      0.014721346,
      0.00595864
)

buildAccMeasures <- function (trueValues, predictedValues, methodName) {
    cMatrix <- confusionMatrix(predictedValues, trueValues)
    accuracy <- cMatrix$overall["Accuracy"]
    # sensitivity <- cMatrix$byClass["Sensitivity"]
    # specificity <- cMatrix$byClass["Specificity"]
    # f1 <- 2*sensitivity*specificity/(sensitivity+specificity)
    return (model_accs <- tibble(Method = methodName,
                                 Accuracy = accuracy,
                                 #"Sensitivity/Recall" = sensitivity,
                                 "F1 score" = mean(cMatrix$byClass[,7], na.rm = TRUE),
                                 "F1 weighted" = mean(cMatrix$byClass[,7]*(1-w), na.rm = TRUE)
    )
    )
}

#oovWords <- checkOOV(test, train)

#### For each word in the oov list we have to predict the embedding ####
#embedding2 <- predict(model_cbow, type = "embedding", newdata = unlist(oovWords))

