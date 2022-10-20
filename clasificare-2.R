library(readxl)
library(data.table)
library(caTools)
library(superml)
library(word2vec)
library(fastText)
library(stringr)
library(text2vec)
library(glmnetUtils)
library(caret)
library(tidyverse)
library(e1071)
library(rpart)
library(xgboost)
library(tree)
library(naivebayes)
library(MASS)
library(ada)
library(RWeka)
library(bst)
library(C50)
library(ranger)
library(Rtsne)
#### Read the data file                             ####
path <- "D:/r-projects/autoencoding"
setwd(path)
source('utility.R')
products <- read_excel(file.path(path, "products.xlsx"))
#View(products)

#### Remove unclassified products                   ####
products <- as.data.table(products)
setkey(products, 'cod')
products <- products[cod != "-1"]

#### Perform some preprocessing                     ####
products$nume <- sapply(products$nume, tolower)
products$nume <- trimws(products$nume, which = c("both"))
products$nume <- gsub(",","",products$nume)
products$nume <- gsub(":","",products$nume)
products$nume <- gsub(";","",products$nume)
products$nume <- sapply(products$nume, str_squish)

#### Classes of products                            ####
unique(products$cod)

length(unique(products$cod))

#### Compute the number of products in each class   ####
classes <- unique(products$cod)
noProducts <- lapply(classes, countElems, products)
#View(noProducts)


#### Build the dataset for auto classification       ####

myCols <- c("nume", "cod")
products <- products[, ..myCols]
codes<- unique(products$cod)
df<-data.frame(codes, numcodes = 1:15)
for(i in 1:nrow(df)) {
    products[cod == df[i,1], cod := df[i,2]]
}

sample <- sample.split(products$cod, SplitRatio = 0.7)
train  <- subset(products, sample == TRUE)
test   <- subset(products, sample == FALSE)




#### Build Word Embedding                           ####

#### 1. Count Vectorization                         ####


#### fit the model on the entire dataset            ####
cfv <- CountVectorizer$new(max_features = 3000, remove_stopwords = TRUE, ngram_range = c(1,3))
cfv$fit(products$nume)
train_cf_features_CV <- cfv$transform(train$nume)
test_cf_features_CV <- cfv$transform(test$nume)
train_cf_features_CV <- cbind(train$cod, train_cf_features_CV)
test_cf_features_CV <- cbind(test$cod, test_cf_features_CV)

## prepare data for ML
train_cf_features_CV <- as.data.frame(train_cf_features_CV)
train_cf_features_CV[, 1] <- as.factor(train_cf_features_CV[, 1])
train_cf_features_CV[, 2:ncol(train_cf_features_CV)] <- sapply(train_cf_features_CV[, 2:ncol(train_cf_features_CV)], as.numeric)
colnames(train_cf_features_CV)[1] <- "cod"

test_cf_features_CV <- as.data.frame(test_cf_features_CV)
test_cf_features_CV[, 1] <- as.factor(test_cf_features_CV[, 1])
test_cf_features_CV[, 2:ncol(test_cf_features_CV)] <- sapply(test_cf_features_CV[, 2:ncol(test_cf_features_CV)], as.numeric)
colnames(test_cf_features_CV)[1] <- "cod"


#### 3. TF-IDF                                      ####
#### fit the model on the entire dataset            ####
tfv <- TfIdfVectorizer$new(max_features = 3000, remove_stopwords = TRUE, ngram_range = c(1,3))

tfv$fit(products$nume)

train_tf_features_TF_IDF <- tfv$transform(train$nume)
test_tf_features_TF_IDF <- tfv$transform(test$nume)
train_tf_features_TF_IDF <- cbind(train$cod, train_tf_features_TF_IDF)
test_tf_features_TF_IDF <- cbind(test$cod, test_tf_features_TF_IDF)


### prepare for ML
train_tf_features_TF_IDF <- as.data.frame(train_tf_features_TF_IDF)
train_tf_features_TF_IDF[ ,1] <- as.factor(train_tf_features_TF_IDF[,1])
train_tf_features_TF_IDF[, 2:ncol(train_tf_features_TF_IDF)] <- sapply(train_tf_features_TF_IDF[,2:ncol(train_tf_features_TF_IDF)], as.numeric)
colnames(train_tf_features_TF_IDF)[1] <- "cod"

test_tf_features_TF_IDF <- as.data.frame(test_tf_features_TF_IDF)
test_tf_features_TF_IDF[ ,1] <- as.factor(test_tf_features_TF_IDF[,1])
test_tf_features_TF_IDF[, 2:ncol(test_tf_features_TF_IDF)] <- sapply(test_tf_features_TF_IDF[,2:ncol(test_tf_features_TF_IDF)], as.numeric)
colnames(test_tf_features_TF_IDF)[1] <- "cod"

#### 4. Word2vec                                    ####

#### The CBOW model                       ####
model_cbow <- word2vec(x = products$nume, type = "cbow", dim = 50, window = 4, iter = 15, lr = 0.5, hs = TRUE, min_count = 1, threads = 4L, split = c(" \n!?\"#$<=>[]\\^`{|}~\t\v\f\r", "\n?!"),)
embedding <- as.matrix(model_cbow)

#### Now we build the embeddings of the product names ####

#### embeddings for product names are on columns ####
train_features_W2V_CBOW_ADD <- sapply(train$nume, buildEmbedding, embedding, "ADD")
test_features_W2V_CBOW_ADD <- sapply(test$nume, buildEmbedding, embedding, "ADD")

train_features_W2V_CBOW_MEAN <- sapply(train$nume, buildEmbedding, embedding, "MEAN")
test_features_W2V_CBOW_MEAN <- sapply(test$nume, buildEmbedding, embedding, "MEAN")

train_features_W2V_CBOW_ADD <- t(rbind(train$cod, train_features_W2V_CBOW_ADD))
test_features_W2V_CBOW_ADD <- t(rbind(test$cod, test_features_W2V_CBOW_ADD))

train_features_W2V_CBOW_MEAN <- t(rbind(train$cod, train_features_W2V_CBOW_MEAN))
test_features_W2V_CBOW_MEAN <- t(rbind(test$cod, test_features_W2V_CBOW_MEAN))


### prepare data for ML
train_features_W2V_CBOW_ADD <- as.data.frame(train_features_W2V_CBOW_ADD)
train_features_W2V_CBOW_ADD[,1] <- as.factor(train_features_W2V_CBOW_ADD[,1])
train_features_W2V_CBOW_ADD[,2:ncol(train_features_W2V_CBOW_ADD)] <- sapply(train_features_W2V_CBOW_ADD[,2:ncol(train_features_W2V_CBOW_ADD)], as.numeric)
colnames(train_features_W2V_CBOW_ADD)[1] <- "cod"

test_features_W2V_CBOW_ADD <- as.data.frame(test_features_W2V_CBOW_ADD)
test_features_W2V_CBOW_ADD[,1] <- as.factor(test_features_W2V_CBOW_ADD[,1])
test_features_W2V_CBOW_ADD[,2:ncol(test_features_W2V_CBOW_ADD)] <- sapply(test_features_W2V_CBOW_ADD[,2:ncol(test_features_W2V_CBOW_ADD)], as.numeric)
colnames(test_features_W2V_CBOW_ADD)[1] <- "cod"

train_features_W2V_CBOW_MEAN <- as.data.frame(train_features_W2V_CBOW_MEAN)
train_features_W2V_CBOW_MEAN[,1] <- as.factor(train_features_W2V_CBOW_MEAN[,1])
train_features_W2V_CBOW_MEAN[,2:ncol(train_features_W2V_CBOW_MEAN)] <- sapply(train_features_W2V_CBOW_MEAN[,2:ncol(train_features_W2V_CBOW_MEAN)], as.numeric)
colnames(train_features_W2V_CBOW_MEAN)[1] <- "cod"

test_features_W2V_CBOW_MEAN <- as.data.frame(test_features_W2V_CBOW_MEAN)
test_features_W2V_CBOW_MEAN[,1] <- as.factor(test_features_W2V_CBOW_MEAN[,1])
test_features_W2V_CBOW_MEAN[,2:ncol(test_features_W2V_CBOW_MEAN)] <- sapply(test_features_W2V_CBOW_MEAN[,2:ncol(test_features_W2V_CBOW_MEAN)], as.numeric)
colnames(test_features_W2V_CBOW_MEAN)[1] <- "cod"


#### skip-gram model                                ####
model_skip <- word2vec(x = train$nume, type = "skip-gram", dim = 50, window = 4, iter = 15, lr = 0.5, hs = TRUE, min_count = 1, threads = 4L, split = c(" \n!?\"#$<=>[]\\^`{|}~\t\v\f\r", "\n?!"),)
embedding_skip <- as.matrix(model_skip)

#### embeddings for product names are on columns ####
train_features_W2V_SKIP_ADD <- sapply(train$nume, buildEmbedding, embedding_skip, "ADD")
test_features_W2V_SKIP_ADD <- sapply(test$nume, buildEmbedding, embedding_skip, "ADD")
train_features_W2V_SKIP_MEAN <- sapply(train$nume, buildEmbedding, embedding_skip, "MEAN")
test_features_W2V_SKIP_MEAN <- sapply(test$nume, buildEmbedding, embedding_skip, "MEAN")

train_features_W2V_SKIP_ADD <- t(rbind(train$cod, train_features_W2V_SKIP_ADD))
test_features_W2V_SKIP_ADD <- t(rbind(test$cod, test_features_W2V_SKIP_ADD))
train_features_W2V_SKIP_MEAN <- t(rbind(train$cod, train_features_W2V_SKIP_MEAN))
test_features_W2V_SKIP_MEAN <- t(rbind(test$cod, test_features_W2V_SKIP_MEAN))


### prepare data for ML
train_features_W2V_SKIP_ADD <- as.data.frame(train_features_W2V_SKIP_ADD)
train_features_W2V_SKIP_ADD[,1] <- as.factor(train_features_W2V_SKIP_ADD[,1])
train_features_W2V_SKIP_ADD[,2:ncol(train_features_W2V_SKIP_ADD)] <- sapply(train_features_W2V_SKIP_ADD[,2:ncol(train_features_W2V_SKIP_ADD)], as.numeric)
colnames(train_features_W2V_SKIP_ADD)[1] <- "cod"

test_features_W2V_SKIP_ADD <- as.data.frame(test_features_W2V_SKIP_ADD)
test_features_W2V_SKIP_ADD[,1] <- as.factor(test_features_W2V_SKIP_ADD[,1])
test_features_W2V_SKIP_ADD[,2:ncol(test_features_W2V_SKIP_ADD)] <- sapply(test_features_W2V_SKIP_ADD[,2:ncol(test_features_W2V_SKIP_ADD)], as.numeric)
colnames(test_features_W2V_SKIP_ADD)[1] <- "cod"

train_features_W2V_SKIP_MEAN <- as.data.frame(train_features_W2V_SKIP_MEAN)
train_features_W2V_SKIP_MEAN[,1] <- as.factor(train_features_W2V_SKIP_MEAN[,1])
train_features_W2V_SKIP_MEAN[,2:ncol(train_features_W2V_SKIP_MEAN)] <- sapply(train_features_W2V_SKIP_MEAN[,2:ncol(train_features_W2V_SKIP_MEAN)], as.numeric)
colnames(train_features_W2V_SKIP_MEAN)[1] <- "cod"

test_features_W2V_SKIP_MEAN <- as.data.frame(test_features_W2V_SKIP_MEAN)
test_features_W2V_SKIP_MEAN[,1] <- as.factor(test_features_W2V_SKIP_MEAN[,1])
test_features_W2V_SKIP_MEAN[,2:ncol(test_features_W2V_SKIP_MEAN)] <- sapply(test_features_W2V_SKIP_MEAN[,2:ncol(test_features_W2V_SKIP_MEAN)], as.numeric)
colnames(test_features_W2V_SKIP_MEAN)[1] <- "cod"


#### 5. fastText                                    ####
#### First save product names in a txt file, one product per line ####
prod_names <- products$nume

write.table(prod_names, row.names = FALSE,  col.names = FALSE, quote = FALSE, file = paste0(path , "/", "products.txt"))

list_paramsCBOW <- list(command = 'cbow',
                   lr = 0.5,
                   dim = 50,
                   input = "products.txt",
                   output = file.path(".", 'cbow_word_vectors'),
                   verbose = 2,
                   thread = 4,
                   minCount = 1,
                   wordNgrams = 3,
                   minn = 1,
                   maxn = 3,
                   dim = 50,
                   ws = 4,
                   epoch = 15
                   )

resCBOW <- fasttext_interface(list_paramsCBOW,
                         path_output = file.path(".", 'cbow_products_log.txt'),
                         MilliSecs = 5,
                         remove_previous_file = TRUE,
                         print_process_time = TRUE)


embedding_fastText_CBOW <- read.table('cbow_word_vectors.vec', skip = 1, row.names = 1)

train_features_FASTTEXT_CBOW <- sapply(train$nume, buildFastTextEmbedding, embedding_fastText_CBOW)
test_features_FASTTEXT_CBOW <- sapply(test$nume, buildFastTextEmbedding, embedding_fastText_CBOW)

train_features_FASTTEXT_CBOW <- t(rbind(train$cod, train_features_FASTTEXT_CBOW))
test_features_FASTTEXT_CBOW <- t(rbind(test$cod, test_features_FASTTEXT_CBOW))


### prepare data for ML
train_features_FASTTEXT_CBOW <- as.data.frame(train_features_FASTTEXT_CBOW)
train_features_FASTTEXT_CBOW[,1] <- as.factor(train_features_FASTTEXT_CBOW[,1])
train_features_FASTTEXT_CBOW[,2:ncol(train_features_FASTTEXT_CBOW)] <- sapply(train_features_FASTTEXT_CBOW[,2:ncol(train_features_FASTTEXT_CBOW)], as.numeric)
colnames(train_features_FASTTEXT_CBOW)[1] <- "cod"

test_features_FASTTEXT_CBOW <- as.data.frame(test_features_FASTTEXT_CBOW)
test_features_FASTTEXT_CBOW[,1] <- as.factor(test_features_FASTTEXT_CBOW[,1])
test_features_FASTTEXT_CBOW[,2:ncol(test_features_FASTTEXT_CBOW)] <- sapply(test_features_FASTTEXT_CBOW[,2:ncol(test_features_FASTTEXT_CBOW)], as.numeric)
colnames(test_features_FASTTEXT_CBOW)[1] <- "cod"



list_paramsSKIP <- list(command = 'skipgram',
                        lr = 0.5,
                        dim = 50,
                        input = "products.txt",
                        output = file.path(".", 'skip_word_vectors'),
                        verbose = 2,
                        thread = 4,
                        minCount = 1,
                        wordNgrams = 3,
                        minn = 1,
                        maxn = 3,
                        dim = 50,
                        ws = 4,
                        epoch = 15
)

resSKIP <- fasttext_interface(list_paramsSKIP,
                              path_output = file.path(".", 'skip_products_log.txt'),
                              MilliSecs = 5,
                              remove_previous_file = TRUE,
                              print_process_time = TRUE)


embedding_fastText_SKIP <- read.table('skip_word_vectors.vec', skip = 1, row.names = 1)

train_features_FASTTEXT_SKIP <- sapply(train$nume, buildFastTextEmbedding, embedding_fastText_SKIP)
test_features_FASTTEXT_SKIP <- sapply(test$nume, buildFastTextEmbedding, embedding_fastText_SKIP)

train_features_FASTTEXT_SKIP <- t(rbind(train$cod, train_features_FASTTEXT_SKIP))
test_features_FASTTEXT_SKIP <- t(rbind(test$cod, test_features_FASTTEXT_SKIP))

### prepare data for ML
train_features_FASTTEXT_SKIP <- as.data.frame(train_features_FASTTEXT_SKIP)
train_features_FASTTEXT_SKIP[,1] <- as.factor(train_features_FASTTEXT_SKIP[,1])
train_features_FASTTEXT_SKIP[,2:ncol(train_features_FASTTEXT_SKIP)] <- sapply(train_features_FASTTEXT_SKIP[,2:ncol(train_features_FASTTEXT_SKIP)], as.numeric)
colnames(train_features_FASTTEXT_SKIP)[1] <- "cod"

test_features_FASTTEXT_SKIP <- as.data.frame(test_features_FASTTEXT_SKIP)
test_features_FASTTEXT_SKIP[,1] <- as.factor(test_features_FASTTEXT_SKIP[,1])
test_features_FASTTEXT_SKIP[,2:ncol(test_features_FASTTEXT_SKIP)] <- sapply(test_features_FASTTEXT_SKIP[,2:ncol(test_features_FASTTEXT_SKIP)], as.numeric)
colnames(test_features_FASTTEXT_SKIP)[1] <- "cod"



#### 6. Glove                                       ####
p <- readLines('products.txt', n = -1, warn = FALSE)
tokens <- space_tokenizer(p)
it = itoken(tokens, progressbar = FALSE)
vocab <- create_vocabulary(it)
vectorizer <- vocab_vectorizer(vocab)
tcm <- create_tcm(it, vectorizer, skip_grams_window = 4L)
glove = GlobalVectors$new(rank = 50, x_max = 10)
wv_main = glove$fit_transform(tcm, n_iter = 15, convergence_tol = 0.01, n_threads = 8)
wv_context = glove$components
word_vectors = wv_main + t(wv_context)

train_features_GLOVE <- sapply(train$nume, buildEmbedding, word_vectors)
test_features_GLOVE <- sapply(test$nume, buildEmbedding, word_vectors)
train_features_GLOVE <- t(rbind(train$cod, train_features_GLOVE))
test_features_GLOVE <- t(rbind(test$cod, test_features_GLOVE))


### prepare data for ML

train_features_GLOVE <- as.data.frame(train_features_GLOVE)
train_features_GLOVE[,1] <- as.factor(train_features_GLOVE[,1])
train_features_GLOVE[,2:ncol(train_features_GLOVE)] <- sapply(train_features_GLOVE[,2:ncol(train_features_GLOVE)], as.numeric)
colnames(train_features_GLOVE)[1] <- "cod"

test_features_GLOVE <- as.data.frame(test_features_GLOVE)
test_features_GLOVE[,1] <- as.factor(test_features_GLOVE[,1])
test_features_GLOVE[,2:ncol(test_features_GLOVE)] <- sapply(test_features_GLOVE[,2:ncol(test_features_GLOVE)], as.numeric)
colnames(test_features_GLOVE)[1] <- "cod"

#### Classification starts here #####

coln<-paste0("V", 2:3001)
coln <- c("cod", coln)
colnames(train_cf_features_CV) <-coln
colnames(train_tf_features_TF_IDF) <-coln

colnames(test_cf_features_CV) <-coln
colnames(test_tf_features_TF_IDF) <-coln

coln<-paste0("V", 2:51)
coln <- c("cod", coln)
colnames(train_features_W2V_CBOW_ADD) <-coln
colnames(test_features_W2V_CBOW_ADD) <-coln

colnames(train_features_W2V_CBOW_MEAN) <-coln
colnames(test_features_W2V_CBOW_MEAN) <-coln

colnames(train_features_W2V_SKIP_ADD) <-coln
colnames(test_features_W2V_SKIP_ADD) <-coln

colnames(train_features_W2V_SKIP_MEAN) <-coln
colnames(test_features_W2V_SKIP_MEAN) <-coln

colnames(train_features_FASTTEXT_CBOW) <-coln
colnames(test_features_FASTTEXT_SKIP) <-coln

colnames(train_features_GLOVE) <-coln
colnames(test_features_GLOVE) <-coln

### Logistic regression
set.seed(5678)
### 1. Count Vectorization
fitLR_CV <- glmnetUtils::glmnet(cod ~ ., data=train_cf_features_CV, family="multinomial")
predLR_CV <-  predict(fitLR_CV, newdata=test_cf_features_CV, type="class")
pred <- c()
for(i in 1:nrow(predLR_CV)) {
    pred <- c(pred, names(which.max(table(predLR_CV[i,]))))
}
pred <- as.factor(pred)
cm_LR_CV <- confusionMatrix(pred, test_cf_features_CV$cod)

r <- buildAccMeasures(trueValues = test_cf_features_CV$cod, predictedValues = pred, "Logistic regression - Count Vectorization")
r %>% knitr::kable()

### 2. TF-IDF
fitLR_TF <- glmnetUtils::glmnet(cod ~ ., data=train_tf_features_TF_IDF, family="multinomial")
predLR_TF <-  predict(fitLR_TF, newdata=test_tf_features_TF_IDF, type="class")
pred <- c()
for(i in 1:nrow(predLR_TF)) {
    pred <- c(pred, names(which.max(table(predLR_TF[i,]))))
}
pred <- as.factor(pred)
cm_LR_TF_IDF <- confusionMatrix(pred, test_tf_features_TF_IDF$cod)

r <- bind_rows(r, buildAccMeasures(trueValues = test_cf_features_CV$cod, predictedValues = pred, "Logistic regression - TF-IDF"))
r %>% knitr::kable()

### 3.1 Word2Vec CBOW - ADD

fitLR_W2V_CBOW_ADD <- glmnetUtils::glmnet(cod ~ ., data=train_features_W2V_CBOW_ADD, family="multinomial")
predLR_W2V_CBOW_ADD <-  predict(fitLR_W2V_CBOW_ADD, newdata=test_features_W2V_CBOW_ADD, type="class")
pred <- c()
for(i in 1:nrow(predLR_W2V_CBOW_ADD)) {
    pred <- c(pred, names(which.max(table(predLR_W2V_CBOW_ADD[i,]))))
}
pred <- as.factor(pred)
cm_LR_W2V_CBOW_ADD <- confusionMatrix(pred, test_features_W2V_CBOW_ADD$cod)

r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_CBOW_ADD$cod, predictedValues = pred, "Logistic regression - W2V_CBOW_ADD"))
r %>% knitr::kable()


### 3.2 Word2Vec CBOW - MEAN
fitLR_W2V_CBOW_MEAN <- glmnetUtils::glmnet(cod ~ ., data=train_features_W2V_CBOW_MEAN, family="multinomial")
predLR_W2V_CBOW_MEAN <-  predict(fitLR_W2V_CBOW_MEAN, newdata=test_features_W2V_CBOW_MEAN, type="class")
pred <- c()
for(i in 1:nrow(predLR_W2V_CBOW_MEAN)) {
    pred <- c(pred, names(which.max(table(predLR_W2V_CBOW_MEAN[i,]))))
}
pred <- as.factor(pred)
cm_LR_W2V_COBW_MEAN <- confusionMatrix(pred, test_features_W2V_CBOW_MEAN$cod)

r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_CBOW_MEAN$cod, predictedValues = pred, "Logistic regression - W2V_CBOW_MEAN"))
r %>% knitr::kable()

### 3.3 Word2Vec SKIP - ADD
fitLR_W2V_SKIP_ADD <- glmnetUtils::glmnet(cod ~ ., data=train_features_W2V_SKIP_ADD, family="multinomial")
predLR_W2V_SKIP_ADD <-  predict(fitLR_W2V_SKIP_ADD, newdata=test_features_W2V_SKIP_ADD, type="class")
pred <- c()
for(i in 1:nrow(predLR_W2V_SKIP_ADD)) {
    pred <- c(pred, names(which.max(table(predLR_W2V_SKIP_ADD[i,]))))
}
pred <- as.factor(pred)
cm_LR_W2V_SKIP_ADD <- confusionMatrix(pred, test_features_W2V_SKIP_ADD$cod)

r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_SKIP_ADD$cod, predictedValues = pred, "Logistic regression - W2V_SKIP_ADD"))
r %>% knitr::kable()

### 3.4 Word2Vec SKIP - MEAN
fitLR_W2V_SKIP_MEAN <- glmnetUtils::glmnet(cod ~ ., data=train_features_W2V_SKIP_MEAN, family="multinomial")
predLR_W2V_SKIP_MEAN <-  predict(fitLR_W2V_SKIP_MEAN, newdata=test_features_W2V_SKIP_MEAN, type="class")
pred <- c()
for(i in 1:nrow(predLR_W2V_SKIP_MEAN)) {
    pred <- c(pred, names(which.max(table(predLR_W2V_SKIP_MEAN[i,]))))
}
pred <- as.factor(pred)
cm_LR_W2V_SKIP_MEAM <- confusionMatrix(pred, test_features_W2V_SKIP_MEAN$cod)

r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_SKIP_MEAN$cod, predictedValues = pred, "Logistic regression - W2V_SKIP_MEAN"))
r %>% knitr::kable()


### 4.1 FASTTEXT CBOW
fitLR_FASTTEXT_CBOW <- glmnetUtils::glmnet(cod ~ ., data=train_features_FASTTEXT_CBOW, family="multinomial")
predLR_FASTTEXT_CBOW <-  predict(fitLR_FASTTEXT_CBOW, newdata=test_features_FASTTEXT_CBOW, type="class")
pred <- c()
for(i in 1:nrow(predLR_FASTTEXT_CBOW)) {
    pred <- c(pred, names(which.max(table(predLR_FASTTEXT_CBOW[i,]))))
}
pred <- as.factor(pred)
cm_LR_FASTTEXT_CBOW <- confusionMatrix(pred, test_features_FASTTEXT_CBOW$cod)

r <- bind_rows(r, buildAccMeasures(trueValues = test_features_FASTTEXT_CBOW$cod, predictedValues = pred, "Logistic regression - FASTTEXT_CBOW"))
r %>% knitr::kable()

### 4.2 FASTTEXT SKIP
fitLR_FASTTEXT_SKIP <- glmnetUtils::glmnet(cod ~ ., data=train_features_FASTTEXT_SKIP, family="multinomial")
predLR_FASTTEXT_SKIP <-  predict(fitLR_FASTTEXT_SKIP, newdata=test_features_FASTTEXT_SKIP, type="class")
pred <- c()
for(i in 1:nrow(predLR_FASTTEXT_SKIP)) {
    pred <- c(pred, names(which.max(table(predLR_FASTTEXT_SKIP[i,]))))
}
pred <- as.factor(pred)
cm_LR_FASTTEXT_SKIP <- confusionMatrix(pred, test_features_FASTTEXT_SKIP$cod)

r <- bind_rows(r, buildAccMeasures(trueValues = test_features_FASTTEXT_SKIP$cod, predictedValues = pred, "Logistic regression - FASTTEXT_SKIP"))
r %>% knitr::kable()


### 5. GLOVE
fitLR_GLOVE <- glmnetUtils::glmnet(cod ~ ., data=train_features_GLOVE, family="multinomial")
predLR_GLOVE <-  predict(fitLR_GLOVE, newdata=test_features_GLOVE, type="class")
pred <- c()
for(i in 1:nrow(predLR_GLOVE)) {
    pred <- c(pred, names(which.max(table(predLR_GLOVE[i,]))))
}
pred <- as.factor(pred)
cm_LR_GLOVE <- confusionMatrix(pred, test_features_GLOVE$cod)

r <- bind_rows(r, buildAccMeasures(trueValues = test_features_GLOVE$cod, predictedValues = pred, "Logistic regression - GLOVE"))
r %>% knitr::kable()


tr_control <- trainControl(method="repeatedcv", number=10, repeats=2)
#####################
# Naive Bayes
#####################
### 1. CV

fitNB_CV <- multinomial_naive_bayes(x = train_cf_features_CV[,2:ncol(train_cf_features_CV)], y = train_cf_features_CV$cod)
predNB_CV <- predict(fitNB_CV, newdata = as.matrix(test_cf_features_CV[,2:ncol(test_cf_features_CV)]))
cm_NB_CV <- confusionMatrix(predNB_CV, test_cf_features_CV$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_cf_features_CV$cod, predictedValues = predNB_CV, "Naive Bayes multinomial - Count Vectorization"))
r %>% knitr::kable()

### 2. TF-IDF
fitNB_TF_IDF <- multinomial_naive_bayes(x = train_tf_features_TF_IDF[,2:ncol(train_tf_features_TF_IDF)], y = train_tf_features_TF_IDF$cod)
predNB_TF_IDF<-predict(fitNB_TF_IDF, newdata = as.matrix(test_tf_features_TF_IDF[,2:ncol(test_tf_features_TF_IDF)]))
cm_NB_TF_IDF <- confusionMatrix(predNB_TF_IDF, test_tf_features_TF_IDF$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_tf_features_TF_IDF$cod, predictedValues = predNB_TF_IDF, "Naive Bayes multinomial - TF IDF"))
r %>% knitr::kable()

#################
# Clasification Tree - CART
#################

fitCART_CV <- rpart(cod ~ ., data = train_cf_features_CV, parms = list(split='gini'), method = "class")
predCART_CV<-predict(fitCART_CV, newdata = test_cf_features_CV, type = 'class')
cm_CART_CV <- confusionMatrix(predCART_CV, test_cf_features_CV$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_cf_features_CV$cod, predictedValues = predCART_CV, "CART Gini - Count Vectorization"))
r %>% knitr::kable()

### 2. TF-IDF
fitCART_TF_IDF <- rpart(cod ~ ., data = train_tf_features_TF_IDF, parms = list(split='gini'), method = "class")
predCART_TF_IDF<-predict(fitCART_TF_IDF, newdata = test_tf_features_TF_IDF, type = 'class')
cm_CART_TF_IDF <- confusionMatrix(predCART_TF_IDF, test_tf_features_TF_IDF$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_tf_features_TF_IDF$cod, predictedValues = predCART_TF_IDF, "CART Gini - TF IDF"))
r %>% knitr::kable()

### 3.1 W2V CBOW ADD
fitCART_W2V_CBOW_ADD <- rpart(cod ~ ., data = train_features_W2V_CBOW_ADD, parms = list(split='gini'), method = "class")
predCART_W2V_CBOW_ADD<-predict(fitCART_W2V_CBOW_ADD, newdata = test_features_W2V_CBOW_ADD, type = 'class')
cm_CART_W2V_CBOW_ADD <- confusionMatrix(predCART_W2V_CBOW_ADD, test_features_W2V_CBOW_ADD$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_features_W2V_CBOW_ADD$cod, predictedValues = predCART_W2V_CBOW_ADD, "CART Gini - W2V CBOW ADD"))
r %>% knitr::kable()

### 3.2 W2V CBOW MEAN
fitCART_W2V_CBOW_MEAN <- rpart(cod ~ ., data=train_features_W2V_CBOW_MEAN, parms = list(split='gini'), method = "class")
predCART_W2V_CBOW_MEAN<-predict(fitCART_W2V_CBOW_MEAN, newdata = test_features_W2V_CBOW_MEAN, type = 'class')
cm_CART_W2V_CBOW_MEAN <- confusionMatrix(predCART_W2V_CBOW_MEAN, test_features_W2V_CBOW_MEAN$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_features_W2V_CBOW_MEAN$cod, predictedValues = predCART_W2V_CBOW_MEAN, "CART Gini - W2V CBOW MEAN"))
r %>% knitr::kable()


### 3.3 W2V SKIP ADD
fitCART_W2V_SKIP_ADD <- rpart(cod ~ ., data=train_features_W2V_SKIP_ADD, parms = list(split='gini'), method = "class")
predCART_W2V_SKIP_ADD<-predict(fitCART_W2V_SKIP_ADD, newdata = test_features_W2V_SKIP_ADD, type = 'class')
cm_CART_W2V_SKIP_ADD <- confusionMatrix(predCART_W2V_SKIP_ADD, test_features_W2V_SKIP_ADD$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_features_W2V_SKIP_ADD$cod, predictedValues = predCART_W2V_SKIP_ADD, "CART Gini - W2V SKIP ADD"))
r %>% knitr::kable()

### 3.4 W2v SKIP MEAN
fitCART_W2V_SKIP_MEAN <- rpart(cod ~ ., data=train_features_W2V_SKIP_MEAN, parms = list(split='gini'), method = "class")
predCART_W2V_SKIP_MEAN<-predict(fitCART_W2V_SKIP_MEAN, newdata = test_features_W2V_SKIP_MEAN, type = 'class')
cm_CART_W2V_SKIP_MEAN <- confusionMatrix(predCART_W2V_SKIP_MEAN, test_features_W2V_SKIP_MEAN$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_features_W2V_SKIP_MEAN$cod, predictedValues = predCART_W2V_SKIP_MEAN, "CART Gini - W2V SKIP MEAN"))
r %>% knitr::kable()

### 4.1 FASTTEXT CBOW
fitCART_FASTTEXT_CBOW <- rpart(cod ~ ., data=train_features_FASTTEXT_CBOW, parms = list(split='gini'), method = "class")
predCART_FASTTEXT_CBOW<-predict(fitCART_FASTTEXT_CBOW, newdata = test_features_FASTTEXT_CBOW, type = 'class')
cm_CART_FASTTEXT_CBOW <- confusionMatrix(predCART_FASTTEXT_CBOW, test_features_FASTTEXT_CBOW$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_features_FASTTEXT_CBOW$cod, predictedValues = predCART_FASTTEXT_CBOW, "CART Gini - FASTTEXT CBOW"))
r %>% knitr::kable()

### 4.2 FASTTEXT SKIP
fitCART_FASTTEXT_SKIP <- rpart(cod ~ ., data=train_features_FASTTEXT_SKIP, parms = list(split='gini'), method = "class")
predCART_FASTTEXT_SKIP<-predict(fitCART_FASTTEXT_SKIP, newdata = test_features_FASTTEXT_SKIP, type = 'class')
cm_CART_FASTTEXT_SKIP <- confusionMatrix(predCART_FASTTEXT_SKIP, test_features_FASTTEXT_SKIP$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_features_FASTTEXT_SKIP$cod, predictedValues = predCART_FASTTEXT_SKIP, "CART Gini - FASTTEXT SKIP"))
r %>% knitr::kable()

### 5. GLOVE
fitCART_GLOVE <- rpart(cod ~ ., data=train_features_GLOVE, parms = list(split='gini'), method = "class")
predCART_GLOVE<-predict(fitCART_GLOVE, newdata = test_features_GLOVE, type = 'class')
cm_CART_GLOVE <- confusionMatrix(predCART_GLOVE, test_features_GLOVE$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_features_GLOVE$cod, predictedValues = predCART_GLOVE, "CART Gini - GLOVE"))
r %>% knitr::kable()
###################


#################
# Clasification Tree - CART - information
#################

fitCARTi_CV <- rpart(cod ~ ., data = train_cf_features_CV, parms = list(split='information'), method = "class")
predCARTi_CV<-predict(fitCARTi_CV, newdata = test_cf_features_CV, type = 'class')
cm_CARTi_CV <- confusionMatrix(predCARTi_CV, test_cf_features_CV$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_cf_features_CV$cod, predictedValues = predCARTi_CV, "CART information - Count Vectorization"))
r %>% knitr::kable()

### 2. TF-IDF
fitCARTi_TF_IDF <- rpart(cod ~ ., data = train_tf_features_TF_IDF, parms = list(split='information'), method = "class")
predCARTi_TF_IDF<-predict(fitCARTi_TF_IDF, newdata = test_tf_features_TF_IDF, type = 'class')
cm_CARTi_TF_IDF <- confusionMatrix(predCARTi_TF_IDF, test_tf_features_TF_IDF$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_tf_features_TF_IDF$cod, predictedValues = predCARTi_TF_IDF, "CART information - TF IDF"))
r %>% knitr::kable()

### 3.1 W2V CBOW ADD
fitCARTi_W2V_CBOW_ADD <- rpart(cod ~ ., data = train_features_W2V_CBOW_ADD, parms = list(split='information'), method = "class")
predCARTi_W2V_CBOW_ADD<-predict(fitCARTi_W2V_CBOW_ADD, newdata = test_features_W2V_CBOW_ADD, type = 'class')
cm_CARTi_W2V_CBOW_ADD <- confusionMatrix(predCARTi_W2V_CBOW_ADD, test_features_W2V_CBOW_ADD$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_features_W2V_CBOW_ADD$cod, predictedValues = predCARTi_W2V_CBOW_ADD, "CART information - W2V CBOW ADD"))
r %>% knitr::kable()

### 3.2 W2V CBOW MEAN
fitCARTi_W2V_CBOW_MEAN <- rpart(cod ~ ., data=train_features_W2V_CBOW_MEAN, parms = list(split='information'), method = "class")
predCARTi_W2V_CBOW_MEAN<-predict(fitCARTi_W2V_CBOW_MEAN, newdata = test_features_W2V_CBOW_MEAN, type = 'class')
cm_CARTi_W2V_CBOW_MEAN <- confusionMatrix(predCARTi_W2V_CBOW_MEAN, test_features_W2V_CBOW_MEAN$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_features_W2V_CBOW_MEAN$cod, predictedValues = predCARTi_W2V_CBOW_MEAN, "CART information - W2V CBOW MEAN"))
r %>% knitr::kable()


### 3.3 W2V SKIP ADD
fitCARTi_W2V_SKIP_ADD <- rpart(cod ~ ., data=train_features_W2V_SKIP_ADD, parms = list(split='information'), method = "class")
predCARTi_W2V_SKIP_ADD<-predict(fitCARTi_W2V_SKIP_ADD, newdata = test_features_W2V_SKIP_ADD, type = 'class')
cm_CARTi_W2V_SKIP_ADD <- confusionMatrix(predCARTi_W2V_SKIP_ADD, test_features_W2V_SKIP_ADD$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_features_W2V_SKIP_ADD$cod, predictedValues = predCARTi_W2V_SKIP_ADD, "CART information - W2V SKIP MEAN"))
r %>% knitr::kable()

### 3.4 W2v SKIP MEAN
fitCARTi_W2V_SKIP_MEAN <- rpart(cod ~ ., data=train_features_W2V_SKIP_MEAN, parms = list(split='information'), method = "class")
predCARTi_W2V_SKIP_MEAN<-predict(fitCARTi_W2V_SKIP_MEAN, newdata = test_features_W2V_SKIP_MEAN, type = 'class')
cm_CARTi_W2V_SKIP_MEAN <- confusionMatrix(predCARTi_W2V_SKIP_MEAN, test_features_W2V_SKIP_MEAN$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_features_W2V_SKIP_MEAN$cod, predictedValues = predCARTi_W2V_SKIP_MEAN, "CART information - W2V SKIP MEAN"))
r %>% knitr::kable()

### 4.1 FASTTEXT CBOW
fitCARTi_FASTTEXT_CBOW <- rpart(cod ~ ., data=train_features_FASTTEXT_CBOW, parms = list(split='information'), method = "class")
predCARTi_FASTTEXT_CBOW<-predict(fitCARTi_FASTTEXT_CBOW, newdata = test_features_FASTTEXT_CBOW, type = 'class')
cm_CARTi_FASTTEXT_CBOW <- confusionMatrix(predCARTi_FASTTEXT_CBOW, test_features_FASTTEXT_CBOW$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_features_FASTTEXT_CBOW$cod, predictedValues = predCARTi_FASTTEXT_CBOW, "CART information - FASTTEXT CBOW"))
r %>% knitr::kable()

### 4.2 FASTTEXT SKIP
fitCARTi_FASTTEXT_SKIP <- rpart(cod ~ ., data=train_features_FASTTEXT_SKIP, parms = list(split='information'), method = "class")
predCARTi_FASTTEXT_SKIP<-predict(fitCARTi_FASTTEXT_SKIP, newdata = test_features_FASTTEXT_SKIP, type = 'class')
cm_CARTi_FASTTEXT_SKIP <- confusionMatrix(predCARTi_FASTTEXT_SKIP, test_features_FASTTEXT_SKIP$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_features_FASTTEXT_SKIP$cod, predictedValues = predCARTi_FASTTEXT_SKIP, "CART information - FASTTEXT SKIP"))
r %>% knitr::kable()

### 5. GLOVE
fitCARTi_GLOVE <- rpart(cod ~ ., data=train_features_GLOVE, parms = list(split='information'), method = "class")
predCARTi_GLOVE<-predict(fitCARTi_GLOVE, newdata = test_features_GLOVE, type = 'class')
cm_CARTi_GLOVE <- confusionMatrix(predCARTi_GLOVE, test_features_GLOVE$cod)
r<-bind_rows(r,buildAccMeasures(trueValues = test_features_GLOVE$cod, predictedValues = predCARTi_GLOVE, "CART information - GLOVE"))
r %>% knitr::kable()
###################


###################
# Bagged CART
####################

### 1. Count Vectorization
fitBCART_CV <- train(cod ~ ., data=train_cf_features_CV, method="treebag", metric='Accuracy', trControl=tr_control)
predBCART_CV <- predict(fitBCART_CV, newdata=test_cf_features_CV)
cm_BCART_CV <- confusionMatrix(predBCART_CV, test_cf_features_CV$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_cf_features_CV$cod, predictedValues = predBCART_CV, "Bagged CART - Count Vectorization"))
r %>% knitr::kable()

### 2. TF-IDF
fitBCART_TF_IDF <- train(cod ~ ., data=train_tf_features_TF_IDF, method="treebag", metric='Accuracy', trControl=tr_control)
predBCART_TF_IDF <- predict(fitBCART_TF_IDF, newdata=test_tf_features_TF_IDF)
cm_BCART_TF_IDF <- confusionMatrix(predBCART_TF_IDF, test_tf_features_TF_IDF$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_tf_features_TF_IDF$cod, predictedValues = predBCART_TF_IDF, "Bagged CART - TF IDF"))
r %>% knitr::kable()

### 3.1 W2v CBOW ADD
fitBCART_W2V_CBOW_ADD <- train(cod ~ ., data=train_features_W2V_CBOW_ADD, method="treebag", metric='Accuracy', trControl=tr_control)
predBCART_W2V_CBOW_ADD <- predict(fitBCART_W2V_CBOW_ADD, newdata=test_features_W2V_CBOW_ADD)
cm_BCART_W2V_CBOW_ADD <- confusionMatrix(predBCART_W2V_CBOW_ADD, test_features_W2V_CBOW_ADD$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_CBOW_ADD$cod, predictedValues = predBCART_W2V_CBOW_ADD, "Bagged CART - W2V CBOW ADD"))
r %>% knitr::kable()

### 3.2 W2v CBOW MEAN
fitBCART_W2V_CBOW_MEAN <- train(cod ~ ., data=train_features_W2V_CBOW_MEAN, method="treebag", metric='Accuracy', trControl=tr_control)
predBCART_W2V_CBOW_MEAN <- predict(fitBCART_W2V_CBOW_MEAN, newdata=test_features_W2V_CBOW_MEAN)
cm_BCART_W2V_CBOW_MEAN <- confusionMatrix(predBCART_W2V_CBOW_MEAN, test_features_W2V_CBOW_MEAN$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_CBOW_MEAN$cod, predictedValues = predBCART_W2V_CBOW_MEAN, "Bagged CART - W2V CBOW MEAN"))
r %>% knitr::kable()

### 3.3 W2V SKIP ADD
fitBCART_W2V_SKIP_ADD <- train(cod ~ ., data=train_features_W2V_SKIP_ADD, method="treebag", metric='Accuracy', trControl=tr_control)
predBCART_W2V_SKIP_ADD <- predict(fitBCART_W2V_SKIP_ADD, newdata=test_features_W2V_SKIP_ADD)
cm_BCART_W2V_SKIP_ADD <- confusionMatrix(predBCART_W2V_SKIP_ADD, test_features_W2V_SKIP_ADD$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_SKIP_ADD$cod, predictedValues = predBCART_W2V_SKIP_ADD, "Bagged CART - W2V SKIP ADD"))
r %>% knitr::kable()

### 3.4 W2V SKIP MEAN
fitBCART_W2V_SKIP_MEAN <- train(cod ~ ., data=train_features_W2V_SKIP_MEAN, method="treebag", metric='Accuracy', trControl=tr_control)
predBCART_W2V_SKIP_MEAN <- predict(fitBCART_W2V_SKIP_MEAN, newdata=test_features_W2V_SKIP_MEAN)
cm_BCART_W2V_SKIP_MEAN <- confusionMatrix(predBCART_W2V_SKIP_MEAN, test_features_W2V_SKIP_MEAN$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_SKIP_MEAN$cod, predictedValues = predBCART_W2V_SKIP_MEAN, "Bagged CART - W2V SKIP MEAN"))
r %>% knitr::kable()


### 4.1 FASTTEXT CBOW
fitBCART_FASTTEXT_CBOW <- train(cod ~ ., data=train_features_FASTTEXT_CBOW, method="treebag", metric='Accuracy', trControl=tr_control)
predBCART_FASTTEXT_CBOW <- predict(fitBCART_FASTTEXT_CBOW, newdata=test_features_FASTTEXT_CBOW)
cm_BCART_FASTTEXT_CBOW <- confusionMatrix(predBCART_FASTTEXT_CBOW, test_features_FASTTEXT_CBOW$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_FASTTEXT_CBOW$cod, predictedValues = predBCART_FASTTEXT_CBOW, "Bagged CART - FASTTEXT CBOW"))
r %>% knitr::kable()


### 4.2 FASTTEXT SKIP
fitBCART_FASTTEXT_SKIP <- train(cod ~ ., data=train_features_FASTTEXT_SKIP, method="treebag", metric='Accuracy', trControl=tr_control)
predBCART_FASTTEXT_SKIP <- predict(fitBCART_FASTTEXT_SKIP, newdata=test_features_FASTTEXT_SKIP)
cm_BCART_FASTTEXT_SKIP <- confusionMatrix(predBCART_FASTTEXT_SKIP, test_features_FASTTEXT_SKIP$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_FASTTEXT_SKIP$cod, predictedValues = predBCART_FASTTEXT_SKIP, "Bagged CART - FASTTEXT SKIP"))
r %>% knitr::kable()


### 5. GLOVE
fitBCART_GLOVE <- train(cod ~ ., data=train_features_GLOVE, method="treebag", metric='Accuracy', trControl=tr_control)
predBCART_GLOVE <- predict(fitBCART_GLOVE, newdata=test_features_GLOVE)
cm_BCART_GLOVE <- confusionMatrix(predBCART_GLOVE, test_features_GLOVE$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_GLOVE$cod, predictedValues = predBCART_GLOVE, "Bagged CART - GLOVE"))
r %>% knitr::kable()




####################
# C4.5
####################

### 1. Count Vectorization
fitC45_CV <- train(cod ~ ., data=train_cf_features_CV, method="J48", metric='Accuracy',trControl=tr_control)
predC45_CV <- predict(fitC45_CV, newdata=test_cf_features_CV)

cm_C45_CV <- confusionMatrix(predC45_CV, test_cf_features_CV$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_cf_features_CV$cod, predictedValues = predC45_CV, "C4.5 - Count Vectorization"))

### 2. TF IDF
fitC45_TF_IDF <- train(cod ~ ., data=train_tf_features_TF_IDF, method="J48", metric='Accuracy',trControl=tr_control)
predC45_TF_IDF <- predict(fitC45_TF_IDF, newdata=test_tf_features_TF_IDF)
cm_C45_TF_IDF <- confusionMatrix(predC45_TF_IDF, test_tf_features_TF_IDF$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_tf_features_TF_IDF$cod, predictedValues = predC45_TF_IDF, "C4.5 - TF IDF"))


### 3.1. W2V CBOW ADD
fitC45_W2V_CBOW_ADD <- train(cod ~ ., data=train_features_W2V_CBOW_ADD, method="J48", metric='Accuracy',trControl=tr_control)
predC45_W2V_CBOW_ADD <- predict(fitC45_W2V_CBOW_ADD, newdata=test_features_W2V_CBOW_ADD)
cm_C45_W2v_CBOW_ADD <- confusionMatrix(predC45_W2V_CBOW_ADD, test_features_W2V_CBOW_ADD$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_CBOW_ADD$cod, predictedValues = predC45_W2V_CBOW_ADD, "C4.5 - W2V CBOW ADD"))
r %>% knitr::kable()


### 3.2. W2V CBOW MEAN
fitC45_W2V_CBOW_MEAN <- train(cod ~ ., data=train_features_W2V_CBOW_MEAN, method="J48", metric='Accuracy',trControl=tr_control)
predC45_W2V_CBOW_MEAN <- predict(fitC45_W2V_CBOW_MEAN, newdata=test_features_W2V_CBOW_MEAN)
cm_C45_W2v_CBOW_MEAN <- confusionMatrix(predC45_W2V_CBOW_MEAN, test_features_W2V_CBOW_MEAN$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_CBOW_MEAN$cod, predictedValues = predC45_W2V_CBOW_MEAN, "C4.5 - W2V CBOW MEAN"))

### 3.3. W2V SKIP ADD
fitC45_W2V_SKIP_ADD <- train(cod ~ ., data=train_features_W2V_SKIP_ADD, method="J48", metric='Accuracy',trControl=tr_control)
#fitC45_W2V_SKIP_ADD <- J48(cod ~ ., data=train_features_W2V_SKIP_ADD)
predC45_W2V_SKIP_ADD <- predict(fitC45_W2V_SKIP_ADD, newdata=test_features_W2V_SKIP_ADD)
cm_C45_W2v_SKIP_ADD <- confusionMatrix(predC45_W2V_SKIP_ADD, test_features_W2V_SKIP_ADD$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_SKIP_ADD$cod, predictedValues = predC45_W2V_SKIP_ADD, "C4.5 - W2V SKIP ADD"))
r %>% knitr::kable()


### 3.4. W2V SKIP MEAN
fitC45_W2V_SKIP_MEAN <- train(cod ~ ., data=train_features_W2V_SKIP_MEAN, method="J48", metric='Accuracy',trControl=tr_control)
#fitC45_W2V_SKIP_MEAN <- J48(cod ~ ., data=train_features_W2V_SKIP_MEAN)
predC45_W2V_SKIP_MEAN <- predict(fitC45_W2V_SKIP_MEAN, newdata=test_features_W2V_SKIP_MEAN)
cm_C45_W2v_SKIP_MEAN <- confusionMatrix(predC45_W2V_SKIP_MEAN, test_features_W2V_SKIP_MEAN$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_SKIP_MEAN$cod, predictedValues = predC45_W2V_SKIP_MEAN, "C4.5 - W2V SKIP MEAN"))


### 4.1 FASTTEXT CBOW
fitC45_FASTTEXT_CBOW <- train(cod ~ ., data=train_features_FASTTEXT_CBOW, method="J48", metric='Accuracy',trControl=tr_control)
predC45_FASTTEXT_CBOW <- predict(fitC45_FASTTEXT_CBOW, newdata=test_features_FASTTEXT_CBOW)
cm_C45_FASTTEXT_CBOW <- confusionMatrix(predC45_FASTTEXT_CBOW, test_features_FASTTEXT_CBOW$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_FASTTEXT_CBOW$cod, predictedValues = predC45_FASTTEXT_CBOW, "C4.5 - FASTTEXT CBOW"))

### 4.2 FASTTEXT SKIP
fitC45_FASTTEXT_SKIP <- train(cod ~ ., data=train_features_FASTTEXT_SKIP, method="J48", metric='Accuracy',trControl=tr_control)
predC45_FASTTEXT_SKIP <- predict(fitC45_FASTTEXT_SKIP, newdata=test_features_FASTTEXT_SKIP)
cm_C45_FASTTEXT_SKIP <- confusionMatrix(predC45_FASTTEXT_SKIP, test_features_FASTTEXT_SKIP$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_FASTTEXT_SKIP$cod, predictedValues = predC45_FASTTEXT_SKIP, "C4.5 - FASTTEXT SKIP"))

### 5 GLOVE
fitC45_GLOVE <- train(cod ~ ., data=train_features_GLOVE, method="J48", metric='Accuracy',trControl=tr_control)
predC45_GLOVE <- predict(fitC45_GLOVE, newdata=test_features_GLOVE)
cm_C45_GLOVE <- confusionMatrix(predC45_GLOVE, test_features_GLOVE$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_GLOVE$cod, predictedValues = predC45_GLOVE, "C4.5 - GLOVE"))


####################
# C50
####################

### 1. Count Vectorization
fitC50_CV <- train(cod ~ ., data=train_cf_features_CV, method="C5.0Tree", metric='Accuracy', trControl=tr_control)
predC50_CV <- predict(fitC50_CV, newdata=test_cf_features_CV)
cm_C50_CV <- confusionMatrix(predC50_CV, test_cf_features_CV$cod)
r <- bind_rows(r,buildAccMeasures(trueValues = test_cf_features_CV$cod, predictedValues = predC50_CV, "C50 - Count Vectorization"))
r %>% knitr::kable()

### 2. TF- IDF
fitC50_TF_IDF <- train(cod ~ ., data=train_tf_features_TF_IDF, method="C5.0Tree", metric='Accuracy', trControl=tr_control)
predC50_TF_IDF <- predict(fitC50_TF_IDF, newdata=test_tf_features_TF_IDF)
cm_C50_TF_IDF <- confusionMatrix(predC50_TF_IDF, test_tf_features_TF_IDF$cod)
r <- bind_rows(r,buildAccMeasures(trueValues = test_tf_features_TF_IDF$cod, predictedValues = predC50_TF_IDF, "C50 - TF IDF"))
r %>% knitr::kable()

### 3.1 W2V CBOW ADD
fitC50_W2V_CBOW_ADD <- train(cod ~ ., data=train_features_W2V_CBOW_ADD, method="C5.0Tree", metric='Accuracy', trControl=tr_control)
predC50_W2V_CBOW_ADD <- predict(fitC50_W2V_CBOW_ADD, newdata=test_features_W2V_CBOW_ADD)
cm_C50_W2V_CBOW_ADD <- confusionMatrix(predC50_W2V_CBOW_ADD, test_features_W2V_CBOW_ADD$cod)
r <- bind_rows(r,buildAccMeasures(trueValues = test_features_W2V_CBOW_ADD$cod, predictedValues = predC50_W2V_CBOW_ADD, "C50 - W2V CBOW ADD"))
r %>% knitr::kable()

### 3.2 W2V CBOW MEAN
fitC50_W2V_CBOW_MEAN <- train(cod ~ ., data=train_features_W2V_CBOW_MEAN, method="C5.0Tree", metric='Accuracy', trControl=tr_control)
predC50_W2V_CBOW_MEAN <- predict(fitC50_W2V_CBOW_MEAN, newdata=test_features_W2V_CBOW_MEAN)
cm_C50_W2V_CBOW_MEAN <- confusionMatrix(predC50_W2V_CBOW_MEAN, test_features_W2V_CBOW_MEAN$cod)
r <- bind_rows(r,buildAccMeasures(trueValues = test_features_W2V_CBOW_MEAN$cod, predictedValues = predC50_W2V_CBOW_MEAN, "C50 - W2V CBOW MEAN"))
r %>% knitr::kable()


### 3.3 W2V SKIP ADD
fitC50_W2V_SKIP_ADD <- train(cod ~ ., data=train_features_W2V_SKIP_ADD, method="C5.0Tree", metric='Accuracy', trControl=tr_control)
predC50_W2V_SKIP_ADD <- predict(fitC50_W2V_SKIP_ADD, newdata=test_features_W2V_SKIP_ADD)
cm_C50_W2V_SKIP_ADD <- confusionMatrix(predC50_W2V_SKIP_ADD, test_features_W2V_SKIP_ADD$cod)
r <- bind_rows(r,buildAccMeasures(trueValues = test_features_W2V_SKIP_ADD$cod, predictedValues = predC50_W2V_SKIP_ADD, "C50 - W2V SKIP ADD"))
r %>% knitr::kable()

### 3.4 W2V SKIP MEAN
fitC50_W2V_SKIP_MEAN <- train(cod ~ ., data=train_features_W2V_SKIP_MEAN, method="C5.0Tree", metric='Accuracy', trControl=tr_control)
predC50_W2V_SKIP_MEAN <- predict(fitC50_W2V_SKIP_MEAN, newdata=test_features_W2V_SKIP_MEAN)
cm_C50_W2V_SKIP_MEAN <- confusionMatrix(predC50_W2V_SKIP_MEAN, test_features_W2V_SKIP_MEAN$cod)
r <- bind_rows(r,buildAccMeasures(trueValues = test_features_W2V_SKIP_MEAN$cod, predictedValues = predC50_W2V_SKIP_MEAN, "C50 - W2V SKIP MEAN"))
r %>% knitr::kable()


### 4.1 FASTTEXT CBOW
fitC50_FASTTEXT_CBOW <- train(cod ~ ., data=train_features_FASTTEXT_CBOW, method="C5.0Tree", metric='Accuracy', trControl=tr_control)
predC50_FASTTEXT_CBOW <- predict(fitC50_FASTTEXT_CBOW, newdata=test_features_FASTTEXT_CBOW)
cm_C50_FASTTEXT_CBOW <- confusionMatrix(predC50_FASTTEXT_CBOW, test_features_FASTTEXT_CBOW$cod)
r <- bind_rows(r,buildAccMeasures(trueValues = test_features_FASTTEXT_CBOW$cod, predictedValues = predC50_FASTTEXT_CBOW, "C50 - FASTTEXT CBOW"))
r %>% knitr::kable()

### 4.2 FASTTEXT SKIP
fitC50_FASTTEXT_SKIP <- train(cod ~ ., data=train_features_FASTTEXT_SKIP, method="C5.0Tree", metric='Accuracy', trControl=tr_control)
predC50_FASTTEXT_SKIP <- predict(fitC50_FASTTEXT_SKIP, newdata=test_features_FASTTEXT_SKIP)
cm_C50_FASTTEXT_SKIP <- confusionMatrix(predC50_FASTTEXT_SKIP, test_features_FASTTEXT_SKIP$cod)
r <- bind_rows(r,buildAccMeasures(trueValues = test_features_FASTTEXT_SKIP$cod, predictedValues = predC50_FASTTEXT_SKIP, "C50 - FASTTEXT SKIP"))
r %>% knitr::kable()

### 5. GLOVE
fitC50_GLOVE <- train(cod ~ ., data=train_features_GLOVE, method="C5.0Tree", metric='Accuracy', trControl=tr_control)
predC50_GLOVE <- predict(fitC50_GLOVE, newdata=test_features_GLOVE)
cm_C50_GLOVE <- confusionMatrix(predC50_GLOVE, test_features_GLOVE$cod)
r <- bind_rows(r,buildAccMeasures(trueValues = test_features_GLOVE$cod, predictedValues = predC50_GLOVE, "C50 - GLOVE"))
r %>% knitr::kable()




# Random Forest
###################
### 1. Count Vectorization
fitRF_CV <- train(cod ~ ., data=train_cf_features_CV, method="ranger", metric='Accuracy',trControl=tr_control)
predRF_CV <- predict(fitRF_CV, newdata=test_cf_features_CV)
cmRF_CV <- confusionMatrix(predRF_CV, data = test_cf_features_CV$cod)
r <- bind_rows(r,buildAccMeasures(trueValues = test_cf_features_CV$cod, predictedValues = predRF_CV, "Random forest optimizat - CV"))
r %>% knitr::kable()

### 2. TF IDF
fitRF_TF_IDF <- train(cod ~ ., data=train_tf_features_TF_IDF, method="ranger", metric='Accuracy', trControl=tr_control)
predRF_TF_IDF <- predict(fitRF_TF_IDF, newdata=test_tf_features_TF_IDF)
cmRF_TF_IDF <- confusionMatrix(predRF_TF_IDF, test_tf_features_TF_IDF$cod)
r <- bind_rows(r,buildAccMeasures(trueValues = test_tf_features_TF_IDF$cod, predictedValues = predRF_TF_IDF, "Random forest optimizat - TF IDF"))
r %>% knitr::kable()

### 3.1 W2V CBOW ADD
fitRF_W2V_CBOW_ADD <- train(cod ~ ., data=train_features_W2V_CBOW_ADD, method="ranger", metric='Accuracy', trControl=tr_control)
predRF_W2V_CBOW_ADD <- predict(fitRF_W2V_CBOW_ADD, newdata=test_features_W2V_CBOW_ADD)
cmRF_W2V_CBOW_ADD <- confusionMatrix(predRF_W2V_CBOW_ADD, test_features_W2V_CBOW_ADD$cod)
r <- bind_rows(r,buildAccMeasures(trueValues = test_features_W2V_CBOW_ADD$cod, predictedValues = predRF_W2V_CBOW_ADD, "Random forest optimizat - W2V CBOW ADD"))
r %>% knitr::kable()

### 3.2 W2V CBOW MEAN
fitRF_W2V_CBOW_MEAN <- train(cod ~ ., data=train_features_W2V_CBOW_MEAN, method="ranger", metric='Accuracy', trControl=tr_control)
#fitRF_W2V_CBOW_MEAN <- ranger(cod~., data = train_features_W2V_CBOW_MEAN, classification = TRUE)
predRF_W2V_CBOW_MEAN <- predict(fitRF_W2V_CBOW_MEAN, newdata=test_features_W2V_CBOW_MEAN)
cmRF_W2V_CBOW_MEAN <- confusionMatrix(predRF_W2V_CBOW_MEAN, test_features_W2V_CBOW_MEAN$cod)
r <- bind_rows(r,buildAccMeasures(trueValues = test_features_W2V_CBOW_MEAN$cod, predictedValues = predRF_W2V_CBOW_MEAN, "Random forest optimizat - W2V CBOW MEAN"))
r %>% knitr::kable()

### 3.3 W2V SKIP ADD
fitRF_W2V_SKIP_ADD <- train(cod ~ ., data=train_features_W2V_SKIP_ADD, method="ranger", metric='Accuracy',trControl=tr_control)
predRF_W2V_SKIP_ADD <- predict(fitRF_W2V_SKIP_ADD, newdata=test_features_W2V_SKIP_ADD)
cmRF_W2V_SKIP_ADD <- confusionMatrix(predRF_W2V_SKIP_ADD, test_features_W2V_SKIP_ADD$cod)
r <- bind_rows(r,buildAccMeasures(trueValues = test_features_W2V_SKIP_ADD$cod, predictedValues = predRF_W2V_SKIP_ADD, "Random forest optimizat - W2V SKIP ADD"))
r %>% knitr::kable()

### 3.4 W2V SKIP MEAN
fitRF_W2V_SKIP_MEAN <- train(cod ~ ., data=train_features_W2V_SKIP_MEAN, method="ranger", metric='Accuracy',trControl=tr_control)
predRF_W2V_SKIP_MEAN <- predict(fitRF_W2V_SKIP_MEAN, newdata=test_features_W2V_SKIP_MEAN)
cmRF_W2V_SKIP_MEAN <- confusionMatrix(predRF_W2V_SKIP_MEAN, test_features_W2V_SKIP_MEAN$cod)
r <- bind_rows(r,buildAccMeasures(trueValues = test_features_W2V_SKIP_MEAN$cod, predictedValues = predRF_W2V_SKIP_MEAN, "Random forest optimizat - W2V SKIP MEAN"))
r %>% knitr::kable()

### 4.1 FASTTEXT CBOW
fitRF_FASTTEXT_CBOW <- train(cod ~ ., data=train_features_FASTTEXT_CBOW, method="ranger", metric='Accuracy',trControl=tr_control)
predRF_FASTTEXT_CBOW <- predict(fitRF_FASTTEXT_CBOW, newdata=test_features_FASTTEXT_CBOW)
cmRF_FASTTEXT_CBOW <- confusionMatrix(predRF_FASTTEXT_CBOW, data = test_features_FASTTEXT_CBOW$cod)
r <- bind_rows(r,buildAccMeasures(trueValues = test_features_FASTTEXT_CBOW$cod, predictedValues = predRF_FASTTEXT_CBOW, "Random forest optimizat - FASTTEXT CBOW"))
r %>% knitr::kable()

### 4.2 FASTTEXT SKIP
fitRF_FASTTEXT_SKIP <- train(cod ~ ., data=train_features_FASTTEXT_SKIP, method="ranger", metric='Accuracy',trControl=tr_control)
predRF_FASTTEXT_SKIP <- predict(fitRF_FASTTEXT_SKIP, newdata=test_features_FASTTEXT_SKIP)
cmRF_FASTTEXT_SKIP <- confusionMatrix(predRF_FASTTEXT_SKIP, data = test_features_FASTTEXT_SKIP$cod)
r <- bind_rows(r,buildAccMeasures(trueValues = test_features_FASTTEXT_SKIP$cod, predictedValues = predRF_FASTTEXT_SKIP, "Random forest optimizat - FASTTEXT SKIP"))
r %>% knitr::kable()

### 5. GLOVE
fitRF_GLOVE <- train(cod ~ ., data=train_features_GLOVE, method="ranger", metric='Accuracy',trControl=tr_control)
#fitRF_GLOVE <- ranger(cod~., data = train_features_GLOVE, classification = TRUE)
predRF_GLOVE <- predict(fitRF_GLOVE, newdata=test_features_GLOVE)
cmRF_GLOVE <- confusionMatrix(predRF_GLOVE, data = test_features_GLOVE$cod)
r <- bind_rows(r,buildAccMeasures(trueValues = test_features_GLOVE$cod, predictedValues = predRF_GLOVE, "Random forest optimizat - GLOVE"))
r %>% knitr::kable()

##############################
# Support Vector Machine (SVM)
##############################
### 1. Count Vectorization
#tune.out <- tune(svm , cod~.,data=train_cf_features_CV ,kernel ="sigmoid", ranges=list(cost=c(0.01, 0.1, 0.5, 1,5,10,100) ))
tune.out <- tune(svm , cod~.,data=train_cf_features_CV ,kernel ="sigmoid", ranges=list(cost=c(1) ))
cost <- tune.out$best.parameters$cost
best_model <- tune.out$best.model
predSVMS_CV <- predict(best_model, test_cf_features_CV)
cmSVMS_CV <- confusionMatrix(predSVMS_CV, test_cf_features_CV$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_cf_features_CV$cod, predictedValues = predSVMS_CV, "SVM Sigmoid kernel CV"))
r %>% knitr::kable()

### 2. TF IDF
tune.out <- tune(svm , cod~.,data=train_tf_features_TF_IDF ,kernel ="sigmoid", ranges=list(cost=c(1) ))
cost <- tune.out$best.parameters$cost
best_model <- tune.out$best.model
predSVMS_TF_IDF <- predict(best_model, test_tf_features_TF_IDF)
cmSVMS_TF_IDF <- confusionMatrix(predSVMS_TF_IDF, test_tf_features_TF_IDF$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_tf_features_TF_IDF$cod, predictedValues = predSVMS_TF_IDF, "SVM Sigmoid kernel TFv IDF"))
r %>% knitr::kable()

### 3.1 W2V CBOW ADD
tune.out <- tune(svm , cod~.,data=train_features_W2V_CBOW_ADD ,kernel ="sigmoid", ranges=list(cost=c(1) ))
cost <- tune.out$best.parameters$cost
best_model <- tune.out$best.model
predSVMS_W2V_CBOW_ADD <- predict(best_model, test_features_W2V_CBOW_ADD)
cmSVMS_W2V_CBOW_ADD <- confusionMatrix(predSVMS_W2V_CBOW_ADD, test_features_W2V_CBOW_ADD$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_CBOW_ADD$cod, predictedValues = predSVMS_W2V_CBOW_ADD, "SVM Sigmoid kernel W2V CBOW ADD"))
r %>% knitr::kable()

### 3.2 W2V CBOW MEAN
tune.out <- tune(svm , cod~.,data=train_features_W2V_CBOW_MEAN ,kernel ="sigmoid", ranges=list(cost=c(1) ))
cost <- tune.out$best.parameters$cost
best_model <- tune.out$best.model
predSVMS_W2V_CBOW_MEAN <- predict(best_model, test_features_W2V_CBOW_MEAN)
cmSVMS_W2V_CBOW_MEAN <- confusionMatrix(predSVMS_W2V_CBOW_MEAN, test_features_W2V_CBOW_MEAN$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_CBOW_MEAN$cod, predictedValues = predSVMS_W2V_CBOW_MEAN, "SVM Sigmoid kernel W2V CBOW MEAN"))
r %>% knitr::kable()


### 3.3 W2V SKIP ADD
tune.out <- tune(svm , cod~.,data=train_features_W2V_SKIP_ADD ,kernel ="sigmoid", ranges=list(cost=c(1) ))
cost <- tune.out$best.parameters$cost
best_model <- tune.out$best.model
predSVMS_W2V_SKIP_ADD <- predict(best_model, test_features_W2V_SKIP_ADD)
cmSVMS_W2V_SKIP_ADD <- confusionMatrix(predSVMS_W2V_SKIP_ADD, test_features_W2V_SKIP_ADD$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_SKIP_ADD$cod, predictedValues = predSVMS_W2V_SKIP_ADD, "SVM Sigmoid kernel W2V SKIP ADD"))
r %>% knitr::kable()

### 3.4 W2V SKIP MEAN
tune.out <- tune(svm , cod~.,data=train_features_W2V_SKIP_MEAN ,kernel ="sigmoid", ranges=list(cost=c(1) ))
cost <- tune.out$best.parameters$cost
best_model <- tune.out$best.model
predSVMS_W2V_SKIP_MEAN <- predict(best_model, test_features_W2V_SKIP_MEAN)
cmSVMS_W2V_SKIP_MEAN <- confusionMatrix(predSVMS_W2V_SKIP_MEAN, test_features_W2V_SKIP_MEAN$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_SKIP_MEAN$cod, predictedValues = predSVMS_W2V_SKIP_MEAN, "SVM Sigmoid kernel W2V SKIP MEAN"))
r %>% knitr::kable()

### 4.1 FASTTEXT CBOW
tune.out <- tune(svm , cod~.,data=train_features_FASTTEXT_CBOW ,kernel ="sigmoid", ranges=list(cost=c(1) ))
cost <- tune.out$best.parameters$cost
best_model <- tune.out$best.model
predSVMS_FASTTEXT_CBOW <- predict(best_model, test_features_FASTTEXT_CBOW)
cmSVMS_FASTTEXT_CBOW <- confusionMatrix(predSVMS_FASTTEXT_CBOW, test_features_FASTTEXT_CBOW$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_FASTTEXT_CBOW$cod, predictedValues = predSVMS_FASTTEXT_CBOW, "SVM Sigmoid kernel FASTTEXT CBOW"))
r %>% knitr::kable()

### 4.2 FASTTEXT SKIP
tune.out <- tune(svm , cod~.,data=train_features_FASTTEXT_SKIP ,kernel ="sigmoid", ranges=list(cost=c(1) ))
cost <- tune.out$best.parameters$cost
best_model <- tune.out$best.model
predSVMS_FASTTEXT_SKIP <- predict(best_model, test_features_FASTTEXT_SKIP)
cmSVMS_FASTTEXT_SKIP <- confusionMatrix(predSVMS_FASTTEXT_SKIP, test_features_FASTTEXT_SKIP$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_FASTTEXT_SKIP$cod, predictedValues = predSVMS_FASTTEXT_SKIP, "SVM Sigmoid kernel FASTTEXT SKIP"))
r %>% knitr::kable()


### 5. GLOVE
tune.out <- tune(svm , cod~.,data=train_features_GLOVE ,kernel ="sigmoid", ranges=list(cost=c(1) ))
cost <- tune.out$best.parameters$cost
best_model <- tune.out$best.model
predSVMS_GLOVE <- predict(best_model, test_features_GLOVE)
cmSVMS_GLOVE <- confusionMatrix(predSVMS_GLOVE, test_features_GLOVE$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_GLOVE$cod, predictedValues = predSVMS_GLOVE, "SVM Sigmoid kernel GLOVE"))
r %>% knitr::kable()





####################
# Support Vector Machine (SVM) Radial Kernel
####################
### 1. Count Vectorization
#fitSVMR_CV <- train(cod ~ ., data=train_cf_features_CV, method="svmRadial", metric='Accuracy',
#              tuneGrid=expand.grid(C=seq(1,100,1), sigma=seq(0.05,1,0.05)),trControl=tr_control)
fitSVMR_CV <- train(cod ~ ., data=train_cf_features_CV, method="svmRadial", metric='Accuracy')
predSVMR_CV <- predict(fitSVMR_CV, newdata=test_cf_features_CV)
cmSVMR_CV <- confusionMatrix(predSVMR_CV, test_cf_features_CV$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_cf_features_CV$cod, predictedValues = predSVMR_CV, "SVM Radial kernel CV"))
r %>% knitr::kable()

### 2. TF IDF
#fitSVMR_TF_IDF <- train(cod ~ ., data=train_tf_features_TF_IDF, method="svmRadial", metric='Accuracy',
#                     tuneGrid=expand.grid(C=seq(1,100,10), sigma=seq(0.05,1,0.05)),trControl=tr_control)
fitSVMR_TF_IDF <- train(cod ~ ., data=train_tf_features_TF_IDF, method="svmRadial", metric='Accuracy')
predSVMR_TF_IDF <- predict(fitSVMR_TF_IDF, newdata=test_tf_features_TF_IDF)
cmSVMR_TF_IDF <- confusionMatrix(predSVMR_TF_IDF, test_tf_features_TF_IDF$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_tf_features_TF_IDF$cod, predictedValues = predSVMR_TF_IDF, "SVM Radial kernel TF IDF"))
r %>% knitr::kable()

### 3.1 W2V CBOW ADD
#fitSVMR_W2V_CBOW_ADD <- train(cod ~ ., data=train_features_W2V_CBOW_ADD, method="svmRadial", metric='Accuracy',
#                         ,trControl=tr_control)
fitSVMR_W2V_CBOW_ADD <- train(cod ~ ., data=train_features_W2V_CBOW_ADD, method="svmRadial", metric='Accuracy')
predSVMR_W2V_CBOW_ADD <- predict(fitSVMR_W2V_CBOW_ADD, newdata=test_features_W2V_CBOW_ADD)
cmSVMR_W2V_CBOW_ADD <- confusionMatrix(predSVMR_W2V_CBOW_ADD, test_features_W2V_CBOW_ADD$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_CBOW_ADD$cod, predictedValues = predSVMR_W2V_CBOW_ADD, "SVM Radial kernel W2V CBOW ADD"))
r %>% knitr::kable()

### 3.2 W2V CBOW MEAN
# fitSVMR_W2V_CBOW_MEAN <- train(cod ~ ., data=train_features_W2V_CBOW_MEAN, method="svmRadial", metric='Accuracy',
#                                tuneGrid=expand.grid(seq(1,100,10), sigma=seq(0.05,1,0.05)),trControl=tr_control)
fitSVMR_W2V_CBOW_MEAN <- train(cod ~ ., data=train_features_W2V_CBOW_MEAN, method="svmRadial", metric='Accuracy')
predSVMR_W2V_CBOW_MEAN <- predict(fitSVMR_W2V_CBOW_MEAN, newdata=test_features_W2V_CBOW_MEAN)
cmSVMR_W2V_CBOW_MEAN <- confusionMatrix(predSVMR_W2V_CBOW_MEAN, test_features_W2V_CBOW_MEAN$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_CBOW_MEAN$cod, predictedValues = predSVMR_W2V_CBOW_MEAN, "SVM Radial kernel W2V CBOW MEAN"))
r %>% knitr::kable()


### 3.3 W2V SKIP ADD
# fitSVMR_W2V_SKIP_ADD <- train(cod ~ ., data=train_features_W2V_SKIP_ADD, method="svmRadial", metric='Accuracy',
#                                tuneGrid=expand.grid(C=seq(1,100,10), sigma=seq(0.05,1,0.05)),trControl=tr_control)
fitSVMR_W2V_SKIP_ADD <- train(cod ~ ., data=train_features_W2V_SKIP_ADD, method="svmRadial", metric='Accuracy')
predSVMR_W2V_SKIP_ADD <- predict(fitSVMR_W2V_SKIP_ADD, newdata=test_features_W2V_SKIP_ADD)
cmSVMR_W2V_SKIP_ADD <- confusionMatrix(predSVMR_W2V_SKIP_ADD, test_features_W2V_SKIP_ADD$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_SKIP_ADD$cod, predictedValues = predSVMR_W2V_SKIP_ADD, "SVM Radial kernel W2V SKIP ADD"))
r %>% knitr::kable()

### 3.4 W2V SKIP MEAN
# fitSVMR_W2V_SKIP_MEAN <- train(cod ~ ., data=train_features_W2V_SKIP_MEAN, method="svmRadial", metric='Accuracy',
#                                 tuneGrid=expand.grid(C=seq(1,100,10), sigma=seq(0.05,1,0.05)),trControl=tr_control)
fitSVMR_W2V_SKIP_MEAN <- train(cod ~ ., data=train_features_W2V_SKIP_MEAN, method="svmRadial", metric='Accuracy')
predSVMR_W2V_SKIP_MEAN <- predict(fitSVMR_W2V_SKIP_MEAN, newdata=test_features_W2V_SKIP_MEAN)
cmSVMR_W2V_SKIP_MEAN <- confusionMatrix(predSVMR_W2V_SKIP_MEAN, test_features_W2V_SKIP_MEAN$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_SKIP_MEAN$cod, predictedValues = predSVMR_W2V_SKIP_MEAN, "SVM Radial kernel W2V SKIP MEAN"))
r %>% knitr::kable()


### 4.1 FASTTEXT CBOW
# fitSVMR_FASTTEXT_CBOW <- train(cod ~ ., data=train_features_FASTTEXT_CBOW, method="svmRadial", metric='Accuracy',
#                                 tuneGrid=expand.grid(C=seq(1,100,10), sigma=seq(0.05,1,0.05)),trControl=tr_control)
fitSVMR_FASTTEXT_CBOW <- train(cod ~ ., data=train_features_FASTTEXT_CBOW, method="svmRadial", metric='Accuracy')
predSVMR_FASTTEXT_CBOW <- predict(fitSVMR_FASTTEXT_CBOW, newdata=test_features_FASTTEXT_CBOW)
cmSVMR_FASTTEXT_CBOW <- confusionMatrix(predSVMR_FASTTEXT_CBOW, test_features_FASTTEXT_CBOW$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_FASTTEXT_CBOW$cod, predictedValues = predSVMR_FASTTEXT_CBOW, "SVM Radial kernel FASTTEXT CBOW"))
r %>% knitr::kable()

### 4.2 FASTTEXT SKIP
fitSVMR_FASTTEXT_SKIP <- train(cod ~ ., data=train_features_FASTTEXT_SKIP, method="svmRadial", metric='Accuracy')
predSVMR_FASTTEXT_SKIP <- predict(fitSVMR_FASTTEXT_SKIP, newdata=test_features_FASTTEXT_SKIP)
cmSVMR_FASTTEXT_SKIP <- confusionMatrix(predSVMR_FASTTEXT_SKIP, test_features_FASTTEXT_SKIP$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_FASTTEXT_SKIP$cod, predictedValues = predSVMR_FASTTEXT_SKIP, "SVM Radial kernel FASTTEXT SKIP"))
r %>% knitr::kable()


### 5 GLOVE
# fitSVMR_GLOVE <- train(cod ~ ., data=train_features_GLOVE, method="svmRadial", metric='Accuracy',
#                                 tuneGrid=expand.grid(C=seq(1,100,10), sigma=seq(0.05,1,0.05)),trControl=tr_control)
fitSVMR_GLOVE <- train(cod ~ ., data=train_features_GLOVE, method="svmRadial", metric='Accuracy')
predSVMR_GLOVE <- predict(fitSVMR_GLOVE, newdata=test_features_GLOVE)
cmSVMR_GLOVE <- confusionMatrix(predSVMR_GLOVE, test_features_GLOVE$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_GLOVE$cod, predictedValues = predSVMR_GLOVE, "SVM Radial kernel GLOVE"))
r %>% knitr::kable()



#################
# Neural Networks
#################
### 1. Count Vectorization
# fitNN_CV <- train(cod ~ ., data=train_cf_features_CV, method="nnet", metric='Accuracy',
#              tuneGrid=expand.grid(size = 1:12, decay = seq(0,1,0.2)),trControl=tr_control)
fitNN_CV <- train(cod ~ ., data=train_cf_features_CV, method="nnet", metric='Accuracy',trControl=tr_control)

predNN_CV <- predict(fitNN_CV, newdata=test_cf_features_CV)
cmNN_CV <- confusionMatrix(predNN_CV, test_cf_features_CV$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_cf_features_CV$cod, predictedValues = predNN_CV, "Nnet CV"))
r %>% knitr::kable()

### 2. TF IDF
fitNN_TF_IDF <- train(cod ~ ., data=train_tf_features_TF_IDF, method="nnet", metric='Accuracy',trControl=tr_control)
predNN_TF_IDF <- predict(fitNN_TF_IDF, newdata=test_tf_features_TF_IDF)
cmNN_TF_IDF <- confusionMatrix(predNN_TF_IDF, test_tf_features_TF_IDF$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_tf_features_TF_IDF$cod, predictedValues = predNN_TF_IDF, "Nnet TF IDF"))
r %>% knitr::kable()

### 3.1 W2V CBOW ADD
fitNN_W2V_CBOW_ADD <- train(cod ~ ., data=train_features_W2V_CBOW_ADD, method="nnet", metric='Accuracy',trControl=tr_control)
predNN_W2V_CBOW_ADD <- predict(fitNN_W2V_CBOW_ADD, newdata=test_features_W2V_CBOW_ADD)
cmNN_W2V_CBOW_ADD <- confusionMatrix(predNN_W2V_CBOW_ADD, test_features_W2V_CBOW_ADD$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_CBOW_ADD$cod, predictedValues = predNN_W2V_CBOW_ADD, "Nnet W2V CBOW ADD"))
r %>% knitr::kable()

### 3.2 W2V CBOW MEAN
fitNN_W2V_CBOW_MEAN <- train(cod ~ ., data=train_features_W2V_CBOW_MEAN, method="nnet", metric='Accuracy',trControl=tr_control)
predNN_W2V_CBOW_MEAN <- predict(fitNN_W2V_CBOW_MEAN, newdata=test_features_W2V_CBOW_MEAN)
cmNN_W2V_CBOW_MEAN <- confusionMatrix(predNN_W2V_CBOW_MEAN, test_features_W2V_CBOW_MEAN$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_CBOW_MEAN$cod, predictedValues = predNN_W2V_CBOW_MEAN, "Nnet W2V CBOW MEAN"))
r %>% knitr::kable()

### 3.3 W2V SKIP ADD
fitNN_W2V_SKIP_ADD <- train(cod ~ ., data=train_features_W2V_SKIP_ADD, method="nnet", metric='Accuracy',trControl=tr_control)
predNN_W2V_SKIP_ADD <- predict(fitNN_W2V_SKIP_ADD, newdata=test_features_W2V_SKIP_ADD)
cmNN_W2V_SKIP_ADD <- confusionMatrix(predNN_W2V_SKIP_ADD, test_features_W2V_SKIP_ADD$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_SKIP_ADD$cod, predictedValues = predNN_W2V_SKIP_ADD, "Nnet W2V SKIP ADD"))
r %>% knitr::kable()

### 3.4 W2V SKIP MEAN
fitNN_W2V_SKIP_MEAN <- train(cod ~ ., data=train_features_W2V_SKIP_MEAN, method="nnet", metric='Accuracy',trControl=tr_control)
predNN_W2V_SKIP_MEAN <- predict(fitNN_W2V_SKIP_MEAN, newdata=test_features_W2V_SKIP_MEAN)
cmNN_W2V_SKIP_MEAN <- confusionMatrix(predNN_W2V_SKIP_MEAN, test_features_W2V_SKIP_MEAN$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_W2V_SKIP_MEAN$cod, predictedValues = predNN_W2V_SKIP_MEAN, "Nnet W2V SKIP MEAN"))
r %>% knitr::kable()

### 4.1 FASTTEXT CBOW
fitNN_FASTTEXT_CBOW <- train(cod ~ ., data=train_features_FASTTEXT_CBOW, method="nnet", metric='Accuracy',trControl=tr_control)
predNN_FASTTEXT_CBOW <- predict(fitNN_FASTTEXT_CBOW, newdata=test_features_FASTTEXT_CBOW)
cmNN_FASTTEXT_CBOW <- confusionMatrix(predNN_FASTTEXT_CBOW, test_features_FASTTEXT_CBOW$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_FASTTEXT_CBOW$cod, predictedValues = predNN_FASTTEXT_CBOW, "Nnet FASTTEXT CBOW"))
r %>% knitr::kable()

### 4.1 FASTTEXT SKIP
fitNN_FASTTEXT_SKIP <- train(cod ~ ., data=train_features_FASTTEXT_SKIP, method="nnet", metric='Accuracy',trControl=tr_control)
predNN_FASTTEXT_SKIP <- predict(fitNN_FASTTEXT_SKIP, newdata=test_features_FASTTEXT_SKIP)
cmNN_FASTTEXT_SKIP <- confusionMatrix(predNN_FASTTEXT_SKIP, test_features_FASTTEXT_SKIP$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_FASTTEXT_SKIP$cod, predictedValues = predNN_FASTTEXT_SKIP, "Nnet FASTTEXT SKIP"))
r %>% knitr::kable()


###5. GLOVE
fitNN_GLOVE <- train(cod ~ ., data=train_features_GLOVE, method="nnet", metric='Accuracy',trControl=tr_control)
predNN_GLOVE <- predict(fitNN_GLOVE, newdata=test_features_GLOVE)
cmNN_GLOVE <- confusionMatrix(predNN_GLOVE, test_features_GLOVE$cod)
r <- bind_rows(r, buildAccMeasures(trueValues = test_features_GLOVE$cod, predictedValues = predNN_GLOVE, "Nnet GLOVE"))
r %>% knitr::kable()


####################
# K Nearest Neighbor
####################
### 1. Count Vectorization
fitKNN_CV <- train(cod ~ ., data=train_cf_features_CV, method="knn", metric='Accuracy',
             tuneGrid=expand.grid(k=1:15),trControl=tr_control)

predKNN_CV <- predict(fitKNN_CV, newdata=test_cf_features_CV)
cmKNN_CV <- confusionMatrix(predKNN_CV, test_cf_features_CV$cod)
r <- bind_rows(r, buildAccMeasures(test_cf_features_CV$cod, predKNN_CV, "knn CV"))
r %>% knitr::kable()

### 2. TF IDF
fitKNN_TF_IDF <- train(cod ~ ., data=train_tf_features_TF_IDF, method="knn", metric='Accuracy',
                   tuneGrid=expand.grid(k=1:15),trControl=tr_control)

predKNN_TF_IDF <- predict(fitKNN_TF_IDF, newdata=test_tf_features_TF_IDF)
cmKNN_TF_IDF <- confusionMatrix(predKNN_TF_IDF, test_tf_features_TF_IDF$cod)
r <- bind_rows(r, buildAccMeasures(test_tf_features_TF_IDF$cod, predKNN_TF_IDF, "knn TF IDF"))
r %>% knitr::kable()

### 3.1 W2V CBOW ADD
fitKNN_W2V_CBOW_ADD <- train(cod ~ ., data=train_features_W2V_CBOW_ADD, method="knn", metric='Accuracy',
                       tuneGrid=expand.grid(k=1:15),trControl=tr_control)

predKNN_W2V_CBOW_ADD <- predict(fitKNN_W2V_CBOW_ADD, newdata=test_features_W2V_CBOW_ADD)
cmKNN_W2V_CBOW_ADD <- confusionMatrix(predKNN_W2V_CBOW_ADD, test_features_W2V_CBOW_ADD$cod)
r <- bind_rows(r, buildAccMeasures(test_features_W2V_CBOW_ADD$cod, predKNN_W2V_CBOW_ADD, "knn W2 CBOW ADD"))
r %>% knitr::kable()

### 3.2 W2V CBOW MEAN
fitKNN_W2V_CBOW_MEAN <- train(cod ~ ., data=train_features_W2V_CBOW_MEAN, method="knn", metric='Accuracy',
                             tuneGrid=expand.grid(k=1:15),trControl=tr_control)

predKNN_W2V_CBOW_MEAN <- predict(fitKNN_W2V_CBOW_MEAN, newdata=test_features_W2V_CBOW_MEAN)
cmKNN_W2V_CBOW_MEAN <- confusionMatrix(predKNN_W2V_CBOW_MEAN, test_features_W2V_CBOW_MEAN$cod)
r <- bind_rows(r, buildAccMeasures(test_features_W2V_CBOW_MEAN$cod, predKNN_W2V_CBOW_MEAN, "knn W2 CBOW MEAN"))
r %>% knitr::kable()


### 3.3 W2V CBOW ADD
fitKNN_W2V_SKIP_ADD <- train(cod ~ ., data=train_features_W2V_SKIP_ADD, method="knn", metric='Accuracy',
                             tuneGrid=expand.grid(k=1:15),trControl=tr_control)

predKNN_W2V_SKIP_ADD <- predict(fitKNN_W2V_SKIP_ADD, newdata=test_features_W2V_SKIP_ADD)
cmKNN_W2V_SKIP_ADD <- confusionMatrix(predKNN_W2V_SKIP_ADD, test_features_W2V_SKIP_ADD$cod)
r <- bind_rows(r, buildAccMeasures(test_features_W2V_SKIP_ADD$cod, predKNN_W2V_SKIP_ADD, "knn W2 SKIP ADD"))
r %>% knitr::kable()

### 3.4 W2V SKIP MEAN
fitKNN_W2V_SKIP_MEAN <- train(cod ~ ., data=train_features_W2V_SKIP_MEAN, method="knn", metric='Accuracy',
                              tuneGrid=expand.grid(k=1:15),trControl=tr_control)

predKNN_W2V_SKIP_MEAN <- predict(fitKNN_W2V_SKIP_MEAN, newdata=test_features_W2V_SKIP_MEAN)
cmKNN_W2V_SKIP_MEAN <- confusionMatrix(predKNN_W2V_SKIP_MEAN, test_features_W2V_SKIP_MEAN$cod)
r <- bind_rows(r, buildAccMeasures(test_features_W2V_SKIP_MEAN$cod, predKNN_W2V_SKIP_MEAN, "knn W2 SKIP MEAN"))
r %>% knitr::kable()

### 4.1 FASTTEXT CBOW
fitKNN_FASTTEXT_CBOW <- train(cod ~ ., data=train_features_FASTTEXT_CBOW, method="knn", metric='Accuracy',
                              tuneGrid=expand.grid(k=1:15),trControl=tr_control)

predKNN_FASTTEXT_CBOW <- predict(fitKNN_FASTTEXT_CBOW, newdata=test_features_FASTTEXT_CBOW)
cmKNN_FASTTEXT_CBOW <- confusionMatrix(predKNN_FASTTEXT_CBOW, test_features_FASTTEXT_CBOW$cod)
r <- bind_rows(r, buildAccMeasures(test_features_FASTTEXT_CBOW$cod, predKNN_FASTTEXT_CBOW, "knn FASTTEXT CBOW"))
r %>% knitr::kable()

### 4.2 FASTTEXT SKIP
fitKNN_FASTTEXT_SKIP <- train(cod ~ ., data=train_features_FASTTEXT_SKIP, method="knn", metric='Accuracy',
                              tuneGrid=expand.grid(k=1:15),trControl=tr_control)

predKNN_FASTTEXT_SKIP <- predict(fitKNN_FASTTEXT_SKIP, newdata=test_features_FASTTEXT_SKIP)
cmKNN_FASTTEXT_SKIP <- confusionMatrix(predKNN_FASTTEXT_SKIP, test_features_FASTTEXT_SKIP$cod)
r <- bind_rows(r, buildAccMeasures(test_features_FASTTEXT_SKIP$cod, predKNN_FASTTEXT_SKIP, "knn FASTTEXT SKIP"))
r %>% knitr::kable()


### 5. GLOVE
fitKNN_GLOVE <- train(cod ~ ., data=train_features_GLOVE, method="knn", metric='Accuracy',
                              tuneGrid=expand.grid(k=1:15),trControl=tr_control)

predKNN_GLOVE <- predict(fitKNN_GLOVE, newdata=test_features_GLOVE)
cmKNN_GLOVE <- confusionMatrix(predKNN_GLOVE, test_features_GLOVE$cod)
r <- bind_rows(r, buildAccMeasures(test_features_GLOVE$cod, predKNN_GLOVE, "knn GLOVE"))
r %>% knitr::kable()

####################
# XgBoost
####################
### 1. Count Vectorization
fitXG_CV <- train(cod ~ ., data=train_cf_features_CV, method="xgbTree", metric='Accuracy',
                             tuneGrid=expand.grid(nrounds=1:6, max_depth=2:6, eta=seq(0,1,0.2), gamma = 1, colsample_bytree = 1, min_child_weight = 1, subsample= c(0.25, 0.5, 0.75)),
                             trControl=tr_control)
predXG_CV <- predict(fitXG_CV, newdata=test_cf_features_CV)
cmXG_CV <- confusionMatrix(predXG_CV, test_cf_features_CV$cod)
r <- bind_rows(r, buildAccMeasures(test_cf_features_CV$cod, predXG_CV, "Xgboost - CV"))
r %>% knitr::kable()

### 2. TF IDF
fitXG_TF_IDF <- train(cod ~ ., data=train_tf_features_TF_IDF, method="xgbTree", metric='Accuracy',
                  tuneGrid=expand.grid(nrounds=1:6, max_depth=2:6, eta=seq(0,1,0.2), gamma = 1, colsample_bytree = 1, min_child_weight = 1, subsample= c(0.25, 0.5, 0.75)),
                  trControl=tr_control)
predXG_TF_IDF <- predict(fitXG_TF_IDF, newdata=test_tf_features_TF_IDF)
cmXG_TF_IDF <- confusionMatrix(predXG_TF_IDF, test_tf_features_TF_IDF$cod)
r <- bind_rows(r, buildAccMeasures(test_tf_features_TF_IDF$cod, predXG_TF_IDF, "Xgboost - TF IDF"))
r %>% knitr::kable()

### 3.1 W2V CBOW ADD
fitXG_W2V_CBOW_ADD <- train(cod ~ ., data=train_features_W2V_CBOW_ADD, method="xgbTree", metric='Accuracy',
                      tuneGrid=expand.grid(nrounds=1:6, max_depth=2:6, eta=seq(0,1,0.2), gamma = 1, colsample_bytree = 1, min_child_weight = 1, subsample= c(0.25, 0.5, 0.75)),
                      trControl=tr_control)
predXG_W2V_CBOW_ADD <- predict(fitXG_W2V_CBOW_ADD, newdata=test_features_W2V_CBOW_ADD)
cmXG_W2V_CBOW_ADD <- confusionMatrix(predXG_W2V_CBOW_ADD, test_features_W2V_CBOW_ADD$cod)
r <- bind_rows(r, buildAccMeasures(test_features_W2V_CBOW_ADD$cod, predXG_W2V_CBOW_ADD, "Xgboost - W2V CBOW ADD"))
r %>% knitr::kable()

### 3.2 W2V CBOW MEAN
fitXG_W2V_CBOW_MEAN <- train(cod ~ ., data=train_features_W2V_CBOW_MEAN, method="xgbTree", metric='Accuracy',
                            tuneGrid=expand.grid(nrounds=1:6, max_depth=2:6, eta=seq(0,1,0.1), gamma = 1, colsample_bytree = 1, min_child_weight = 1, subsample= c(0.25, 0.5, 0.75)),
                            trControl=tr_control)
predXG_W2V_CBOW_MEAN <- predict(fitXG_W2V_CBOW_MEAN, newdata=test_features_W2V_CBOW_MEAN)
cmXG_W2V_CBOW_MEAN <- confusionMatrix(predXG_W2V_CBOW_MEAN, test_features_W2V_CBOW_MEAN$cod)
r <- bind_rows(r, buildAccMeasures(test_features_W2V_CBOW_MEAN$cod, predXG_W2V_CBOW_MEAN, "Xgboost - W2V CBOW MEAN"))
r %>% knitr::kable()

### 3.3 W2V SKIP ADD
fitXG_W2V_SKIP_ADD <- train(cod ~ ., data=train_features_W2V_SKIP_ADD, method="xgbTree", metric='Accuracy',
                            tuneGrid=expand.grid(nrounds=1:6, max_depth=2:6, eta=seq(0,1,0.1), gamma = 1, colsample_bytree = 1, min_child_weight = 1, subsample= c(0.25, 0.5, 0.75)),
                            trControl=tr_control)
predXG_W2V_SKIP_ADD <- predict(fitXG_W2V_SKIP_ADD, newdata=test_features_W2V_SKIP_ADD)
cmXG_W2V_SKIP_ADD <- confusionMatrix(predXG_W2V_SKIP_ADD, test_features_W2V_SKIP_ADD$cod)
r <- bind_rows(r, buildAccMeasures(test_features_W2V_SKIP_ADD$cod, predXG_W2V_SKIP_ADD, "Xgboost - W2V SKIP ADD"))
r %>% knitr::kable()


### 3.4 W2V SKIP MEAN
fitXG_W2V_SKIP_MEAN <- train(cod ~ ., data=train_features_W2V_SKIP_MEAN, method="xgbTree", metric='Accuracy',
                             tuneGrid=expand.grid(nrounds=1:6, max_depth=2:6, eta=seq(0,1,0.1), gamma = 1, colsample_bytree = 1, min_child_weight = 1, subsample= c(0.25, 0.5, 0.75)),
                             trControl=tr_control)
predXG_W2V_SKIP_MEAN <- predict(fitXG_W2V_SKIP_MEAN, newdata=test_features_W2V_SKIP_MEAN)
cmXG_W2V_SKIP_MEAN <- confusionMatrix(predXG_W2V_SKIP_MEAN, test_features_W2V_SKIP_MEAN$cod)
r <- bind_rows(r, buildAccMeasures(test_features_W2V_SKIP_MEAN$cod, predXG_W2V_SKIP_MEAN, "Xgboost - W2V SKIP MEAN"))
r %>% knitr::kable()


### 4.1 FASTTEXT CBOW
fitXG_FASTTEXT_CBOW <- train(cod ~ ., data=train_features_FASTTEXT_CBOW, method="xgbTree", metric='Accuracy',
                             tuneGrid=expand.grid(nrounds=1:6, max_depth=2:6, eta=seq(0,1,0.1), gamma = 1, colsample_bytree = 1, min_child_weight = 1, subsample= c(0.25, 0.5, 0.75)),
                             trControl=tr_control)
predXG_FASTTEXT_CBOW <- predict(fitXG_FASTTEXT_CBOW, newdata=test_features_FASTTEXT_CBOW)
cmXG_FASTTEXT_CBOW <- confusionMatrix(predXG_FASTTEXT_CBOW, test_features_FASTTEXT_CBOW$cod)
r <- bind_rows(r, buildAccMeasures(test_features_FASTTEXT_CBOW$cod, predXG_FASTTEXT_CBOW, "Xgboost - FASTTEXT_CBOW"))
r %>% knitr::kable()

### 4.2 FASTTEXT SKIP
fitXG_FASTTEXT_SKIP <- train(cod ~ ., data=train_features_FASTTEXT_SKIP, method="xgbTree", metric='Accuracy',
             tuneGrid=expand.grid(nrounds=1:6, max_depth=2:6, eta=seq(0,1,0.1), gamma = 1, colsample_bytree = 1, min_child_weight = 1, subsample= c(0.25, 0.5, 0.75)),
             trControl=tr_control)
predXG_FASTTEXT_SKIP <- predict(fitXG_FASTTEXT_SKIP, newdata=test_features_FASTTEXT_SKIP)
cmXG_FASTTEXT_SKIP <- confusionMatrix(predXG_FASTTEXT_SKIP, test_features_FASTTEXT_SKIP$cod)
r <- bind_rows(r, buildAccMeasures(test_features_FASTTEXT_SKIP$cod, predXG_FASTTEXT_SKIP, "Xgboost - FASTTEXT_SKIP"))
r %>% knitr::kable()

### 5. GLOVE
fitXG_GLOVE <- train(cod ~ ., data=train_features_GLOVE, method="xgbTree", metric='Accuracy',
                             tuneGrid=expand.grid(nrounds=1:6, max_depth=2:6, eta=seq(0,1,0.1), gamma = 1, colsample_bytree = 1, min_child_weight = 1, subsample= c(0.25, 0.5, 0.75)),
                             trControl=tr_control)
predXG_GLOVE <- predict(fitXG_GLOVE, newdata=test_features_GLOVE)
cmXG_GLOVE <- confusionMatrix(predXG_GLOVE, test_features_GLOVE$cod)
r <- bind_rows(r, buildAccMeasures(test_features_GLOVE$cod, predXG_GLOVE, "Xgboost - GLOVE"))
r %>% knitr::kable()


###################### plot the data sets in 2D #################
## Count Vectorization
t_cv<-train_cf_features_CV[!duplicated(train_cf_features_CV[,2:3001]),]
tt_cv<-as.matrix(t_cv[, 2:3001])
tsne_cv <- Rtsne(tt_cv, perplexity=300, theta=0.0, initial_dims = 3000, num_threads = 4)
plot(tsne_cv$Y,col=t_cv$cod, asp=0, pch = 19, cex = 1.25, xlab="", ylab="")
title("Count Vectorization")
legend("topleft", pch = 19, col = unique(t_cv$cod),lwd = 3, legend = unique(df[t_cv$cod,1]))

### TF-IDF
t_tf<-train_tf_features_TF_IDF[!duplicated(train_tf_features_TF_IDF[,2:3001]),]
tt_tf<-as.matrix(t_tf[, 2:3001])
tsne_tf <- Rtsne(tt_tf, perplexity=300, theta=0.0, initial_dims = 3000, num_threads = 4)
plot(tsne_tf$Y,col=t_tf$cod, asp=0, pch = 19, cex = 1.25, xlab="", ylab="")
title("TF-IDF")
legend("topleft", pch = 19, col = unique(t_tf$cod),lwd = 3, legend = unique(df[t_tf$cod,1]))


## W2V_CBOW_ADD
t_w2v_cbow_add<-train_features_W2V_CBOW_ADD[!duplicated(train_features_W2V_CBOW_ADD[,2:51]),]
tt_w2v_cbow_add<-as.matrix(t_w2v_cbow_add[, 2:51])
tsne_w2v_cbow_add <- Rtsne(tt_w2v_cbow_add, perplexity=300, theta=0.0, initial_dims = 50, num_threads = 4)
plot(tsne_w2v_cbow_add$Y,col=t_w2v_cbow_add$cod, asp=0, pch = 19, cex = 1.25, xlab="", ylab="")
title("W2V CBOW ADD")
legend("bottomleft", pch = 19, col = unique(t_w2v_cbow_add$cod),lwd = 3, legend = unique(df[t_w2v_cbow_add$cod,1]))

## W2V_CBOW_MEAN
t_w2v_cbow_mean<-train_features_W2V_CBOW_MEAN[!duplicated(train_features_W2V_CBOW_MEAN[,2:51]),]
tt_w2v_cbow_mean<-as.matrix(t_w2v_cbow_mean[, 2:51])
tsne_w2v_cbow_mean <- Rtsne(tt_w2v_cbow_mean, perplexity=300, theta=0.0, initial_dims = 50, num_threads = 4)
plot(tsne_w2v_cbow_mean$Y,col=t_w2v_cbow_mean$cod, asp=0, pch = 19, cex = 1.25, xlab="", ylab="")
title("W2V CBOW MEAN")
legend("topleft", pch = 19, col = unique(t_w2v_cbow_mean$cod),lwd = 3, legend = unique(df[t_w2v_cbow_mean$cod,1]))

## W2V_SKIP_ADD
t_w2v_skip_add<-train_features_W2V_SKIP_ADD[!duplicated(train_features_W2V_SKIP_ADD[,2:51]),]
tt_w2v_skip_add<-as.matrix(t_w2v_skip_add[, 2:51])
tsne_w2v_skip_add <- Rtsne(tt_w2v_skip_add, perplexity=300, theta=0.0, initial_dims = 50, num_threads = 4)
plot(tsne_w2v_skip_add$Y,col=t_w2v_skip_add$cod, asp=0, pch = 19, cex = 1.25, xlab="", ylab="")
title("W2V SKIP ADD")
legend("topleft", pch = 19, col = unique(t_w2v_skip_add$cod),lwd = 3, legend = unique(df[t_w2v_skip_add$cod,1]))


## W2V_SKIP_MEAN
t_w2v_skip_mean<-train_features_W2V_SKIP_MEAN[!duplicated(train_features_W2V_SKIP_MEAN[,2:51]),]
tt_w2v_skip_mean<-as.matrix(t_w2v_skip_mean[, 2:51])
tsne_w2v_skip_mean <- Rtsne(tt_w2v_skip_mean, perplexity=300, theta=0.0, initial_dims = 50, num_threads = 4)
plot(tsne_w2v_skip_mean$Y,col=t_w2v_skip_mean$cod, asp=0, pch = 19, cex = 1.25, xlab="", ylab="")
title("W2V SKIP MEAN")
legend("bottomleft", pch = 19, col = unique(t_w2v_skip_mean$cod),lwd = 3, legend = unique(df[t_w2v_skip_mean$cod,1]))


## FASTTEXT_CBOW
t_fasttext_cbow <-train_features_FASTTEXT_CBOW[!duplicated(train_features_FASTTEXT_CBOW[,2:51]),]
tt_fasttext_cbow <-as.matrix(t_fasttext_cbow[, 2:51])
tsne_fasttext_cbow <- Rtsne(tt_fasttext_cbow, perplexity=300, theta=0.0, initial_dims = 50, num_threads = 4)
plot(tsne_fasttext_cbow$Y,col=t_fasttext_cbow$cod, asp=0, pch = 19, cex = 1.25, xlab="", ylab="")
title("FASTTEXT CBOW")
legend("topright", pch = 19, col = unique(t_fasttext_cbow$cod),lwd = 3, legend = unique(df[t_fasttext_cbow$cod,1]))


## FASTTEXT_SKIP
t_fasttext_skip <-train_features_FASTTEXT_SKIP [!duplicated(train_features_FASTTEXT_SKIP[,2:51]),]
tt_fasttext_skip <-as.matrix(t_fasttext_skip[, 2:51])
tsne_fasttext_skip <- Rtsne(tt_fasttext_skip, perplexity=300, theta=0.0, pca= FALSE, num_threads = 4)
plot(tsne_fasttext_skip$Y,col=t_fasttext_skip$cod, asp=0, pch = 19, cex = 1.25, xlab="", ylab="")
title("FASTTEXT SKIP GRAM")
legend("bottomleft", pch = 19, col = unique(t_fasttext_skip$cod),lwd = 3, legend = unique(df[t_fasttext_skip$cod,1]))

## GLOVE
t_glove <-train_features_GLOVE [!duplicated(train_features_GLOVE[,2:51]),]
tt_glove <-as.matrix(t_glove[, 2:51])
tsne_glove <- Rtsne(tt_glove, perplexity=300, theta=0.0, initial_dims = 50, num_threads = 4)
plot(tsne_glove$Y,col=t_glove$cod, asp=0, pch = 19, cex = 1.25, xlab="", ylab="")
title("GLOVE")
legend("bottomleft", pch = 19, col = unique(t_glove$cod),lwd = 3, legend = unique(df[t_glove$cod,1]))
