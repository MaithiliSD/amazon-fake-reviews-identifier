rm(list = ls())

#### KNN ####
library(kknn)

reviews <- read.csv(file.choose(), header = TRUE)
reviews$LABEL = factor(reviews$LABEL, labels = c('Fake', 'Genuine'))

  # Use min-max normalization to normalize data
mmnorm <- function(x,minx,maxx) {
  z <- ((x - minx)/(maxx - minx))
  return(z) 
}

# Define 'x', 'minx' and 'maxx' for 'mmnorm' function
reviews_norm <- as.data.frame(cbind(
  DOC_ID = reviews$DOC_ID,
  LABEL = reviews$LABEL,
  RATING = mmnorm(as.numeric(reviews[,3]), min(as.numeric(reviews[,3])), max(as.numeric(reviews[,3]))), 
  VERIFIED_PURCHASE = mmnorm(as.numeric(reviews[,4]), min(as.numeric(reviews[,4])), max(as.numeric(reviews[,4]))), 
  PRODUCT_CATEGORY = reviews$PRODUCT_CATEGORY,
  PRODUCT_ID = reviews$PRODUCT_ID,
  PRODUCT_TITLE = reviews$PRODUCT_TITLE,
  REVIEW_TITLE = reviews$REVIEW_TITLE,
  REVIEW_TEXT = reviews$REVIEW_TEXT,
  TOTAL_WORDS = mmnorm(as.numeric(reviews[,10]), min(as.numeric(reviews[,10])), max(as.numeric(reviews[,10]))), 
  TOTAL_SENTENCES = mmnorm(as.numeric(reviews[,11]), min(as.numeric(reviews[,11])), max(as.numeric(reviews[,11]))),
  TOTAL_PUNCTUATIONS = mmnorm(as.numeric(reviews[,12]), min(as.numeric(reviews[,12])), max(as.numeric(reviews[,12]))),
  PRODUCTNAME_IN_REVIEW = mmnorm(as.numeric(reviews[,13]), min(as.numeric(reviews[,13])), max(as.numeric(reviews[,13]))),
  TITLE_CHARACTERS = mmnorm(as.numeric(reviews[,14]), min(as.numeric(reviews[,14])), max(as.numeric(reviews[,14]))),
  TOTAL_STOPWORDS = mmnorm(as.numeric(reviews[,15]), min(as.numeric(reviews[,15])), max(as.numeric(reviews[,15]))), 
  TITLE_PUNCTUATIONS = mmnorm(as.numeric(reviews[,16]), min(as.numeric(reviews[,16])), max(as.numeric(reviews[,16])))
)
)

  # Factor the 'LABEL' column in the dataframe
reviews_norm$LABEL = factor(reviews_norm$LABEL, labels = c('Fake', 'Genuine'))


  # Divide the data : 70% training 30% testing
index <- sort(sample(nrow(reviews_norm), as.integer(.70*nrow(reviews_norm))))
training <- reviews_norm[index,]
testing <- reviews_norm[-index,]


  # k-Nearest Neighbour for k = 3
predict_k5 <- kknn(formula = LABEL ~RATING+VERIFIED_PURCHASE+TOTAL_WORDS+TOTAL_SENTENCES+TOTAL_PUNCTUATIONS+PRODUCTNAME_IN_REVIEW+TITLE_CHARACTERS+TOTAL_STOPWORDS+TITLE_PUNCTUATIONS+REVIEW_TITLE+REVIEW_TEXT, training, testing, k=5, kernel = "rectangular")

fit_k1 <- fitted(predict_k5)
table(testing$LABEL, fit_k1)

 # calculate error rate
kknn_wrong_k5 <- sum(fit_k1 != testing$LABEL)
kknn_error_rate_k5 <- kknn_wrong_k5/length(fit_k1)

  # k-Nearest Neighbour for k = 5
predict_k10 <- kknn(formula = LABEL ~RATING+VERIFIED_PURCHASE+TOTAL_WORDS+TOTAL_SENTENCES+TOTAL_PUNCTUATIONS+PRODUCTNAME_IN_REVIEW+TITLE_CHARACTERS+TOTAL_STOPWORDS+TITLE_PUNCTUATIONS+REVIEW_TITLE+REVIEW_TEXT, training, testing, k=10, kernel = "rectangular")

fit_k2 <- fitted(predict_k10)
table(testing$LABEL, fit_k2)

 # calculate error rate
kknn_wrong_k10 <- sum(fit_k2 != testing$LABEL)
kknn_error_rate_k10 <- kknn_wrong_k10/length(fit_k2)

  # k-Nearest Neighbour for k = 10 - BEST
predict_k15 <- kknn(formula = LABEL ~RATING+VERIFIED_PURCHASE+PRODUCT_TITLE+PRODUCT_CATEGORY+TOTAL_WORDS+TOTAL_SENTENCES+TOTAL_PUNCTUATIONS+PRODUCTNAME_IN_REVIEW+TITLE_CHARACTERS+TOTAL_STOPWORDS+TITLE_PUNCTUATIONS+REVIEW_TITLE+REVIEW_TEXT, training, testing, k=15, kernel = "rectangular")

fit_k3 <- fitted(predict_k15)
table(testing$LABEL, fit_k3)

  # calculate error rate
kknn_wrong_k15 <- sum(fit_k3 != testing$LABEL)
kknn_error_rate_k15 <- kknn_wrong_k15/length(fit_k3)


##### Naive Bayes ####

library(e1071)
library(class)

  # Divide the data : 70% training 30% testing
index <- sort(sample(nrow(reviews), as.integer(.70*nrow(reviews))))
training_nonorm <- reviews[index,]
testing_nonorm <- reviews[-index,]

nBayes_all <- naiveBayes(LABEL ~VERIFIED_PURCHASE, data = training_nonorm)

## Naive Bayes classification using all variables 

category_all<-predict(nBayes_all, testing_nonorm)
table(NBayes_all=category_all,LABEL = testing_nonorm$LABEL)

NB_wrong <- sum(category_all!=testing_nonorm$LABEL)
NB_error_rate <- NB_wrong/length(category_all)


#### CART #### Doesn't work
library(rpart)
library(rpart.plot)
library(rattle)
library(RColorBrewer)

CART_class <- rpart(LABEL ~VERIFIED_PURCHASE, data = training_nonorm)
rpart.plot(CART_class)
CART_predict2 <- predict(CART_class, testing_nonorm, type = "class") 
table(Actual = testing_nonorm[,2],CART=CART_predict2)
CART_predict <- predict(CART_class, testing_nonorm) 

str(CART_predict)
CART_predict_cat <- ifelse(CART_predict[,1] >= 0.8,'F','G')
CART_predict_cat <- ifelse(CART_predict[,2] <= 0.2,'F','G')

table(Actual = testing_nonorm[,2], CART = CART_predict_cat)
CART_wrong <- sum(testing_nonorm[,2] != CART_predict_cat)
CART_error_rate <- CART_wrong/length(testing_nonorm[,2])
CART_error_rate

#### Random Forest #####

library(randomForest)

fit <- randomForest( LABEL ~RATING+VERIFIED_PURCHASE+PRODUCT_CATEGORY+TOTAL_WORDS+TOTAL_SENTENCES+TOTAL_PUNCTUATIONS+PRODUCTNAME_IN_REVIEW+TITLE_CHARACTERS+TOTAL_STOPWORDS, data = training_nonorm, importance = TRUE, ntree = 500)
importance(fit)
varImpPlot(fit)
prediction <- predict(fit, testing_nonorm)
table(actual = testing_nonorm[,2], prediction)

wrong<- (testing_nonorm[,2]!= prediction)
error_rate<-sum(wrong)/length(wrong)
error_rate 

#### SVM #### 

library(e1071)

# RATING+VERIFIED+TOTAL_WORDS+TOTAL_SENTENCES+TOTAL_PUNCTUATIONS+PRODUCTNAME_IN_REVIEW+TITLE_CHARACTERS+TOTAL_STOPWORDS+TITLE_PUNCTUATIONS+PRODUCT_CATEGORY+REVIEW_TITLE+REVIEW_TEXT

svm.model <- svm (LABEL ~RATING+VERIFIED_PURCHASE+TOTAL_WORDS+TOTAL_SENTENCES+TOTAL_PUNCTUATIONS+PRODUCTNAME_IN_REVIEW+TITLE_CHARACTERS+TOTAL_STOPWORDS+TITLE_PUNCTUATIONS+REVIEW_TITLE+REVIEW_TEXT, data = training)
svm.pred <- predict(svm.model,  testing)

table(actual=testing[,2], svm.pred)
SVM_wrong <- (testing$LABEL!=svm.pred)
rate <-sum(SVM_wrong)/length(SVM_wrong)
rate

#### KMEANS ####

reviews_dist <- dist(reviews_norm[, c(-2, -1, -5, -6, -7)])
hclust_resutls <- hclust(reviews_dist)
plot(hclust_resutls)
hclust_2 <- cutree(hclust_resutls, 2)
table(hclust_2, as.numeric(reviews_norm[,2]))


kmeans_2 <- kmeans(reviews_norm[,-2], 2, nstart = 10)
kmeans_2$cluster
table(kmeans_2$cluster, reviews_norm[,2])

kmeans_wrong <- (kmeans_2$cluster != as.numeric(reviews_norm[,2]))
rate <-sum(kmeans_wrong)/length(kmeans_wrong)
rate

#### ANN #### - very slow, don't try, you'll regret.

library("neuralnet")                      
net_bc  <- neuralnet( LABEL ~RATING+VERIFIED_PURCHASE+TOTAL_WORDS+TOTAL_SENTENCES+TOTAL_PUNCTUATIONS+PRODUCTNAME_IN_REVIEW+TITLE_CHARACTERS+TOTAL_STOPWORDS+TITLE_PUNCTUATIONS, training, hidden=15, threshold=0.01)

#Plot the neural network

net_bc_results <-compute(net_bc, testing[,c(-1,-2)])
ANN=as.numeric(net_bc_results$net.result)

ANN_cat<-ifelse(ANN<.3,0,1)

table(Actual=testing$LABEL,ANN_cat)

wrong<- ( testing$LABEL!=ANN_cat)
rate<-sum(wrong)/length(wrong)
rate



