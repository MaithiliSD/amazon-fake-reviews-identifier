rm(list=ls())

#install.packages("tokenizers")
#install.packages("stringr")
#install.packages("quanteda")

library(tokenizers)
library(quanteda)
library(stringr)

#### choose file ####
## choose amazon_reviews.csv (converted from amazon_reviews.txt)
file <- file.choose()
reviews <- read.csv(file, header = TRUE)

#### 1=fake, 0=genuine ####
reviews$LABEL <- ifelse(reviews$LABEL == 	"__label1__", 1, 0)
reviews$LABEL <- factor(reviews$LABEL, labels=c("Genuine", "Fake"))

#### total words in a review ####
for(i in reviews$DOC_ID) {
  sentences <- reviews$REVIEW_TEXT[i]
  tot <- tokenize_words(toString(reviews$REVIEW_TEXT[i]))
  reviews$TOTAL_WORDS[i] <- length(tot[[1]])
}

#### total sentences in a review ####
for(i in reviews$DOC_ID) {
  sentences <- reviews$REVIEW_TEXT[i]
  tot <- tokenize_sentences(toString(reviews$REVIEW_TEXT[i]))
  reviews$TOTAL_SENTENCES[i] <- length(tot[[1]])
}

#### number of punctuations in a review ####
for(i in reviews$DOC_ID) {
  punct <- tokenize_words(toString(reviews$REVIEW_TEXT[i]), strip_punct = FALSE)
  nopunct <- tokenize_words(toString(reviews$REVIEW_TEXT[i]), strip_punct = TRUE)
  
  before <- sapply(punct, length)
  after <- sapply(nopunct, length)
  reviews$TOTAL_PUNCTUATIONS[i] <- before - after
}

####  product name match ####
for(i in reviews$DOC_ID) {
  rev_text <- as.character(reviews$REVIEW_TEXT[i])
  prod_title <- as.character(reviews$PRODUCT_TITLE[i])
  bool <- str_detect(rev_text, fixed(prod_title))
  reviews$PRODUCTNAME_IN_REVIEW[i] <- bool
}

#### total characters in review title ####
for(i in reviews$DOC_ID) {
  titlechar <- as.character(reviews$REVIEW_TITLE[i])
  reviews$TITLE_CHARACTERS[i] <- str_length(titlechar)
}

#### total punctuations in review title ####
for(i in reviews$DOC_ID) {
  punct <- tokenize_words(toString(reviews$REVIEW_TITLE[i]), strip_punct = FALSE)
  nopunct <- tokenize_words(toString(reviews$REVIEW_TITLE[i]), strip_punct = TRUE)
  
  before <- sapply(punct, length)
  after <- sapply(nopunct, length)
  reviews$TITLE_PUNCTUATIONS[i] <- before - after
}

####  total number of stopwords in a review ####
# NOTE: this function takes around 5 mins to get executed
for(i in reviews$DOC_ID) {
  temp = toString(reviews$REVIEW_TEXT[i])
  sentence = tolower(temp)
  sentence = gsub('[[:punct:]]', '', sentence)
  
  # Length of text with stop words
  before <- sapply(strsplit(sentence, " "), length)
  
  # Convert setence to tokens
  doc_token <- tokens(sentence)
  
  # Remove stopwords
  toks_nostop_basic <- tokens_select(doc_token, pattern = stopwords('en'), selection = 'remove')
  
  # Length of text without stopwords
  after <- sapply(toks_nostop_basic, length)
  
  # Number of stopwords
  reviews$TOTAL_STOPWORDS[i] <- before - after
}

reviews.df = as.data.frame(reviews)

#### Change the path below, where you want to export the new csv file ####
write.csv(reviews.df,"/Users/maithili/Desktop/amazon_reviews_cols.csv", row.names = FALSE)
