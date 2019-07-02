library(glmnet)
library(quanteda)
require(doMC)

setwd("~/Documents/China Legal Research/OGI_parser_spring")
data <- read.csv(file = "merged_ogi_agg.csv", stringsAsFactors=F, header=TRUE)
corpus <- corpus(data$facts)
dfm <- dfm(corpus, remove_punct=TRUE)
dfm <- dfm_trim(dfm, min_docfreq = 1000)

set.seed(123)
training <- sample(1:nrow(data), floor(.90 * nrow(data)))
test <- (1:nrow(data))[1:nrow(data) %in% training == FALSE]

registerDoMC(cores=3)
lasso <- cv.glmnet(dfm[training,], data$win_code[training], 
                   family="binomial", alpha=1, nfolds=5, parallel=TRUE, intercept=TRUE,
                   type.measure="deviance")
plot(lasso)

## function to compute accuracy
accuracy <- function(ypred, y){
  tab <- table(ypred, y)
  return(sum(diag(tab))/sum(tab))
}
# function to compute precision
precision <- function(ypred, y){
  tab <- table(ypred, y)
  return((tab[2,2])/(tab[2,1]+tab[2,2]))
}
# function to compute recall
recall <- function(ypred, y){
  tab <- table(ypred, y)
  return(tab[2,2]/(tab[1,2]+tab[2,2]))
}
# computing predicted values
preds <- predict(lasso, dfm[test,], type="response") > mean(data$win_code[test])
# confusion matrix
table(preds, data$win_code[test])
accuracy(preds, data$win_code[test])

precision(preds, data$win_code[test])
recall(preds, data$win_code[test])

best.lambda <- which(lasso$lambda==lasso$lambda.min)
beta <- lasso$glmnet.fit$beta[,best.lambda]
head(beta)

df <- data.frame(coef = as.numeric(beta),
                 word = names(beta), stringsAsFactors=F)

df <- df[order(df$coef),]
#head(df[,c("coef", "word")], n=30)
#highest predictors
paste(df$word[1:20], collapse=", ")

df <- df[order(df$coef, decreasing=TRUE),]
#head(df[,c("coef", "word")], n=30)
#lowest predictors
#paste(df$word[1:5], collapse=", ")

# computing predicted values
preds <- predict(lasso, dfm[training,], type="response") > mean(data$win_code[training])
# confusion matrix
table(preds, data$win_code[training])
accuracy(preds, data$win_code[training])

precision(preds, data$win_code[training])
recall(preds, data$win_code[training])