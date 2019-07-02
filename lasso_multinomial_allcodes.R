library(glmnet)
library(quanteda)
require(doMC)

setwd("~/Documents/China Legal Research")
data <- read.csv(file = "final.csv", stringsAsFactors=F, header=TRUE)
training <- (1:100)
test <- (101:200)

corpus <- corpus(data$Decision)
dfm <- dfm(corpus, remove_punct=TRUE)
dfm <- dfm_trim(dfm, min_docfreq = 2)

fit = glmnet(dfm[training,], data$codes[training], family = "multinomial", type.multinomial = "grouped")
plot(fit, xvar = "lambda", label = TRUE, type.coef = "2norm")

cvfit=cv.glmnet(dfm[training,],data$codes[training] , family="multinomial", type.multinomial = "grouped", parallel = TRUE)
plot(cvfit)

preds <- predict(cvfit, dfm[test,], s = "lambda.min", type = "class", alpha=1)

decisions <- data[201:400, 'Decision']
result <- cbind(decisions, preds)
#write.csv(result, "predicted_codes.csv")