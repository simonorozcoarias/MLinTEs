# Calculate statistical significance of difference between model predictions

# load libraries
library(mlbench)
library(caret)
library(doParallel)

# function to calculate F1 score

metrics <- function (data, lev = NULL, model = NULL) {
  levels = c(1,12,13,14,16,17,18,19,20,3,4,9)
  dat <- data.frame(obs = data$obs, pred = data$pred)
  multiMetrics <- multiClassSummary(dat, lev=levels)
  f1_val <- multiMetrics["F1"]
  pres <- multiMetrics["Precision"]
  sen <- multiMetrics["Sensitivity"]
  spe <- multiMetrics["Specificity"]
  rec <- multiMetrics["Recall"]
  acc <- multiMetrics["Accuracy"]
  metrics = c(f1_val, pres, sen, spe, rec, acc)
  names(metrics) <- c("F1", "Precision", "Sensitivity", "Specificity", "Recall", "Accuracy")
  metrics
}

# load the dataset
setwd("~/work/MLenTEs")
repbase <- read.csv(file='databases/repbase_LTRs_I_3dom.fasta.lineages_final.kmers')
Y <- as.factor(repbase[,1])
# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3, summaryFunction = metrics)

cl <- makePSOCKcluster(20)
registerDoParallel(cl)

# Logistic Regression
set.seed(7)
fit.lr <- train(x = repbase[,2:ncol(repbase)], Y, method="LogitBoost", trControl=control, metric="F1")
# LDA
set.seed(7)
fit.lda <- train(x = repbase[,2:ncol(repbase)], Y, method="lda", trControl=control, metric="F1")
# kNN
set.seed(7)
fit.knn <- train(x = repbase[,2:ncol(repbase)], Y, method="knn", trControl=control, metric="F1")
# SVM
set.seed(7)
fit.svm <- train(x = repbase[,2:ncol(repbase)], Y, method="svmRadial", trControl=control, metric="F1")
# Multi-Layer Perceptron
set.seed(7)
fit.mlp <- train(x = repbase[,2:ncol(repbase)], Y, method="mlp", trControl=control, metric="F1")
# Random Forest
set.seed(7)
fit.rf <- train(x = repbase[,2:ncol(repbase)], Y, method="rf", trControl=control, metric="F1")
# AdaBoost Classification Trees
set.seed(7)
fit.dt <- train(x = repbase[,2:ncol(repbase)], Y, method="adaboost", trControl=control, metric="F1")
# Naive Bayes Classifier
set.seed(7)
fit.nb <- train(x = repbase[,2:ncol(repbase)], Y, method="nbDiscrete", trControl=control, metric="F1")


# collect resamples
results <- resamples(list(LR=fit.lr, LDA=fit.lda, KNN=fit.knn, SVM=fit.svm, MLP=fit.mlp, RF=fit.rf, DT=fit.dt, NB=fit.nb))
# difference in model predictions
diffs <- diff(results)
# summarize p-values for pair-wise comparisons
summary(diffs)
# plot of differences
pdf("bwplot.pdf")
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(diffs, scales=scales)
dev.off()
# t-test between two models
compare_models(fit.svm, fit.lda)
