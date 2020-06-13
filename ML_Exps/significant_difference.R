# Calculate statistical significance of difference between model predictions

# load libraries
library(mlbench)
library(caret)
library(doParallel)

# load the dataset
setwd("~/work/MLenTEs")
repbase <- read.csv(file='databases/repbase_LTRs_I_3dom.fasta.lineages_final.kmers')
Y <- make.names(repbase[,1])
# prepare training scheme
control_summaryFunction = get("multiClassSummary")
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions = "final", classProbs = TRUE, allowParallel = T, summaryFunction = control_summaryFunction)
cl <- makePSOCKcluster(20)
registerDoParallel(cl)

# Logistic Regression
set.seed(7)
fit.lr <- train(x = repbase[,2:ncol(repbase)], Y, method="LogitBoost", trControl=control, metric="Mean_F1")
fit.lr$results
# LDA
set.seed(7)
fit.lda <- train(x = repbase[,2:ncol(repbase)], Y, method="lda", trControl=control, metric="Mean_F1")
fit.lda$results
# kNN
set.seed(7)
fit.knn <- train(x = repbase[,2:ncol(repbase)], Y, method="knn", trControl=control, metric="Mean_F1")
fit.knn$results
# SVM
set.seed(7)
fit.svm <- train(x = repbase[,2:ncol(repbase)], Y, method="svmRadial", trControl=control, metric="Mean_F1")
fit.svm$results
# Multi-Layer Perceptron
set.seed(7)
fit.mlp <- train(x = repbase[,2:ncol(repbase)], Y, method="mlp", trControl=control, metric="Mean_F1")
fit.mlp$results
# Random Forest
set.seed(7)
fit.rf <- train(x = repbase[,2:ncol(repbase)], Y, method="rf", trControl=control, metric="Mean_F1")
fit.rf$results
# AdaBoost Classification Trees
set.seed(7)
fit.dt <- train(x = repbase[,2:ncol(repbase)], Y, method="adaboost", trControl=control, metric="Mean_F1")
fit.dt$results
# Naive Bayes Classifier
set.seed(7)
fit.nb <- train(x = repbase[,2:ncol(repbase)], Y, method="nbDiscrete", trControl=control, metric="Mean_F1")
fit.nb$results

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

