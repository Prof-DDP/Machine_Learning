# Data Preprocessing

# Importing dataset

dataset = read.csv('Data.csv')
#dataset = dataset[, 2:3]

# Splitting up training and test set
#install.packages('caTools')
library(caTools) #imports caTools
set.seed(123)

#SplitRatio is for training set, TRUE for training data & FALSE for testing data

split = sample.split(dataset$Purchased, SplitRatio=0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

# Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(training_set[, 2:3])




