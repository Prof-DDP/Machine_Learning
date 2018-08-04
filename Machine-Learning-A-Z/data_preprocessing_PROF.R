# Data Preprocessing

# Importing dataset

'''
NOTE: The first element of something in R is 1 instead of 0 
like in python. That\'ll be important to keep in mind when working
with the dataset
'''

dataset = read.csv('Data.csv')

# Handing missing data

#replaces the missing data in the age column w/ the average of the column in the weird, R way
#dataset$Age essentially means dataset @ column Age
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

#same thing for salary column
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm=TRUE)),
                        dataset$Salary)

# Encoding categorical data

#c turns things into vectors
dataset$Country = factor(dataset$Country, 
                         levels= c('France', 'Spain', 'Germany'),
                         labels= c(1, 3, 2))

dataset$Purchased = factor(dataset$Purchased, 
                         levels= c('No', 'Yes'),
                         labels= c(0,1))

# Splitting up training and test set
#install.packages('caTools')
library(caTools) #imports caTools
set.seed(123)

#SplitRatio is for training set, TRUE for training data & FALSE for testing data

split = sample.split(dataset$Purchased, SplitRatio=0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

# Feature Scaling
training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(training_set[, 2:3])