#Multiple linear regression

# Data Preprocessing

# Importing dataset

dataset = read.csv('50_Startups.csv')
#dataset = dataset[, 2:3]

# Encoding categorical data
dataset$State = factor(dataset$State,
                         levels = c('New York', 'California', 'Florida'),
                         labels = c(1, 2, 3))

# Splitting up training and test set
#install.packages('caTools')
library(caTools) #imports caTools
set.seed(123)

#SplitRatio is for training set, TRUE for training data & FALSE for testing data

split = sample.split(dataset$Profit, SplitRatio=0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

# Fitting Multiple Linear Regression to Training set
regressor = lm(formula = Profit ~ .,
               data = training_set) #express profit as linear combination of all independant variables

# Predicting Test set results
y_pred = predict(regressor, newdata=test_set)

# Building optimal model using Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, #removing variable w/ highest P-value
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend, #removing variable w/ highest P-value
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend, #removing variable w/ highest P-value
               data = dataset)
summary(regressor)










