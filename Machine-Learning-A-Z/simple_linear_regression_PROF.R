# Simple Linear Regression

# Importing dataset

dataset = read.csv('Salary_Data.csv')
#dataset = dataset[, 2:3]

# Splitting up training and test set
#install.packages('caTools')
library(caTools) #imports caTools
set.seed(123)

#SplitRatio is for training set, TRUE for training data & FALSE for testing data

split = sample.split(dataset$Salary, SplitRatio=2/3)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

# Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(training_set[, 2:3])

# Fitting model to Training set

#Salary ~ YearsExperience means Salary is porportional to years experience

regressor = lm(formula=Salary ~ YearsExperience,
               data=training_set)

# Predicting the Test set results
y_pred = predict(regressor, newdata=test_set)

# Visualizing the Training set resutls
#install.packages('ggplot2')
library(ggplot2)
ggplot()+
  geom_point(aes(x=training_set$YearsExperience, y=training_set$Salary),
             colour='red') + 
  geom_line(aes(x=training_set$YearsExperience, y=predict(regressor, newdata=training_set)),
            colour='blue') +
  ggtitle('Salary vs. Experience (Training set)') +
  xlab('Years of experience') +
  ylab('Salary')

# Visualizing test set
ggplot()+
  geom_point(aes(x=test_set$YearsExperience, y=test_set$Salary),
             colour='red') + 
  geom_line(aes(x=training_set$YearsExperience, y=predict(regressor, newdata=training_set)),
            colour='blue') +
  ggtitle('Salary vs. Experience (Training set)') +
  xlab('Years of experience') +
  ylab('Salary')

