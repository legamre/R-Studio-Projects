# Importing libraries
library(mlbench) 
library(caret) 

# Importing the Boston Housing dataset
data(BostonHousing)
head(BostonHousing)

# Checking if there is missing data
sum(is.na(BostonHousing))

# Achieving reproducible model
set.seed(100)

# Performing stratified random split of the data set
TrainingIndex <- createDataPartition(BostonHousing$medv, p=0.8, list = FALSE)
TrainingSet <- BostonHousing[TrainingIndex,] # Training Set
TestingSet <- BostonHousing[-TrainingIndex,] # Test Set

# Build Training model
Model <- train(medv ~ ., data = TrainingSet,
               method = "lm",
               na.action = na.omit,
               preProcess = c("scale", "center"),
               trControl = trainControl(method = "none")
)

# Applying model for prediction
Model.training <- predict(Model, TrainingSet) # Apply model to make predictions on the Training set
Model.testing <- predict(Model, TestingSet)   # Apply model to make predictions on the Testing set

# Model performance (Displays scatter plot and performance metrics)
# Scatter plot of Training set
plot(TrainingSet$medv, Model.training, 
     col = "blue", 
     xlab = "Actual MEDV", 
     ylab = "Predicted MEDV", 
     main = "Training Set: Actual vs Predicted")
abline(0, 1, col = "red") # Adds a y=x line for reference

# Scatter plot of Testing set
plot(TestingSet$medv, Model.testing, 
     col = "blue", 
     xlab = "Actual MEDV", 
     ylab = "Predicted MEDV", 
     main = "Testing Set: Actual vs Predicted")
abline(0, 1, col = "red") # Adds a y=x line for reference
