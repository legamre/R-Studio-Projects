# Loading necessary libraries
library(datasets)
library(skimr)
library(caret)
library(dplyr)

# Importing the Iris dataset
data("iris")
iris <- datasets::iris

# Data exploration
View(iris)
str(iris)
names(iris)
head(iris, 7)
tail(iris, 7)

# Basic statistics
summary(iris)
summary(iris$Sepal.Length)

# Checking if there is missing data
sum(is.na(iris))

# Expanding basic statistics
skim(iris)

# Expanding basic statistics after grouping data by species
iris %>%
  dplyr::group_by(Species) %>%
  skimr::skim()

# Panel plots
plot(iris, col = "red")

# Scatter plot
plot(iris$Sepal.Width, iris$Sepal.Length, col = "red", 
     xlab = "Sepal width", ylab = "Sepal length", 
     main = "Sepal Width vs Length")

# Histogram
hist(iris$Sepal.Width, col = "red", main = "Histogram of Sepal Width")

# Feature plots
featurePlot(x = iris[, 1:4], 
            y = iris$Species, 
            plot = "box",
            strip = strip.custom(par.strip.text = list(cex = .7)),
            scales = list(x = list(relation = "free"), 
                          y = list(relation = "free")))

# Machine learning in R
set.seed(100)

# Performs stratified random split of the data set
TrainingIndex <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
TrainingSet <- iris[TrainingIndex,] # Training Set
TestingSet <- iris[-TrainingIndex,] # Test Set

# SVM model (polynomial kernel)
# Build Training model with cross-validation
Model <- train(Species ~ ., data = TrainingSet,
               method = "svmPoly",
               na.action = na.omit,
               preProcess = c("scale", "center"),
               trControl = trainControl(method = "cv", number = 10),
               tuneGrid = data.frame(degree = 1, scale = 1, C = 1))

# Apply model for prediction on Training set
Model.training.predictions <- predict(Model, TrainingSet)

# Apply model for prediction on Testing set
Model.testing.predictions <- predict(Model, TestingSet)

# Model performance (Displays confusion matrix and statistics)
Model.training.confusion <- confusionMatrix(Model.training.predictions, TrainingSet$Species)
Model.testing.confusion <- confusionMatrix(Model.testing.predictions, TestingSet$Species)

print(Model.training.confusion)
print(Model.testing.confusion)

# Feature importance
Importance <- varImp(Model)
plot(Importance, col = "red")
