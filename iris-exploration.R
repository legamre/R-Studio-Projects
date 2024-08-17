# Importing the Iris dataset
library(datasets)
data("iris")
iris <- datasets::iris

# Data exploration
View(iris)
iris$Petal.Length
iris$Species
Species <- iris$Species
Species
head(iris, 7)
tail(iris,7)

# Finding basic statistics
summary(iris)
summary(iris$Sepal.Length)

# Checking if there is missing data
sum(is.na(iris))

# Expanding basic statistics
library(skimr)
skim(iris)

# Expanding basic statistics after grouping data by species
iris %>%
dplyr::group_by(Species) %>%
skim()

# Panel plots
plot(iris, col = "red")

# Scatter plot
plot(iris$Sepal.Width, iris$Sepal.Length, col = "red", xlab = "Sepal width", ylab = "Sepal length")

# Histogram
hist(iris$Sepal.Width, col = "red")  

# Feature plots
library(caret)
featurePlot(x = iris[,1:4], 
            y = iris$Species, 
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))


# Machine learning in R
set.seed(100)

# Performs stratified random split of the data set
TrainingIndex <- createDataPartition(iris$Species, p=0.8, list = FALSE)
TrainingSet <- iris[TrainingIndex,] # Training Set
TestingSet <- iris[-TrainingIndex,] # Test Set

# Compare scatter plot of the 80 and 20 data subsets




###############################
# SVM model (polynomial kernel)

# Build Training model
Model <- train(Species ~ ., data = TrainingSet,
               method = "svmPoly",
               na.action = na.omit,
               preProcess=c("scale","center"),
               trControl= trainControl(method="none"),
               tuneGrid = data.frame(degree=1,scale=1,C=1)
)

# Build CV model
Model.cv <- train(Species ~ ., data = TrainingSet,
                  method = "svmPoly",
                  na.action = na.omit,
                  preProcess=c("scale","center"),
                  trControl= trainControl(method="cv", number=10),
                  tuneGrid = data.frame(degree=1,scale=1,C=1))


# Apply model for prediction
Model.training <-predict(Model, TrainingSet) # Apply model to make prediction on Training set
Model.testing <-predict(Model, TestingSet) # Apply model to make prediction on Testing set
Model.cv <-predict(Model.cv, TrainingSet) # Perform cross-validation

# Model performance (Displays confusion matrix and statistics)
Model.training.confusion <-confusionMatrix(Model.training, TrainingSet$Species)
Model.testing.confusion <-confusionMatrix(Model.testing, TestingSet$Species)
Model.cv.confusion <-confusionMatrix(Model.cv, TrainingSet$Species)

print(Model.training.confusion)
print(Model.testing.confusion)
print(Model.cv.confusion)

# Feature importance
Importance <- varImp(Model)
plot(Importance, col = "red")
