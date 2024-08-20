# Importing the dhfr dataset
library(datasets)
library(caret)
data(dhfr)

# Finding basic statistics
summary(dhfr)
summary(dhfr$Y)

# Checking if there is missing data
sum(is.na(dhfr))

# Expanding basic statistics
library(skimr)
skim(dhfr)

 # Expanding basic statistics after grouping data by biological activity
dhfr %>%
  dplyr::group_by(Y) %>%
  skim()

# Scatter plot
plot(dhfr$moe2D_zagreb, dhfr$moe2D_weinerPol, col = "red", xlab = "moe2D_zagreb", ylab = "moe2D_weinerPol")

# Feature plots
library(caret)
featurePlot(x = dhfr[,2:21],
            y = dhfr$Y,
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"),
                          y = list(relation="free")))

# Achieving reproducible model
set.seed(100)

# Performing stratified random split of the data set
TrainingIndex <- createDataPartition(dhfr$Y, p=0.8, list = FALSE)
TrainingSet <- dhfr[TrainingIndex,] # Training Set
TestingSet <- dhfr[-TrainingIndex,] # Test Set

# Building a Training model
Model <- train(Y ~ ., data = TrainingSet,
               method = "svmPoly",
               na.action = na.omit,
               preProcess=c("scale","center"),
               trControl= trainControl(method="none"),
               tuneGrid = data.frame(degree=1,scale=1,C=1))

# Removing zero variance variables
zero_var_cols <- nearZeroVar(TrainingSet)
TrainingSet <- TrainingSet[, -zero_var_cols]

# Building a CV model
Model.cv <- train(Y ~ ., data = TrainingSet,
                  method = "svmPoly",
                  na.action = na.omit,
                  preProcess=c("scale","center"),
                  trControl= trainControl(method="cv", number=10),
                  tuneGrid = data.frame(degree=1,scale=1,C=1))


# Applying model for prediction
Model.training <-predict(Model, TrainingSet) 
Model.testing <-predict(Model, TestingSet)

# Performing cross-validation
Model.cv <-predict(Model.cv, TrainingSet) 

# Model performance - displaying confusion matrix and statistics
Model.training.confusion <-confusionMatrix(Model.training, TrainingSet$Y)
Model.testing.confusion <-confusionMatrix(Model.testing, TestingSet$Y)
Model.cv.confusion <-confusionMatrix(Model.cv, TrainingSet$Y)

print(Model.training.confusion)
print(Model.testing.confusion)
print(Model.cv.confusion)

# Featuring importance
Importance <- varImp(Model)
plot(Importance, top = 25)
plot(Importance, col = "red")

# Leaving the output variable out
dhfr <- dhfr[,-1]

# Randomly introducing NA to the dataset
na.gen <- function(data,n) {
  i <- 1
  while (i < n+1) {
    idx1 <- sample(1:nrow(data), 1)
    idx2 <- sample(1:ncol(data), 1)
    data[idx1,idx2] <- NA
    i = i+1}
  return(data)}

dhfr <- na.gen(dhfr,100)
# Alternative
dhfr <- na.gen(n=100,data=dhfr)

# Checking for missing data again
sum(is.na(dhfr))
colSums(is.na(dhfr))
str(dhfr)

# Listing rows with missing data
missingdata <- dhfr[!complete.cases(dhfr), ]
sum(is.na(missingdata))

# Handling missing data - two alternatives

# 1. Deleting all entries with missing data
clean.data <- na.omit(dhfr)
sum(is.na(clean.data))

# 2. Imputation: Replacing missing values with the column's 
# MEAN
dhfr.impute <- dhfr

for (i in which(sapply(dhfr.impute, is.numeric))) { 
  dhfr.impute[is.na(dhfr.impute[, i]), i] <- mean(dhfr.impute[, i],  na.rm = TRUE) 
}

sum(is.na(dhfr.impute))

# MEDIAN
dhfr.impute <- dhfr

for (i in which(sapply(dhfr.impute, is.numeric))) { 
  dhfr.impute[is.na(dhfr.impute[, i]), i] <- median(dhfr.impute[, i],  na.rm = TRUE) 
}

sum(is.na(dhfr.impute))

# Performing stratified random split of the data set
TrainingIndex <- createDataPartition(dhfr$Y, p=0.8, list = FALSE)
TrainingSet <- dhfr[TrainingIndex,]
TestingSet <- dhfr[-TrainingIndex,] 

# Random forest
# Running normally without parallel processing
# Building model using training set and learning the algorithm
start.time <- proc.time()
Model <- train(Y ~ ., 
               data = TrainingSet, 
               method = "rf" 
         )
stop.time <- proc.time()
run.time <- stop.time - start.time
print(run.time)

# Using parallel processing
library(doParallel)

cl <- makePSOCKcluster(5)
registerDoParallel(cl)

start.time <- proc.time()
Model <- train(Y ~ ., 
               data = TrainingSet, 
               method = "rf" 
         )
stop.time <- proc.time()
run.time <- stop.time - start.time
print(run.time)

stopCluster(cl)

# Run without parallel processing again
start.time <- proc.time()
Model <- train(Y ~ ., 
               data = TrainingSet, 
               method = "rf", 
               tuneGrid = data.frame(mtry = seq(5,15, by=5))
         )
stop.time <- proc.time()
run.time <- stop.time - start.time
print(run.time)

# Using doParallel
library(doParallel)

cl <- makePSOCKcluster(5)
registerDoParallel(cl)

start.time <- proc.time()
Model <- train(Y ~ ., 
               data = TrainingSet, 
               method = "rf",
               tuneGrid = data.frame(mtry = seq(5,15, by=5))
         )
stop.time <- proc.time()
run.time <- stop.time - start.time
print(run.time)

stopCluster(cl)

# Applying model for prediction on Training set
Model.training <-predict(Model, TrainingSet) 

# Model performance
Model.training.confusion <-confusionMatrix(Model.training, TrainingSet$Y)

print(Model.training.confusion)

# Feature importance
Importance <- varImp(Model)
plot(Importance, top = 25)
plot(Importance, col = "red")
