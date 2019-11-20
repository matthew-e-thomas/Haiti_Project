Haiti <- read.csv("HaitiPixels.csv", header=TRUE)

#Hold Out Data Import and Clean Up
Tarp_67 <- read.csv("Hold+Out+Data/067Blue_Tarps.csv", header=T)
Tarp_67$tarp <- "Yes"

library(glmnet)
library(MASS)
library(caret)
library(dplyr)
set.seed(1001)

Haiti$tarp <- ifelse(Haiti$Class == "Blue Tarp", "Yes", "No")

tarp_y <- Haiti$tarp
tarp_x <- Haiti[, 2:4]
tarp_y <- as.factor(tarp_y)

myFolds <- createFolds(tarp_y, k = 10)

# Create reusable trainControl object to compare models
myControl <- trainControl(
  summaryFunction = twoClassSummary,
  method="cv",
  classProbs = TRUE,
  verboseIter = TRUE,
  savePredictions = TRUE,
  index = myFolds
)

levels(tarp_y) <- c("Blue.Tarp", "Rooftop", "Soil", "Various.Non.Tarp",
                    "Vegetation")

model_logistic <- train(
  x = tarp_x, 
  y = tarp_y,
  metric = "ROC",
  method = "glmnet",
  trControl = myControl
)

#LDA Model
model_lda <- train(
  x = tarp_x,
  y = tarp_y,
  method="lda", 
  trControl = myControl
)


#QDA Model
model_qda <- train(
  x = tarp_x,
  y = tarp_y,
  method="qda", 
  trControl = myControl
)

#KNN Model
model_knn <- train(
  x = tarp_x,
  y = tarp_y, 
  method = "knn", 
  trControl = myControl, 
  tuneLength = 20
  )
