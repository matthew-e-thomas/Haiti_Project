Haiti <- read.csv("HaitiPixels.csv", header=TRUE)

#Hold Out Data Import and Clean Up
Tarp_67 <- read.csv("Hold+Out+Data/067Blue_Tarps.csv", header=T)
Tarp_67$tarp <- "Yes" #need to add column for whether it's a blue tarp or not
Tarp_67Not <- read.csv("Hold+Out+Data/067Not_Blue_Tarps.csv", header=T)
Tarp_67Not$tarp <- "No"
Test_Data <- rbind(Tarp_67, Tarp_67Not)

library(glmnet)
library(MASS)
library(caret)
library(dplyr)
set.seed(1001)

Haiti$tarp <- ifelse(Haiti$Class == "Blue Tarp", "Yes", "No")

#Fits for two class models
tarp_y <- Haiti$tarp
tarp_x <- Haiti[, 2:4]
tarp_y <- as.factor(tarp_y)

#fits for multi class models
tarp_y_multi <- Haiti$Class
tarp_x <- Haiti[, 2:4]
tarp_y_multi <- as.factor(tarp_y_multi)

myFolds <- createFolds(tarp_y_multi, k = 10)

# Create reusable trainControl object to compare models
myControl <- trainControl(
  summaryFunction = twoClassSummary,
  method="cv",
  classProbs = TRUE,
  verboseIter = TRUE,
  savePredictions = TRUE,
  index = myFolds
)

#trainControl for multi class
myControl_multi <- trainControl(
  summaryFunction = multiClassSummary,
  method="cv",
  classProbs = TRUE,
  verboseIter = TRUE,
  savePredictions = TRUE,
  index = myFolds
)

levels(tarp_y_multi) <- c("Blue.Tarp", "Rooftop", "Soil", "Various.Non.Tarp",
                    "Vegetation")

model_logistic <- train(
  x = tarp_x, 
  y = tarp_y_multi,
  metric = "Specificity",
  method = "glmnet",
  trControl = myControl
)
confusionMatrix(model_logistic, "none")

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
