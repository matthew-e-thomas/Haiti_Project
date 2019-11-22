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
library(kernlab)
library(dplyr)
set.seed(1001)

Haiti$tarp <- ifelse(Haiti$Class == "Blue Tarp", "Yes", "No")

#Fits for two class models
tarp_y <- Haiti$tarp
tarp_x <- Haiti[, 2:4]
tarp_y <- factor(tarp_y, levels = c("Yes", "No"))

#fits for multi class models
tarp_y_multi <- Haiti$Class
tarp_y_multi <- as.factor(tarp_y_multi)

#create folds so that we can compare models on same CV set
myFolds <- createFolds(tarp_y, k = 10)
myFolds_multi <- createFolds(tarp_y_multi, k = 10)

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
  index = myFolds_multi
)

levels(tarp_y_multi) <- c("Blue.Tarp", "Rooftop", "Soil", "Various.Non.Tarp",
                    "Vegetation")

#run logistic model with two class response
model_logistic <- train(
  x = tarp_x, 
  y = tarp_y,
  metric = "Sens",
  method = "glmnet",
  trControl = myControl
)
confusionMatrix(model_logistic, "none")

#logistic model with multi classes
model_logistic_multi <- train(
  x = tarp_x, 
  y = tarp_y_multi,
  metric = "Mean_Sensitivity",
  method = "glmnet",
  trControl = myControl_multi
)

confusionMatrix(model_logistic_multi, "none")
#two class model does better here

#LDA Model
model_lda <- train(
  x = tarp_x,
  y = tarp_y,
  method="lda", 
  trControl = myControl
)
confusionMatrix(model_lda, "none")

#LDA Multiclass
model_lda_multi <- train(
  x = tarp_x,
  y = tarp_y_multi,
  method="lda", 
  trControl = myControl_multi
)
confusionMatrix(model_lda_multi, "none")

#multi model performs better for LDA, we'll assume for QDA as well

#QDA Model
model_qda <- train(
  x = tarp_x,
  y = tarp_y_multi,
  method="qda", 
  trControl = myControl_multi
)
confusionMatrix(model_qda, "none")
#Sensitivity-.79335  False Alarm Rate:.026629


#KNN Model

#two class model
model_knn <- train(
  x = tarp_x,
  y = tarp_y, 
  method = "knn", 
  metric = "Sens",
  trControl = myControl, 
  tuneLength = 20
  )
model_knn
confusionMatrix(model_knn, "none")
#detection rate: .9362, False Alarm Rate: .001855

#knn using multiple classes
model_knn_multi <- train(
  x = tarp_x,
  y = tarp_y_multi, 
  method = "knn", 
  metric = "Mean_Sensitivity",
  trControl = myControl_multi, 
  preProcess = c("center", "scale"),
  tuneLength = 20
)
model_knn_multi
confusionMatrix(model_knn_multi, "none")
#detection rate:  .9446, False Alarm Rate:  .022
#Multi-class knn works slightly better with centering & scaling


#Support Vector Machines

#Linear kernal, two class
svmGrid <- expand.grid(C= 2^c(0:5))
model_svm1 <- train(
  x = tarp_x,
  y = tarp_y, 
  method = "svmLinear", 
  metric = "Sens"
  trControl = myControl, 
  preProcess = c("center", "scale"),
  tuneGrid = svmGrid
)
model_svm1
confusionMatrix(model_svm1, "none")
