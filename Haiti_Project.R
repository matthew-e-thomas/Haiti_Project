Haiti <- read.csv("HaitiPixels.csv", header=TRUE)

#Hold Out Data Import and Clean Up
Tarp_67 <- read.csv("Hold+Out+Data/067Blue_Tarps.csv", header=T)
Tarp_67$tarp <- "Yes" #need to add column for whether it's a blue tarp or not
Tarp_67Not <- read.csv("Hold+Out+Data/067Not_Blue_Tarps.csv", header=T)
Tarp_67Not$tarp <- "No"
Tarp_057Not <- read.csv("Hold+Out+Data/057Not_Blue_Tarps.csv")
Tarp_057Not$tarp <- "No"
Tarp_069 <- read.csv("Hold+Out+Data/069Blue_Tarps.csv")
Tarp_069$tarp <- "Yes"
Tarp_069Not <- read.csv("Hold+Out+Data/069Not_Blue_Tarps.csv")
Tarp_069Not$tarp <- "No"
Tarp_078 <- read.csv("Hold+Out+Data/078Blue_Tarps.csv")
Tarp_078$tarp <- "Yes"
Tarp_078Not <- read.csv("Hold+Out+Data/078Not_Blue_Tarps.csv")
Tarp_078Not$tarp <- "No"
Test_Data <- rbind(Tarp_67, Tarp_67Not, Tarp_057Not, Tarp_069,
                   Tarp_069Not, Tarp_078, Tarp_078Not)
Test_Data$tarp <- factor(Test_Data$tarp, levels = c("Yes", "No"))

library(glmnet)
library(MASS)
library(caret)
library(kernlab)
library(dplyr)
library(ranger)
library(caTools)
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
model_logistic
plot(model_logistic)
confusionMatrix(model_logistic, "none")

#logistic model with multi classes
model_logistic_multi <- train(
  x = tarp_x, 
  y = tarp_y_multi,
  metric = "Mean_Sensitivity",
  method = "glmnet",
  trControl = myControl_multi
)
model_logistic_multi
confusionMatrix(model_logistic_multi, "none")
#two class model does better here

#run on test data
prob_logit <- predict(model_logistic, Test_Data[,1:3], type="prob")
Yes_or_No <- ifelse(prob_logit[1] > 0.6, "Yes", "No")
tarp_class <- factor(Yes_or_No, levels=levels(Test_Data[["tarp"]]))
confusionMatrix(tarp_class, Test_Data$tarp)
colAUC(prob_logit, Test_Data[["tarp"]], plotROC=T)

prob_logit_multi <- predict(model_logistic_multi, Test_Data[, 1:3])
table(prob_logit_multi, Test_Data$tarp)
#Interestingly, the two class model has slightly higher
#sensitivity, but the multi class has lower False Alarm Rate


#LDA Model
model_lda <- train(
  x = tarp_x,
  y = tarp_y,
  method="lda", 
  trControl = myControl
)
model_lda
confusionMatrix(model_lda, "none")
probs_lda <- predict(model_lda, Test_Data)
confusionMatrix(probs_lda, Test_Data$tarp)

#LDA Multiclass
model_lda_multi <- train(
  x = tarp_x,
  y = tarp_y_multi,
  method="lda", 
  trControl = myControl_multi
)
model_lda_multi
confusionMatrix(model_lda_multi, "none")

#multi model performs better for LDA, we'll assume for QDA as well

#test on hold-out data
lda_probs <- predict(model_lda_multi, Test_Data[, 1:3])
table(lda_probs, Test_Data$tarp)

#QDA Model
model_qda <- train(
  x = tarp_x,
  y = tarp_y_multi,
  method="qda", 
  trControl = myControl_multi
)
model_qda
confusionMatrix(model_qda, "none")
#Sensitivity-.865205  False Alarm Rate:.026629

#on hold-out data
qda_probs <- predict(model_qda, Test_Data[, 1:3])
table(qda_probs, Test_Data$tarp)

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

#test hold-out data
knn_probs1 <- predict(model_knn, Test_Data[, 1:3])
confusionMatrix(knn_probs1, Test_Data$tarp)

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
plot(model_knn, title="KNN Training Data")

#on hold-out data
knn_probs <- predict(model_knn_multi, Test_Data[, 1:3])
table(knn_probs, Test_Data$tarp)

#Support Vector Machines

#Linear kernal, two class
svmGrid <- expand.grid(C= 2^c(0:5))
model_svm1 <- train(
  x = tarp_x,
  y = tarp_y, 
  method = "svmLinear", 
  metric = "Sens",
  trControl = myControl, 
  preProcess = c("center", "scale"),
  tuneGrid = svmGrid
)
model_svm1
confusionMatrix(model_svm1, "none")
#Sensitivity = .8766348, False Alarm = .000904
plot(model_svm1)

#test on hold out
svm_probs_linear <- predict(model_svm1, Test_Data[, 1:3])
confusionMatrix(svm_probs_linear, Test_Data$tarp)

svmGrid_poly <- expand.grid(C = 2^c(0:5), degree=c(2, 3, 4), scale=T)
#Polynomial kernel, two class
model_svm1_poly <- train(
  x = tarp_x,
  y = tarp_y, 
  method = "svmPoly", 
  metric = "Sens",
  trControl = myControl, 
  preProcess = c("center", "scale"),
  tuneGrid = svmGrid_poly
)
model_svm1_poly
confusionMatrix(model_svm1_poly, "none")
#Detection Rate=0.9082862, False Alarm Rate=.001321

#Polynomial kernel, multi-class
model_svm1_poly_multi <- train(
  x = tarp_x,
  y = tarp_y_multi, 
  method = "svmPoly", 
  metric = "Mean_Sensitivity",
  trControl = myControl_multi, 
  preProcess = c("center", "scale"),
  tuneGrid = svmGrid_poly
)
model_svm1_poly_multi #degree 4, C=2
confusionMatrix(model_svm1_poly_multi,"none")
#Detection Rate = .930432, False Alarm Rate = 0.001236

#Radial kernel, multi-class
svmGrid_radial <- expand.grid(C = 2^c(0:3), sigma= 2^c(-25, -20, -15,-10, -5, 0))
model_svm1_radial_multi <- train(
  x = tarp_x,
  y = tarp_y_multi, 
  method = "svmRadial", 
  metric = "Mean_Sensitivity",
  trControl = myControl_multi, 
  preProcess = c("center", "scale"),
  tuneGrid = svmGrid_radial
)
model_svm1_radial_multi
confusionMatrix(model_svm1_radial_multi, "none")
#Best model c=8 and sigma=1
#detection rate = .939993, False Alarm Rate = .001822
plot(model_svm1_radial_multi)
#test on hold out data
svm_probs <- predict(model_svm1_radial_multi, Test_Data[, 1:3])
table(svm_probs, Test_Data$tarp)


#Random Forest

#Two class random forest
model_randomforest <- train(
  x = tarp_x,
  y = tarp_y, 
  method = "ranger", 
  metric = "Sens",
  trControl = myControl
)
model_randomforest
#Detection Rate=.9388399, False Alarm Rate=0.001771

#on hold-out data
rforest_probs <- predict(model_randomforest, Test_Data[, 1:3])
confusionMatrix(rforest_probs, Test_Data$tarp)

#random forest multi class
model_randomforest_multi <- train(
  x = tarp_x,
  y = tarp_y_multi, 
  method = "ranger", 
  metric = "Mean_Sensitivity",
  trControl = myControl_multi
)
model_randomforest_multi
plot(model_randomforest_multi)
confusionMatrix(model_randomforest_multi, "none")
#detection rate=.950874, False Alarm Rate = .002325
probs_rf <- predict(model_randomforest_multi, Test_Data[, 1:3])
table(probs_rf, Test_Data$tarp)
#detection rate=.803798  False Alarm Rate=.007292

