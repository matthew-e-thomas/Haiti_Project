Haiti <- read.csv("HaitiPixels.csv", header=TRUE)

library(glmnet)
library(caret)
library(dplyr)
set.seed(1001)

Haiti$tarp <- ifelse(Haiti$Class == "Blue Tarp", "Yes", "No")

tarp_y <- Haiti$tarp
tarp_x <- Haiti[, 2:4]
tarp_y <- as.factor(tarp_y)

myFolds <- createFolds(tarp_y, k = 10)

# Create reusable trainControl object: myControl
myControl <- trainControl(
  summaryFunction = twoClassSummary,
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
