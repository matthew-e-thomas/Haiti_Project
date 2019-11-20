Haiti <- read.csv("HaitiPixels.csv", header=TRUE)

library(glmnet)
library(caret)
library(dplyr)
set.seed(1001)


tarp_y <- Haiti$Class
tarp_x <- Haiti[, -1]

myFolds <- createFolds(tarp_y, k = 10)

# Create reusable trainControl object: myControl
myControl <- trainControl(
  summaryFunction = multiClassSummary,
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
