---
title: "Group 12 HW2 Technical Deliverable"
author: "Ching-Fen Hung, Cora Goodwin, Tao Fang, Sam Benson Devine"
date: "2025-10-18"
output: pdf_document
---

```{r setup, include=FALSE, message=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

1. Introduction

- Main objective of the report : To generate a model that can accurately predict
if a customer will convert from a free user to a premium subscriber within the
next 6 months if they receive targeted promotion (adopters). 
As adopters are a very small portion of the sample, we are using the F-score as
our statistic of measure for the model. This balances the true positive rate
vs the true negative rate. Accuracy is not great predictor as it is too easy to
generate a model that accurately predicts non-adopters, while disregarding our
class of concern.

-Libraries load

```{r, echo=FALSE}
library(knitr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ranger)
library(caret)
library(pROC)
library(smotefamily)
```

- We will use the XYZ Data set for demonstration
```{r}
xyz = read.csv("XYZData.csv", header = TRUE)
# remove userID column
xyz = xyz[,-1]
```

2. EDA and data partition
- intial EDA
```{r}
# number of observations and variables
str(xyz)

# variable description
summary(xyz)

# data is clean and complete

# check class imbalance
sum(xyz$adopter)/nrow(xyz) # check class imbalance
```

- class of concern is low (3.707%), use smote to oversample minority class after setting aside 20% for validation and 10% for test set

```{r}
set.seed(6131)
testindex = createDataPartition(xyz$adopter, p = 0.1, list = FALSE)
test_xyz= xyz[testindex, ]
train_xyz = xyz[-testindex, ]
set.seed(6131)
valindex = createDataPartition(xyz$adopter, p = 0.22222, list = FALSE)
val_data = xyz[valindex,]
train_xyz=xyz[-valindex,]
smote_output = SMOTE(train_xyz, target = train_xyz$adopter, K = 5, dup_size = 0)
train_data = smote_output$data[,-27] # remove the last column added by smote
```

- eda on os data for comparison

```{r}
# number of observations and variables
str(train_data)

# variable description
summary(train_data)

# mean and std for oversampled is comparable to original data, free to proceed

# check class imbalance
sum(train_data$adopter)/nrow(train_data)
```
3. Model train and validation

3.1 KNN

- normalize data for knn

```{r}
# normalize data first
norm = function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
train_data_norm = as.data.frame(lapply(train_data, norm))
val_data_norm = as.data.frame(lapply(val_data, norm))
```

- generate and train knn model using 10-fold cross validation

```{r}
# set cross validation param for knn
control = trainControl(method="cv", number=10)
set.seed(6131)
knn_model = train(as.factor(adopter) ~ ., 
                  data = train_data_norm, 
                  method = "knn", 
                  trControl = control)

```

- predict on validation set, returning probability that each user is an adopter

```{r}
knn_val_pred= predict(knn_model, newdata = val_data_norm, type = "prob")
```

- find cutoff that maximizes F1 then return the confusion matrix for that cutoff

```{r}
F1s = c()
index = c()
for (i in (c(unique(knn_val_pred[,2])))) {
  F1 = confusionMatrix(as.factor(ifelse(knn_val_pred[,"1"] >= i, 1, 0)), 
                       as.factor(val_data$adopter),
                       mode = "prec_recall",
                       positive = "1")$byClass[7]
  if (is.nan(F1) == FALSE) {
    F1s = c(F1s,F1)
    index = c(index,i)
  }
}
cutoff = index[which.max(F1s)]
# confusion matrix for validation set with max F1
knn_cm = confusionMatrix(as.factor(ifelse(knn_val_pred[,"1"] >= cutoff, 1, 0)),
                         as.factor(val_data$adopter),
                         mode = "prec_recall",
                         positive = "1")
knn_cm
knn_cm$byClass[7] #F-score for model

```

- build ROC for knn validation set and report auc

```{r}
knn_roc_curve = roc(val_data_norm$adopter, knn_val_pred[,"1"])
plot(knn_roc_curve, main = "KNN Validation Set ROC Curve", col = "blue")
auc(knn_roc_curve)
```
3.2 Decision Tree

- generate and train knn model using 5-fold cross validation and complexity
  parameter of .0005 to ensure tree does not terminate too soon

```{r}
dt_control = rpart.control(xval = 5, cp = .0005)
set.seed(6131)
dt_model = rpart(as.factor(adopter) ~ .,
                 data = train_data,
                 method = "class",
                 control= dt_control)
```

- predict on validation set, returning probability that each user is an adopter

```{r}
dt_val_pred = predict(dt_model, newdata = val_data, type = "prob")
```

- find cutoff that maximizes F1 then return the confusion matrix for that cutoff

```{r}
# find cutoff that maximizes F1
F1s = c()
index = c()
for (i in (c(unique(dt_val_pred[,2])))) {
  F1 = confusionMatrix(as.factor(ifelse(dt_val_pred[,2] >= i, 1, 0)), 
                       as.factor(val_data$adopter),
                       mode = "prec_recall",
                       positive = "1")$byClass[7]
  if (is.nan(F1) == FALSE) {
    F1s = c(F1s,F1)
    index = c(index,i)
  }
}
# confusion matrix for validation set with max F1
dt_cm = confusionMatrix(as.factor(ifelse(dt_val_pred[,2] >= index[which.max(F1s)], 1, 0)),
                        as.factor(val_data$adopter),
                        mode = "prec_recall",
                        positive = "1")
dt_cm
dt_cm$byClass[7] #F-score for model

```

- build ROC for knn validation set and report auc

```{r}
dt_roc_curve = roc(val_data$adopter, dt_val_pred[,2])
plot(dt_roc_curve, main = "Decision Tree Validation Set ROC Curve", col = "red")
auc(dt_roc_curve)
```

3.3 Random Forest
- generate and train random forest model using 500 trees and 50:1 class weight 
  for adopters

```{r}
set.seed(6131)
rf_model = randomForest(as.factor(adopter) ~ ., 
                        data = train_data, ntree = 500, 
                        classwt = c(1,50) )
```

- predict on validation set

```{r}
rf_val_pred = predict(rf_model, newdata = val_data, type = "prob")
```

- find cutoff that maximizes F1 then return the confusion matrix for that cutoff

```{r}
# find cutoff that maximizes F1
F1s = c()
index = c()
for (i in (c(unique(rf_val_pred[,2])))) {
  F1 = confusionMatrix(as.factor(ifelse(rf_val_pred[,2] >= i, 1, 0)), 
                       as.factor(val_data$adopter),
                       mode = "prec_recall",
                       positive = "1")$byClass[7]
  if (is.nan(F1) == FALSE) {
    F1s = c(F1s,F1)
    index = c(index,i)
  }
}
cutoff = index[which.max(F1s)]
# confusion matrix for validation set with max F1
rf_cm = confusionMatrix(as.factor(ifelse(rf_val_pred[,2] >= cutoff, 1, 0)),
                        as.factor(val_data$adopter),
                        mode = "prec_recall",
                        positive = "1")
rf_cm
rf_cm$byClass[7] #F-score for model
```

- roc/auc for random forest validation set

```{r}
rf_roc_curve = roc(val_data$adopter, rf_val_pred[,2])
plot(rf_roc_curve, main = "Random Forest Validation Set ROC Curve", col = "green")
auc(rf_roc_curve)
```
3.4 Ranger (Fast Random Forest)
- generate and train ranger model using 500 trees and 10:1 class weight for adopters
  
```{r}
set.seed(6131)
rg_model = ranger(as.factor(adopter) ~ ., 
                  data = train_data, 
                  num.trees = 500, probability = TRUE ,
                  class.weights = c(1,10),
                  importance = "permutation")
```

- predict on validation set

```{r}
rg_val_pred = predict(rg_model, data = val_data, probability = "TRUE")$predictions
```

- find cutoff that maximizes F1 then return the confusion matrix for that cutoff

```{r}
# find cutoff that maximizes F1
F1s = c()
index = c()
for (i in (c(unique(rg_val_pred[,2])))) {
  F1 = confusionMatrix(as.factor(ifelse(rg_val_pred[,2] >= i , 1, 0)), 
                       as.factor(val_data$adopter),
                       mode = "prec_recall",
                       positive = "1")$byClass[7]
  if (is.nan(F1) == FALSE) {
    F1s = c(F1s,F1)
    index = c(index,i)
  }
}
cutoff = index[which.max(F1s)]
# confusion matrix for validation set with max F1
rg_cm = confusionMatrix(as.factor(ifelse(rg_val_pred[,2] >= cutoff, 1, 0)),
                        as.factor(val_data$adopter),
                        mode = "prec_recall",
                        positive = "1")
rg_cm
rg_cm$byClass[7] #F-score for model
```

- roc/auc for ranger validation set

```{r}
rg_roc_curve = roc(val_data$adopter, as.numeric(rg_val_pred[,2]))
plot(rg_roc_curve, main = "Ranger Validation Set ROC Curve", col = "forestgreen")
auc(rg_roc_curve)
```

Ranger performed best on validation set using F-score as statistic of measure 
but also had the highest AUC. Therefore we will use ranger for final model 
evaluation test set. First we will evaluate feature of importance in the
ranger model for final report.

3.4a Feature Importance from Ranger Model

```{r}
sort(importance(rg_model), decreasing = TRUE)
```


4. Test set evaluation
- ranger model performs best using our selected statistic, final model run on 
test data for report

```{r}
rg_test_pred = predict(rg_model, data = test_xyz, probability = "TRUE")$predictions
# find cutoff that maximizes F1
F1s = c()
index = c()
for (i in (c(unique(rg_test_pred[,2])))) {
  F1 = confusionMatrix(as.factor(ifelse(rg_val_pred[,2] >= i , 1, 0)), 
                       as.factor(val_data$adopter),
                       mode = "prec_recall",
                       positive = "1")$byClass[7]
  if (is.nan(F1) == FALSE) {
    F1s = c(F1s,F1)
    index = c(index,i)
  }
}
cutoff = index[which.max(F1s)]
# confusion matrix for validation set with max F1
rgt_cm = confusionMatrix(as.factor(ifelse(rg_test_pred[,2] >= cutoff, 1, 0)),
                        as.factor(test_xyz$adopter),
                        mode = "prec_recall",
                        positive = "1")
rgt_cm
rgt_cm$byClass[7] #F-score for model
```
- roc/auc for ranger test set

```{r}
rg_roc_curve_test = roc(test_xyz$adopter, as.numeric(rg_test_pred[,2]))
plot(rg_roc_curve_test, main = "Ranger Test Set ROC Curve", col = "forestgreen")
auc(rg_roc_curve_test)

arearatio = 2 * auc(rg_roc_curve_test) - 1
arearatio

```
- Gain (Cumulative Response) Curve for Test set

```{r}
# Combine the actual and predicted values
gain_data <- data.frame(
  actual = test_xyz$adopter,
  predicted = rg_test_pred[,2]
)

# Sort the data by predicted probabilities
gain_data <- gain_data[order(-gain_data$predicted), ]

# Compute the cumulative gain
gain_data$cumulative_actual <- cumsum(gain_data$actual == 1)
gain_data$cumulative_total <- 1:nrow(gain_data)
gain_data$cumulative_percentage <- gain_data$cumulative_actual / 
                                                       sum(gain_data$actual == "1")
gain_data$cumulative_total_percentage <- gain_data$cumulative_total / nrow(gain_data)

# Create a gain chart
ggplot(gain_data, aes(x = cumulative_total_percentage, y = cumulative_percentage)) +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(
    title = "Gain Chart",
    x = "Cumulative Percentage of Total",
    y = "Cumulative Gain"
  )
```
- Lift Curve for Test set

```{r}
rg_test_lift = test_xyz %>%
  mutate(prob = rg_test_pred[,2]) %>%
  arrange(desc(prob)) %>%
  mutate(cum_pos = cumsum(adopter),
         xaxis = row_number()/n(),
         lift = (cum_pos / row_number()) / (sum(adopter)/n())
         )

ggplot(rg_test_lift, aes(x = xaxis, y = lift)) +
  geom_line() +
  theme_bw() +
  labs(title = "Lift Curve", x = "Proportion of Sample", y = "Lift Ratio") +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red")
```