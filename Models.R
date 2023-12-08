library(tidyverse)
library(scales)
library(pROC)
library(broom)
library(caret)
library(class)
library(randomForest)
library(ranger)

# Read Train and Test Data
client.train = read_csv("C:/Users/user/Documents/Zindi_Artificial_Intelligence_Challenge_Beginner/client_train.csv")
client.test = read_csv("C:/Users/user/Documents/Zindi_Artificial_Intelligence_Challenge_Beginner/client_test.csv")

invoice.train = read_csv("C:/Users/user/Documents/Zindi_Artificial_Intelligence_Challenge_Beginner/invoice_train.csv")
invoice.test = read_csv("C:/Users/user/Documents/Zindi_Artificial_Intelligence_Challenge_Beginner/invoice_test.csv")

#######################################################################

set.seed(790)

glimpse(client.train)
summary(client.train)
summary(client.test)

glimpse(invoice.train)
summary(invoice.train)
summary(invoice.test)

invoice.train$counter_statue = invoice.train$counter_statue %>%
  replace_na(0)

consumption_stats.train = invoice.train %>%
  group_by(client_id) %>%
  summarize(count = n(),
            consommation_level_1_mean = mean(consommation_level_1)+small,
            consommation_level_2_mean = mean(consommation_level_2)+small,
            consommation_level_3_mean = mean(consommation_level_3)+small,
            consommation_level_4_mean = mean(consommation_level_4)+small)

summary(consumption_stats.train)

consumption_stats.test = invoice.test %>%
  group_by(client_id) %>%
  summarize(count = n(),
            consommation_level_1_mean = mean(consommation_level_1)+small,
            consommation_level_2_mean = mean(consommation_level_2)+small,
            consommation_level_3_mean = mean(consommation_level_3)+small,
            consommation_level_4_mean = mean(consommation_level_4)+small)

summary(consumption_stats.test)

train = client.train %>%
  inner_join(consumption_stats.train, by = "client_id") %>%
  mutate(disrict = as.factor(disrict),
         client_catg = as.factor(client_catg),
         target = factor(target)) %>%
  filter(consommation_level_1_mean < 35000,
         consommation_level_2_mean < 45000,
         consommation_level_3_mean < 10000,
         consommation_level_4_mean < 40000)

levels(train$target) = c("No", "Yes")

normalize = function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

train$consommation_level_1_mean = normalize(train$consommation_level_1_mean)
train$consommation_level_2_mean = normalize(train$consommation_level_2_mean)
train$consommation_level_3_mean = normalize(train$consommation_level_3_mean)
train$consommation_level_4_mean = normalize(train$consommation_level_4_mean)

test = client.test %>%
  inner_join(consumption_stats.test, by = "client_id") %>%
  mutate(disrict = as.factor(disrict),
         client_catg = as.factor(client_catg))

str(train)
str(test)

sample_size = sample(1:nrow(train), size = nrow(train)*0.8)

train.train = train[sample_size,]
train.test = train[-sample_size,]

# Chosen Features: 
# - disrict (factor)
# - client_catg (factor)
# - region (numeric)
# - consommation_level_1_mean
# - consommation_level_2_mean
# - consommation_level_3_mean
# - consommation_level_4_mean

small = 0.00000000001

formula = as.formula(target ~ disrict + client_catg + consommation_level_1_mean + consommation_level_2_mean + consommation_level_3_mean + consommation_level_4_mean)
formula_log = as.formula(target ~ disrict + client_catg + log10(consommation_level_1_mean) + log10(consommation_level_2_mean) + log10(consommation_level_3_mean) + log10(consommation_level_4_mean))

#######################################################################

# Logistic Regression Model (with & without log scaling)

model.glm = glm(formula = formula_test, data = train.train, family = "binomial")
model.glm.log10 = glm(formula = formula_log, data = train.train, family = "binomial")

prob.glm = predict(model.glm, train.test, type = "response")
prob.glm.log = predict(model.glm.log10, train.test, type = "response")

pred.glm = ifelse(prob.glm > 0.1, "Yes", "No")
pred.glm.log = ifelse(prob.glm.log > 0.1, "Yes", "No")

confusionMatrix(as.factor(pred.glm.log), train.test$target)

table(pred.glm, train.test$target)
table(pred.glm.log, train.test$target)

glance(model.glm)
glance(model.glm.log10)

ROC.glm = roc(train.test$target, prob.glm)
ROC.glm.log = roc(train.test$target, prob.glm.log)

plot(ROC.glm)
plot(ROC.glm.log)

auc(ROC.glm)
auc(ROC.glm.log)

train.test$pred.glm.log = pred.glm.log
train.test$prob.glm.log = prob.glm.log

#######################################################################

# Logistic Regression with caret

control = trainControl(method = "cv", number = 10, classProbs = T, savePredictions = T,
                       returnResamp = "none", summaryFunction = twoClassSummary, verboseIter = T)

model.caret.glm = train(form = formula, data = train.train, method = "glm",
                        trControl = control, metric = "ROC")

prob.caret.glm = predict(model.caret.glm, train.test, type = "prob")[, 2]

pred.caret.glm = ifelse(prob.caret.glm > 0.1, "Yes", "No")

table(Predicted = pred.caret.glm, Actual = train.test$target)

ROC.caret.glm = roc(train.test$target, prob.caret.glm)
plot(ROC.caret.glm)
auc(ROC.caret.glm)

#######################################################################

# K Nearest Neighbor

train.knn = train.train %>%
  select(target, region, client_catg, disrict, starts_with("consommation"))

test.knn = train.test %>%
  select(target, region, client_catg, disrict, starts_with("consommation"))

pred.knn = knn(train.knn[, -1], test.knn[, -1], cl = train.knn$target, k = 1, prob = TRUE)

confusionMatrix(pred.knn, train.test$target)

ROC.knn = roc(train.test$target, attr(pred.knn, "prob"))
plot(ROC.knn)
auc(ROC.knn)

#######################################################################

# Random Forest

model.rf = randomForest(formula, data = train.train,
                        ntree = 250, mtry = 2)

prob.rf = predict(model.rf, train.test, type = "prob")[,2]

pred.rf = ifelse(prob.rf > 0.1, "Yes", "No")

table(Predicted = pred.rf, Actual = train.test$target)

ROC.rf = roc(train.test$target, prob.rf)
plot(ROC.rf)
auc(ROC.rf)

#######################################################################

# Random Forest with caret (ranger)

control.rf = trainControl(method = "cv", number = 5, verboseIter = T,
                          classProbs = T, summaryFunction = twoClassSummary, savePredictions = T,
                          returnResamp = "none")

tuneGrid.rf = data.frame(mtry = c(2,4,6,8), splitrule = "gini", min.node.size = 10)

model.caret.rf  = train(formula, data = train.train, method = "ranger",
                        trControl = control.rf,
                        tuneGrid = tuneGrid.rf)

model.caret.rf
plot(model.caret.rf)

prob.caret.rf = predict(model.caret.rf, train.test, type = "prob")[, 2]

pred.caret.rf = ifelse(prob.caret.rf > 0.05, "Yes", "No")

confusionMatrix(as.factor(pred.caret.rf), train.test$target)

table(Predicted = pred.caret.rf, Actual = train.test$target)

ROC.caret.rf = roc(train.test$target, prob.caret.rf)
plot(ROC.caret.rf)
auc(ROC.caret.rf)

train.test$pred.caret.rf = pred.caret.rf
train.test$prob.caret.rf = prob.caret.rf

#######################################################################

# glmnet with caret

control.glmnet = trainControl(method = "cv", number = 5, verboseIter = T,
                              summaryFunction = twoClassSummary, classProbs = T,
                              savePredictions = T, returnResamp = "none")

tuneGrid.glmnet = expand.grid(alpha = 0:1, lambda = seq(0.00001, 1, length = 10))

model.caret.glmnet = train(formula_log, data = train.train, method = "glmnet",
                           trControl = control.glmnet,
                           tuneGrid = tuneGrid.glmnet)

prob.caret.glmnet = predict(model.caret.glmnet, train.test, type = "prob")[, 2]

pred.caret.glmnet = ifelse(prob.caret.glmnet > 0.1, "Yes", "No")

confusionMatrix(as.factor(pred.caret.glmnet), train.test$target)

model.caret.glmnet
plot(model.caret.glmnet)

ROC.caret.glmnet = roc(train.test$target, prob.caret.glmnet)
plot(ROC.caret.glmnet)
auc(ROC.caret.glmnet)

train.test$pred.caret.glmnet = pred.caret.glmnet
train.test$prob.caret.glmnet = prob.caret.glmnet

#######################################################################

# final decision

train.test = train.test %>%
  mutate(final.pred = apply(select(., starts_with("pred")), 1, function(x) names(table(x))[which.max(table(x))]))

train.test = train.test %>%
  mutate(final.prob = ifelse(final.pred == "No",
                             pmin(prob.caret.glmnet, prob.glm.log, prob.caret.rf),
                             pmax(prob.caret.glmnet, prob.glm.log, prob.caret.rf)))

confusionMatrix(as.factor(train.test$final.pred), train.test$target)

ROC.train.test = roc(train.test$target, train.test$final.prob)
plot(ROC.train.test)
auc(ROC.train.test)

#######################################################################

# actual test set

test.prob.glm = predict(model.glm.log10, test, type = "response")
test.prob.glmnet = predict(model.caret.glmnet, test, type = "prob")[, 2]
test.prob.rf = predict(model.caret.rf, test, type = "prob")[, 2]

submission = test %>%
  mutate(target = test.prob.rf) %>%
  select(client_id, target)

write_csv(submission, "C:/Users/user/Documents/Zindi_Artificial_Intelligence_Challenge_Beginner/submission.csv")

