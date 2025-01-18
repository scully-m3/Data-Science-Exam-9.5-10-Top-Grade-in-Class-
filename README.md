# Data-Science-Exam-9.5-10-Top-Grade-in-Class-
library(data.table)
library(glmnet)
library(caret)
library(gridExtra)
library(grid)
library(tree)
library(randomForest)
library(tensorflow)
library(dplyr)
library(keras)
library(tfruns)
library(purrr)

#Clear the console
cat("\014")

#Clear all variables
rm(list = ls())

#Set the environment to the current folder
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#load data
Data <- readRDS("Data_DS_Exam1_2024.RDS")
Data_2 <- readRDS("Data_DS_Exam1_2024_pre_processed.RDS")

##Question 1
##1.1
Data_clean <- Data[, colMeans(is.na(Data)) <= 0.15]
cat("Number of features removed:", ncol(Data) - ncol(Data_clean), "\n")
cat("Number of features:", ncol(Data_clean) -1, "\n")

##1.2
missing <- colSums(is.na(Data_clean))
missing <- missing[missing > 0]

cat("Feature with missing values:\n")
print(missing)

median_otherpercap <- median(Data$OtherPerCap, na.rm = TRUE)
print(median_otherpercap)

#Replace with median
Data$OtherPerCap[is.na(Data$OtherPerCap)] <- median_otherpercap

#New mean of OtherPerCap
new_mean <- mean(Data$OtherPerCap)
print(new_mean)

##1.3
#Initial Y observation
initial_Y_observations <- nrow(Data)
print(initial_Y_observations)

#Remove missing Y variables
Data_clean <- Data_clean[!is.na(Data_clean$ViolentCrimesPerPop), ]

#No. observations after removal
final_observations <- nrow(Data_clean)
print(final_observations)

#Calculate the mean of Y
mean_Y <- mean(Data_clean$ViolentCrimesPerPop)
print(mean_Y)

print(initial_Y_observations - final_observations)

##1.4
#Histogram
hist(Data_clean$ViolentCrimesPerPop,
     main = "Histogram of Original Total Number of Violent Crimes per 100,000 People",
     xlab = "Total Number of Violent Crimes per 100,000 People",
     ylab = "Frequency",
     col = "Pink", breaks = 30)

#Log transformation
Data_clean$ViolentCrimesPerPop <- log(Data_clean$ViolentCrimesPerPop + 1)

#Mean of LogY
mean_log_violentcrimes <- mean(Data_clean$ViolentCrimesPerPop)
print(mean_log_violentcrimes)

#Histogram Log
hist(Data_clean$ViolentCrimesPerPop,
     main = "Histogram of Log-Transformed Total Number of Violent Crimes per 100,000 People",
     xlab = "Log Transformed Total Number of Violent Crimes per 100,000 People",
     ylab = "Frequency",
     col = "Gold", breaks = 30)

##1.5
Data_clean <- Data_clean[, !names(Data_clean) %in% "communityName"]


##Question 2
##2.1
model <- lm(ViolentCrimesPerPop ~ ., data = Data_2)

#Linear Model
model_summary <- summary(model)

population_beta <- coef(model_summary)["population", "Estimate"]
population_tstat <- coef(model_summary)["population", "t value"]

print(population_beta)
print(population_tstat)

##2.2
predictions <- predict(model, newdata = Data_2)

#MSPE
MSPE <- mean((Data_2$ViolentCrimesPerPop - predictions)^2)
print(MSPE)

##2.3
set.seed(1)
X <- as.matrix(Data_2[, !colnames(Data_2) %in% "ViolentCrimesPerPop"])
Y <- Data_2$ViolentCrimesPerPop

lambda_grid <- seq(0, 0.03, length.out = 1000)

#10 fold cross validation
lasso_cv <- cv.glmnet(X, Y, alpha = 1, lambda = lambda_grid, nfolds = 10)
optimal_lambda <- lasso_cv$lambda.min

#Optimal lambda coefficient
coefficients <- coef(lasso_cv, s = "lambda.min")

#No. non-zero coefficients
num_features <- sum(coefficients != 0) - 1

#Plot CV error vs log(lambda)
plot(log(lasso_cv$lambda), lasso_cv$cvm, type = "l", lwd = 2,
     xlab = "log(Lambda)", ylab = "Mean Squared Error",
     main = "Lasso Cross-Validation Error")
abline(v = log(optimal_lambda), col = "red", lwd = 2, lty = 2)

#Results
print(optimal_lambda)
print(num_features)

##2.4
#Linear Model
train_preds_linear <- predict(model, newdata = Data_2)
train_mse_linear <- mean((Data_2$ViolentCrimesPerPop - train_preds_linear)^2)
print(train_mse_linear)

#10-fold Cross-Validation for the linear model
set.seed(1)
cv_linear <- caret::train(ViolentCrimesPerPop ~ ., data = Data_2, method = "lm", trControl = trainControl(method = "cv", number = 10))
cv_mse_linear <- (cv_linear$results$RMSE)^2
print(cv_mse_linear)

#Lasso Model
#Preparing X and Y
X_train <- as.matrix(Data_2[, !colnames(Data_2) %in% "ViolentCrimesPerPop"])
y_train <- Data_2$ViolentCrimesPerPop

#Lasso Model Predictions and Training MSE
train_preds_lasso <- predict(lasso_cv$glmnet.fit, newx = X_train, s = lasso_cv$lambda.min)
train_mse_lasso <- mean((y_train - train_preds_lasso)^2)
print(train_mse_lasso)

#Cross-Validation MSE for Lasso Model
cv_mse_lasso <- min(lasso_cv$cvm)
print(cv_mse_lasso)

results_table <- data.frame(
  Model = c("Linear Model", "Lasso Model"),
  Training_MSE = c(round(train_mse_linear, 7), round(train_mse_lasso, 7)),
  CV_MSE = c(round(cv_mse_linear, 7), round(cv_mse_lasso, 7)))

grid.newpage()
grid.table(results_table)

##Question 3
##3.1
#Regression tree to predict ViolentCrimesPerPop
tree_model <- tree(ViolentCrimesPerPop ~ ., data = Data_2)

#Summary of tree model
summary_tree <- summary(tree_model)

#Terminal nodes
num_terminal_nodes <- summary_tree$size
print(num_terminal_nodes)

#Features used
features_used <- summary_tree$used
print(features_used)


#MSE for the tree
train_preds_tree <- predict(tree_model, newdata = Data_2)
train_mse_tree <- mean((Data_2$ViolentCrimesPerPop - train_preds_tree)^2)
print(train_mse_tree)

##3.2
#Regression Tree Plot
plot(tree_model)
text(tree_model, pretty = 0, cex = 0.8)
title("Total Number of Violent Crimes per 100,000 People Regression Tree")

##3.3
#Seed
set.seed(1)

#10 fold cross-validation optimal tree size
cv_tree <- cv.tree(tree_model, FUN = prune.tree, K = 10)

#Plot MSE & tree size
plot(cv_tree$size, cv_tree$dev, type = "b",
     xlab = "Tree Size (Number of Terminal Nodes)",
     ylab = "Cross-Validation Error",
     main = "Cross-Validation Error vs Tree Size")
    grid()

#Optimal tree size
optimal_size <- cv_tree$size[which.min(cv_tree$dev)]
print(optimal_size)

#Prune tree
pruned_tree <- prune.tree(tree_model, best = optimal_size)

#Summary pruned tree
summary_pruned <- summary(pruned_tree)

#No. terminal nodes in the pruned tree
num_terminal_nodes_pruned <- summary_pruned$size
print(num_terminal_nodes_pruned)

#Plot pruned tree
plot(pruned_tree)
text(pruned_tree, pretty = 0, cex = 0.8)
title("Pruned Regression Tree")

##3.4
#MSE for the Pruned Tree
train_preds_pruned <- predict(pruned_tree, newdata = Data_2)
train_mse_pruned <- mean((Data_2$ViolentCrimesPerPop - train_preds_pruned)^2)
print(train_mse_pruned)

cv_mse_pruned <- min(cv_tree$dev) / nrow(Data_2)
print(cv_mse_pruned)

results_table_2 <- data.frame(
  Model = c("Linear Model", "Lasso Model", "Pruned Tree"),
  Training_MSE = c(round(train_mse_linear, 7), round(train_mse_lasso, 7), round(train_mse_pruned, 7)),
  CV_MSE = c(round(cv_mse_linear, 7), round(cv_mse_lasso, 7), round(cv_mse_pruned, 7)))

grid.newpage()
grid.table(results_table_2)

##3.5
#Seed
set.seed(1)

#Bagging model with 250 trees
bagging_model <- randomForest(ViolentCrimesPerPop ~ ., data = Data_2, 
                              mtry = ncol(Data_2) - 1, ntree = 250, importance = TRUE)

#OOB MSE each tree
oob_error <- bagging_model$mse

#Minimum OOB MSE and number of trees
min_oob_mse <- min(oob_error)
optimal_trees <- which.min(oob_error)
cat("Minimum OOB MSE:", round(min_oob_mse, 7), "\n")
cat("Optimal Number of Trees:", optimal_trees, "\n")

#Plot the OOB MSE vs Number of Trees
plot(1:250, oob_error, type = "l", lwd = 2,
     xlab = "Number of Trees", 
     ylab = "OOB MSE", 
     main = "OOB MSE vs Number of Trees for Bagging")
points(optimal_trees, min_oob_mse, col = "red", pch = 19)
grid()

##3.6
#Training MSE
train_preds_bagging <- predict(bagging_model, newdata = Data_2)
train_mse_bagging <- mean((Data_2$ViolentCrimesPerPop - train_preds_bagging)^2)
print(train_mse_bagging)

#Minimum OOB Estimate of MSE
print(min_oob_mse)

#Table
results_table <- data.frame(
  Model = c("Linear Model", "Lasso Model", "Pruned Tree", "Bagging"),
  Training_MSE = c(round(train_mse_linear, 7), round(train_mse_lasso, 7), 
                   round(train_mse_pruned, 7), round(train_mse_bagging, 7)),
  CV_or_OOB_MSE = c(round(cv_mse_linear, 7), round(cv_mse_lasso, 7), 
                    round(cv_mse_pruned, 7), round(min_oob_mse, 7)))
grid.newpage()
grid.table(results_table)

##3.7
#Seed
set.seed(1)

#Loop Prep
m_values <- 1:30
oob_mse_values <- numeric(length(m_values))
train_mse_values <- numeric(length(m_values))

#Loop
for (m in m_values) {
  cat("Fitting Random Forest with mtry =", m, "\n")
  rf_model <- randomForest(ViolentCrimesPerPop ~ ., data = Data_2, 
                           mtry = m, ntree = 250, importance = FALSE)
  oob_mse_values[m] <- min(rf_model$mse)
  train_preds <- predict(rf_model, newdata = Data_2)
  train_mse_values[m] <- mean((Data_2$ViolentCrimesPerPop - train_preds)^2)
}

#Optimal mtry (minimum OOB MSE)
optimal_mtry <- which.min(oob_mse_values)
optimal_oob_mse <- min(oob_mse_values)
optimal_train_mse_rf <- train_mse_values[optimal_mtry]

#Results
print(optimal_mtry)
print(optimal_oob_mse)
print(optimal_train_mse_rf)

#Plot OOB MSE vs mtry
plot(m_values, oob_mse_values, type = "b", lwd = 2, pch = 19,
     xlab = "Number of Features Used at Each Split (mtry)",
     ylab = "OOB MSE",
     main = "OOB MSE vs mtry for Random Forest")
points(optimal_mtry, optimal_oob_mse, col = "red", pch = 19, cex = 1.5)
grid()

#Results
results_table <- data.frame(Model = c("Linear Model", "Lasso Model", "Pruned Tree", "Bagging", "Random Forest"),
  Training_MSE = c(round(train_mse_linear, 7), round(train_mse_lasso, 7), 
                   round(train_mse_pruned, 7), round(train_mse_bagging, 7), round(optimal_train_mse_rf, 7)),
  CV_or_OOB_MSE = c(round(cv_mse_linear, 7), round(cv_mse_lasso, 7), 
                    round(cv_mse_pruned, 7), round(min_oob_mse, 7), round(optimal_oob_mse, 7)))

#Results
grid.newpage()
grid.table(results_table)

##Question 4
tensorflow::tf$random$set_seed(1)
set_random_seed(1, disable_gpu = TRUE)

##4.1
#Normalize the Data
normalized_data <- as.data.frame(scale(Data_2))

#Split Data
train_data <- normalized_data[1:1500, ]
test_data <- normalized_data[1501:nrow(normalized_data), ]

#Mean of population in Training and Test Data
mean_population_train <- mean(train_data$population)
mean_population_test <- mean(test_data$population)

print(mean_population_train)
print(mean_population_test)

##4.2
tensorflow::tf$random$set_seed(1)
set_random_seed(1, disable_gpu = TRUE)

#Exclude ViolentCrimesPerPop from normalization
features <- setdiff(names(Data_2), "ViolentCrimesPerPop")

#Normalise
Data_2[features] <- scale(Data_2[features])

#Split the data
train_data <- Data_2[1:1500, ]
test_data <- Data_2[1501:nrow(Data_2), ]

#Matrix for the neural network
x_train <- as.matrix(train_data[, features])
y_train <- train_data$ViolentCrimesPerPop

x_valid <- as.matrix(test_data[, features])
y_valid <- test_data$ViolentCrimesPerPop

#Define and Compile the Neural Network
model <- keras_model_sequential() %>%
  layer_dense(units = 4, activation = "relu", input_shape = ncol(x_train)) %>%
  layer_dense(units = 1, activation = "linear")

model %>% compile(optimizer = optimizer_rmsprop(), loss = "mse", metrics = "mse")

#Early Stop to Prevent Overfitting
early_stopping <- callback_early_stopping(
  monitor = "val_loss", patience = 2, restore_best_weights = TRUE)

#Train Neural Network
history <- model %>% fit(x = x_train, y = y_train, validation_data = list(x_valid, y_valid), 
                         epochs = 100, batch_size = 32, callbacks = list(early_stopping), verbose = 1)

#Plot Training & Validation MSE
plot_data <- data.frame(epoch = seq_along(history$metrics$loss), training_mse = history$metrics$loss, validation_mse = history$metrics$val_loss)

ggplot(plot_data, aes(x = epoch)) +
  geom_line(aes(y = training_mse, color = "Training MSE")) +
  geom_line(aes(y = validation_mse, color = "Validation MSE")) +
  labs(title = "Neural Network Training Progress",
       x = "Epoch", y = "Mean Squared Error (MSE)") +
  scale_color_manual(values = c("Training MSE" = "blue", "Validation MSE" = "red")) +
  theme_minimal()

#MSE for Last 5 Epochs
tail(history$metrics$loss, 7)
tail(history$metrics$val_loss, 7)

#Best Validation MSE
print(min(history$metrics$val_loss))


##4.3
tensorflow::tf$random$set_seed(1)
set_random_seed(1, disable_gpu = TRUE)

#Multi-Layer Perceptron
model_mlp <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu", input_shape = ncol(x_train)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

model_mlp %>% compile(optimizer = optimizer_adam(), loss = "mse", metrics = "mse")

history_mlp <- model_mlp %>% 
  fit(x_train, y_train, validation_data = list(x_valid, y_valid),
  epochs = 100, batch_size = 32, callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 3)), verbose = 0)

best_mlp_mse <- min(history_mlp$metrics$val_loss)
train_mlp_mse <- min(history_mlp$metrics$mse)

#Hyperparameter Tuning (Grid Search)
run_mlp_tuning <- function(units1, dropout1, learning_rate) {
  model <- keras_model_sequential() %>%
  layer_dense(units = units1, activation = "relu", input_shape = ncol(x_train)) %>%
  layer_dropout(rate = dropout1) %>%
  layer_dense(units = 1, activation = "linear")
  
model %>% compile(optimizer = optimizer_adam(learning_rate = learning_rate), loss = "mse", metrics = "mse")
  
history <- model %>% fit(
x_train, y_train, validation_data = list(x_valid, y_valid),
epochs = 50, batch_size = 32, verbose = 0,
callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 3)))
return(c(min(history$metrics$val_loss), min(history$metrics$mse)))
}

hyper_grid <- expand.grid(units1 = c(16, 32, 64), dropout1 = c(0.2, 0.3), learning_rate = c(0.001, 0.01))
hyper_results <- pmap(hyper_grid, function(units1, dropout1, learning_rate) {
  run_mlp_tuning(units1, dropout1, learning_rate)})

hyper_grid$val_loss <- map_dbl(hyper_results, 1)
hyper_grid$train_loss <- map_dbl(hyper_results, 2)

best_params <- hyper_grid[which.min(hyper_grid$val_loss), ]
best_tuned_train_mse <- best_params$train_loss

# K-Fold Cross-Validation
k <- 5
folds <- createFolds(y_train, k = k, list = TRUE)
cv_mse <- c()
cv_train_mse <- c()

for (i in 1:k) {
valid_idx <- folds[[i]]
x_valid_fold <- x_train[valid_idx, ]
y_valid_fold <- y_train[valid_idx]
x_train_fold <- x_train[-valid_idx, ]
y_train_fold <- y_train[-valid_idx]
  
model_cv <- keras_model_sequential() %>%
layer_dense(units = 32, activation = "relu", input_shape = ncol(x_train)) %>%
layer_dropout(rate = 0.3) %>%
layer_dense(units = 1, activation = "linear")
  
model_cv %>% compile(optimizer = optimizer_adam(), loss = "mse", metrics = "mse")
  
history_cv <- model_cv %>% fit(
x_train_fold, y_train_fold, validation_data = list(x_valid_fold, y_valid_fold),
epochs = 50, batch_size = 32, verbose = 0,
callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 3)))
  
fold_mse <- min(history_cv$metrics$val_loss)
fold_train_mse <- min(history_cv$metrics$mse)
  
cv_mse <- c(cv_mse, fold_mse)
cv_train_mse <- c(cv_train_mse, fold_train_mse)
}

avg_cv_mse <- mean(cv_mse)
avg_cv_train_mse <- mean(cv_train_mse)

#Results
results_table <- data.frame(
Model = c("MLP", "Tuned MLP", "Cross-Validated NN"),
Validation_MSE = c(round(best_mlp_mse, 7), round(min(hyper_grid$val_loss), 7), round(avg_cv_mse, 7)),
Train_MSE = c(round(train_mlp_mse, 7), round(best_tuned_train_mse, 7), round(avg_cv_train_mse, 7)))

grid.newpage()
grid.table(results_table)
