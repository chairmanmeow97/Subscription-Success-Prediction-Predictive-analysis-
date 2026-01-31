# Define the list of required packages
packages <- c("tidyverse", "caret", "randomForest", "xgboost", "e1071", "pROC")

# Identify packages that are not yet installed
packages_to_install <- packages[!(packages %in% installed.packages()[, "Package"])]

# Install any packages that are missing
if (length(packages_to_install) > 0) {
  install.packages(packages_to_install)
}

# Load all the packages
lapply(packages, library, character.only = TRUE)


train_data <- read.csv("train_data.csv")
test_data <- read.csv("test_data.csv")


# Make sure our target variable is properly set up
train_data$y <- factor(train_data$y, levels = c("no", "yes"))
test_data$y <- factor(test_data$y, levels = c("no", "yes"))

# Check the balance of customers who subscribed vs those who didn't
sub_table <- table(train_data$y)
print("Customer subscription distribution:")
print(sub_table)
print(paste0("Subscription rate: ", round(prop.table(sub_table)[2] * 100, 1), "%"))


missing_count <- sapply(train_data, function(x) sum(is.na(x)))
if(sum(missing_count) > 0) {
  print("Missing values found:")
  print(missing_count[missing_count > 0])
} else {
  print("No missing values in the dataset!")
}

# Quick summary of customer demographics
print("Customer age statistics:")
summary(train_data$age)

print("Job distribution:")
sort(table(train_data$job), decreasing = TRUE)

# Simple visualization of call duration by subscription
ggplot(train_data, aes(x = duration, fill = y)) +
  geom_histogram(bins = 30, position = "dodge") +
  labs(title = "Call Duration by Subscription Status",
       x = "Call Duration (seconds)",
       y = "Number of Customers") +
  theme_minimal()

# Set up our evaluation approach
set.seed(123)  # For reproducible results
train_control <- trainControl(
  method = "cv",           # Use cross-validation
  number = 5,              # 5-fold cross-validation
  classProbs = TRUE,       # We want probability estimates
  summaryFunction = twoClassSummary,  # For ROC calculations
  savePredictions = "final"    # Save predictions for later analysis
)

# ===========================================
# RANDOM FOREST MODEL
# ===========================================

# Set up different values to try for mtry (features per split)
rf_grid <- expand.grid(mtry = c(2, 4, 6, 8))

# Track the time to see how long it takes
rf_start_time <- Sys.time()

# Train the Random Forest model
model_rf <- train(
  y ~ .,                 # Predict y using all other variables
  data = train_data,     # Use our training data
  method = "rf",         # Random Forest method
  metric = "ROC",        # Optimize for ROC
  trControl = train_control,  # Using our evaluation setup
  tuneGrid = rf_grid,    # Try these mtry values
  importance = TRUE,     # Calculate variable importance
  ntree = 50            # Build 500 trees
)

rf_end_time <- Sys.time()
rf_time <- rf_end_time - rf_start_time
print(paste("Random Forest training took", round(rf_time, 2), "seconds"))

# See the results
print("Random Forest model results:")
print(model_rf)

# Plot performance by mtry value
plot(model_rf, main = "Random Forest Performance by mtry")

# Which customer attributes matter most?
varImp_rf <- varImp(model_rf)
plot(varImp_rf, top = 15, main = "Random Forest: What Drives Subscriptions?")

# Test the model on new data
predict_rf <- predict(model_rf, test_data)  # Yes/No predictions
predict_rf_prob <- predict(model_rf, test_data, type = "prob")  # Probability predictions

# Create a confusion matrix to evaluate performance
cm_rf <- confusionMatrix(predict_rf, test_data$y, positive = "yes")
print("Random Forest Performance Summary:")
print(cm_rf)

# Calculate and visualize ROC curve
roc_rf <- roc(test_data$y, predict_rf_prob$yes)
plot(roc_rf, main = "Random Forest: Performance Curve", col = "blue")
auc_rf <- auc(roc_rf)
legend("bottomright", legend = paste("AUC =", round(auc_rf, 3)))

# ===========================================
# XGBOOST MODEL
# ===========================================

# Set up different options to test
xgb_grid <- expand.grid(
  nrounds = c(50, 100),         # Number of trees to build
  max_depth = c(3, 5),           # Maximum tree depth
  eta = c(0.01, 0.1, 0.3),       # Learning rate
  gamma = 0,                     # Minimum loss reduction 
  colsample_bytree = 0.8,        # Fraction of features to use
  min_child_weight = 1,          # Minimum sum of weights needed
  subsample = 0.8                # Fraction of data to use per tree
)

# Track the time to see how long it takes
xgb_start_time <- Sys.time()

# Train the XGBoost model
model_xgb <- train(
  y ~ .,                   # Predict y using all other variables
  data = train_data,       # Use our training data
  method = "xgbTree",      # XGBoost decision tree method
  metric = "ROC",          # Optimize for ROC
  trControl = train_control,  # Using our evaluation setup
  tuneGrid = xgb_grid,     # Try these parameter combinations
  verbose = FALSE          # Don't show all the output
)

xgb_end_time <- Sys.time()
xgb_time <- xgb_end_time - xgb_start_time
print(paste("XGBoost training took", round(xgb_time, 2), "seconds"))

# See the results
print("XGBoost model results:")
print(model_xgb)

# Plot performance by parameter values
plot(model_xgb, main = "XGBoost Performance")

# What customer characteristics matter most?
varImp_xgb <- varImp(model_xgb)
plot(varImp_xgb, top = 15, main = "XGBoost: Key Subscription Drivers")

# Test on new data
predict_xgb <- predict(model_xgb, test_data)  # Yes/No predictions
predict_xgb_prob <- predict(model_xgb, test_data, type = "prob")  # Probability predictions

# Create a confusion matrix to evaluate performance
cm_xgb <- confusionMatrix(predict_xgb, test_data$y, positive = "yes")
print("XGBoost Performance Summary:")
print(cm_xgb)

# Calculate and visualize ROC curve
roc_xgb <- roc(test_data$y, predict_xgb_prob$yes)
plot(roc_xgb, main = "XGBoost: Performance Curve", col = "red")
auc_xgb <- auc(roc_xgb)
legend("bottomright", legend = paste("AUC =", round(auc_xgb, 3)))

# ===========================================
# NAIVE BAYES MODEL
# ===========================================

# Set up different options to test

# Track the time to see how long it takes
nb_start_time <- Sys.time()

# Train the Naive Bayes model
model_nb <- train(
  y ~ .,                   # Predict y using all other variables
  data = train_data,       # Use our training data
  method = "naive_bayes",  # Naive Bayes method
  metric = "ROC",          # Optimize for ROC
  trControl = train_control  # Using our evaluation setup
)

nb_end_time <- Sys.time()
nb_time <- nb_end_time - nb_start_time
print(paste("Naive Bayes training took", round(nb_time, 2), "seconds"))

# See the results
print("Naive Bayes model results:")
print(model_nb)

# Plot performance by parameter values
plot(model_nb, main = "Naive Bayes Performance")

# Test on new data
predict_nb <- predict(model_nb, test_data)  # Yes/No predictions
predict_nb_prob <- predict(model_nb, test_data, type = "prob")  # Probability predictions

# Create a confusion matrix to evaluate performance
cm_nb <- confusionMatrix(predict_nb, test_data$y, positive = "yes")
print("Naive Bayes Performance Summary:")
print(cm_nb)

# Calculate and visualize ROC curve
roc_nb <- roc(test_data$y, predict_nb_prob$yes)
plot(roc_nb, main = "Naive Bayes: Performance Curve", col = "green")
auc_nb <- auc(roc_nb)
legend("bottomright", legend = paste("AUC =", round(auc_nb, 3)))

# ===========================================
# MODEL COMPARISON
# ===========================================

# Create a function to extract the key business metrics
get_metrics <- function(conf_matrix, roc_obj) {
  data.frame(
    Accuracy = round(conf_matrix$overall["Accuracy"] * 100, 1),  # % correct
    Success_Rate = round(conf_matrix$byClass["Sensitivity"] * 100, 1),  # % of subscribers found
    Precision = round(conf_matrix$byClass["Pos Pred Value"] * 100, 1),  # % of targeted customers who subscribe
    F1 = round(conf_matrix$byClass["F1"] * 100, 1),  # Balance between the two
    ROC_Score = round(as.numeric(auc(roc_obj)) * 100, 1),  # Overall ranking ability
    Training_Time = c(round(rf_time, 1), round(xgb_time, 1), round(nb_time, 1))[match(deparse(substitute(conf_matrix)), c("cm_rf", "cm_xgb", "cm_nb"))]
  )
}

# Get the metrics for each model
metrics_rf <- get_metrics(cm_rf, roc_rf)
metrics_xgb <- get_metrics(cm_xgb, roc_xgb)
metrics_nb <- get_metrics(cm_nb, roc_nb)

# Create an easy-to-read comparison table
comparison <- rbind(
  cbind(Model = "Random Forest", metrics_rf),
  cbind(Model = "XGBoost", metrics_xgb),
  cbind(Model = "Naive Bayes", metrics_nb)
)

# View the comparison
print("Model Comparison (all figures are percentages except time):")
print(comparison)

# Create a visual comparison of all models
par(mar = c(5, 5, 4, 2) + 0.1)  # Adjust margins for better display
plot(roc_rf, col = "blue", main = "Model Performance Comparison", 
     lwd = 2, cex.main = 1.2, cex.lab = 1.1)
plot(roc_xgb, col = "red", add = TRUE, lwd = 2)
plot(roc_nb, col = "green", add = TRUE, lwd = 2)
legend("bottomright", 
       legend = c(paste("Random Forest:", round(auc_rf, 3)),
                  paste("XGBoost:", round(auc_xgb, 3)),
                  paste("Naive Bayes:", round(auc_nb, 3))),
       col = c("blue", "red", "green"),
       lwd = 2)


