
# PART 1...
################################################################################
# SET-UP
################################################################################

# Clear environment to avoid spillovers
rm(list = ls())

# Install libraries (if necessary) to access their commands
packages <- c(
  "tidyverse",
  "caret",
  "glmnet",
  "randomForest",
  "gbm",
  "tree",
  "corrplot",
  "class",
  "MASS",
  "e1071",
  "nnet",
  "NeuralNetTools",
  "boot",
  "ggplot2"
)

installed <- packages %in% rownames(installed.packages())
if (any(!installed)) {
  install.packages(packages[!installed])
}

lapply(packages, library, character.only = TRUE)

# Load these libraries
library(tidyverse)
library(caret)
library(glmnet)
library(randomForest)
library(gbm)
library(tree)
library(corrplot)
library(class)
library(MASS)
library(e1071)
library(nnet)
library(NeuralNetTools)
library(boot)
library(ggplot2)

# Please toggle between seeds 1-5 to reproduce coursework submission results for Part 1
seed <- 1

################################################################################
# CODE BLOCK 1: DATA INSPECTION / PREPARATION FOR PART 1
################################################################################

# Load group 34 data, rename and view
load("data.sim.34.RData")
df <- df_sample
head(df)

# Confirm there are no missing observations (there are none)
sum(is.na(df))

# Examine data for initial understanding and data types
dim(df)
names(df)
summary(df)
str(df)

# Produce a correlation matrix to check for multi-collinearity
cor_matrix <- cor(df)
print(round(cor_matrix, 2))

# Convert discrete variables to factors based how the data looks
df$X9 <- as.factor(df$X9)
df$X10 <- as.factor(df$X10)
df$X11 <- as.factor(df$X11)

# Create training sample
set.seed(seed)
test.prop <- 0.5
train.size <- nrow(df) * (1 - test.prop)
train.sample <- sample(1:nrow(df), train.size)

# Define explanatory variables and dependent variable in training sample
X.train <- df[train.sample, paste0("X", 1:11)]
Y.train <- df[train.sample, "Y"]

# Center and scale the training predictors to ensure fair comparison between OLS and Shrinkage methods
pp.pars <- preProcess(X.train, method = c("center", "scale"))
X.train.scaled <- predict(pp.pars, X.train)

# Form final training data frame
train.data <- data.frame(Y.train, X.train.scaled)

# Define explanatory variables and dependent variable in test sample
X.test <- df[-train.sample, paste0("X", 1:11)]
Y.test <- df[-train.sample, "Y"]

# Scale the explanatory variable in the test data using the training sample characteristics
X.test.scaled  <- predict(pp.pars, X.test)

# Form final test data frame
test.data  <- data.frame(Y.test, X.test.scaled)

################################################################################
# CODE BLOCK 2: OLS REGRESSION
################################################################################

# Estimate regression of Y on X1-X11 using training data
reg.df <- lm(Y.train ~ .,data = train.data)
summary(reg.df)

# Calculate predicted values and mse to evaluate performance on training data
Yhat.train <- predict(reg.df, newdata = train.data)
mse.train<-mean((Y.train - Yhat.train) ^ 2)

# Calculate predicted values and mse to evaluate performance on test data
Yhat.test <- predict(reg.df, newdata = test.data)
mse.test <- mean((Y.test - Yhat.test) ^ 2)

# Compare train and test mse to evaluate overfitting
cat("train mse:", mse.train, "\n")
cat("test mse:", mse.test, "\n")

################################################################################
# CODE BLOCK 3: RIDGE
################################################################################

# Ensure explanatory variables are in matrix format as required for 'glmnet'
X.train.mat <- model.matrix(~ . - 1, data = X.train.scaled) # This is for training ridge & lasso
X.test.mat <- model.matrix(~ . - 1, data = X.test.scaled)   # This is for testing ridge & lasso

# Form a grid of values for the tuning parameter to see how coefficients vary with this
set.seed(seed)
grid.ridge <- 10^seq(10, -2, length = 100)

# Estimate 100 ridge models using grid of values
ridge.mod <- glmnet(X.train.mat, Y.train, alpha = 0, lambda = grid.ridge)

# Plot how coefficients vary with lambda
plot(ridge.mod, xvar = "lambda", sign.lambda = 1)

# Use 10-fold cross-validation to find optimal value of Lambda
set.seed(seed)
cv.ridge <- cv.glmnet(X.train.mat, Y.train, alpha = 0)

# Estimate ridge model with best value of lambda
bestlam.ridge <- cv.ridge$lambda.min
ridge.mod <- glmnet(X.train.mat, Y.train, alpha = 0, lambda = bestlam.ridge)

# Display optimal value of lambda and log(lambda) to understand degree of shrinkage
cat("optimal value of lambda for ridge:", bestlam.ridge, "\n")

# Show how MSE varies with lambda to visualise relationship between lambda and cross-validated error
plot(cv.ridge, sign.lambda = 1)

# Calculate and display MSE on training data for optimal value of lambda
# This is to evaluate how well it fits training data
ridge.pred.train <- predict(ridge.mod, s = bestlam.ridge, newx = X.train.mat)
mse.ridge.train <- mean((ridge.pred.train - Y.train) ^ 2)
cat("train mse for ridge:", mse.ridge.train, "\n")

# Calculate and display MSE on the test data for optimal value of lambda
# This is to evaluate how well it performs on unseen data
ridge.pred.test <- predict(ridge.mod, s = bestlam.ridge, newx = X.test.mat)
mse.ridge.test <- mean((ridge.pred.test - Y.test) ^ 2)
cat("test mse for ridge:", mse.ridge.test, "\n")

################################################################################
# CODE BLOCK 4: OLS VS. RIDGE
################################################################################

# We use the whole data set to produce final models for comparison...
# First scale all explanatory variables
pp.pars.full <- preProcess(df[, paste0("X", 1:11)], method = c("center", "scale"))
X.full.scaled <- predict(pp.pars.full, df[, paste0("X", 1:11)])

# Ensure explanatory variables are in matrix format for 'glmnet'
X <- model.matrix(~ . - 1, data = X.full.scaled)
Y <- df$Y

# Estimate a model using the whole data set to find ridge coefficients
ridge.whole <- glmnet(X, Y, alpha = 0)
ridge.coefficients <- predict(ridge.whole, type = "coefficients", s = bestlam.ridge)

# Estimate OLS model on the whole data set for coefficient comparison
full.data <- data.frame(Y = Y, X.full.scaled)
ols.whole <- lm(Y ~ ., data = full.data)
ols.coefficients <- coef(ols.whole)

# Compare coefficients
print("OLS coefficients:")
print(ols.coefficients)
print("Ridge coefficients:")
print(ridge.coefficients)

# Create a comparison table for train and test MSE to evaluate over fitting
mse_comparison_table <- data.frame(
  Model = c("OLS", "Ridge"),
  Train_MSE = c(mse.train, mse.ridge.train),
  Test_MSE = c(mse.test, mse.ridge.test)
) %>%
  mutate(Difference = Test_MSE - Train_MSE) %>%
  mutate(across(where(is.numeric), ~ round(., 2)))
print(mse_comparison_table)

################################################################################
# CODE BLOCK 5: LASSO
################################################################################

# Form a grid of values for the tuning parameter to see how coefficients vary with this
grid.lasso <- 10 ^ seq(3, -2, length = 100)

# Estimate 100 Lasso models using grid of values
lasso.mod <- glmnet(X.train.mat, Y.train, alpha = 1, lambda = grid.lasso)

# Plot how coefficients vary with lambda
plot(lasso.mod, "lambda", sign.lambda = 1)

# Use 10-fold cross-validation to find optimal value of lambda
set.seed(seed)
cv.lasso <- cv.glmnet(X.train.mat, Y.train, alpha = 1)
bestlam.lasso <- cv.lasso$lambda.min
lasso.mod <- glmnet(X.train.mat, Y.train, alpha = 1, lambda = bestlam.lasso)

# Display optimal value of lambda to understand degree of shrinkage
cat("Optimal value of lambda for lasso:", bestlam.lasso, "\n")

# Understand how cross-validated error varies with lambda
plot(cv.lasso, sign.lambda = 1)

# Calculate and display MSE on training data for optimal value of lambda
# This is to evaluate how well it fits training data
lasso.pred.train <- predict(lasso.mod, s = bestlam.lasso, newx = X.train.mat)
mse.lasso.train <- mean((lasso.pred.train - Y.train) ^ 2)
cat("train mse for lasso:", mse.lasso.train, "\n")

# Calculate and display MSE on the test data for optimal value of lambda
# This is to evaluate how well it performs on unseen data
lasso.pred.test <- predict(lasso.mod, s = bestlam.lasso, newx = X.test.mat)
mse.lasso.test <- mean((lasso.pred.test - Y.test) ^ 2)
cat("test mse for lasso:", mse.lasso.test, "\n")

################################################################################
# CODE BLOCK 6: OLS VS. RIDGE VS. LASSO
################################################################################

# Estimate a model using the whole data set to find lasso coefficients for comparison
lasso.whole <- glmnet(X, Y, alpha = 1)
lasso.coefficients <- predict(lasso.whole, type = "coefficients", s = bestlam.lasso)

# Compare coefficients across all models
print("OLS coefficients:")
print(ols.coefficients)

print("Ridge coefficients:")
print(ridge.coefficients)

print("Lasso coefficients:")
print(lasso.coefficients)

# Create a comparison table for train and test MSE to evaluate over fitting
mse_comparison_table <- data.frame(
  Model = c("OLS", "Ridge", "Lasso"),
  Train_MSE = c(mse.train, mse.ridge.train, mse.lasso.train),
  Test_MSE = c(mse.test, mse.ridge.test, mse.lasso.test)
) %>%
  mutate(Difference = Test_MSE - Train_MSE) %>%
  mutate(across(where(is.numeric), ~ round(., 2)))
print(mse_comparison_table)

################################################################################
# CODE BLOCK 7: TREE MODEL
################################################################################

# Estimate deep tree using training data and summarise
set.seed(seed)
tree.df <- tree(Y.train ~ ., data = train.data, control = tree.control(nobs = nrow(train.data), mincut = 10))
summary(tree.df)

# Plot tree to visualise how sample space is partitioned
plot(tree.df)
text(tree.df, pretty = 0, cex = 0.6)

# Calculate training MSE for reference
Yhat.train.tree <- predict(tree.df, newdata = train.data)
mse.train.tree <- mean((Y.train - Yhat.train.tree)^2)
cat("Deep Tree MSE on training sample:", mse.train.tree, "\n")

# Calculate test MSE for evaluation
Yhat.test.tree <- predict(tree.df, newdata = test.data)
mse.test.tree <- mean((Y.test - Yhat.test.tree)^2)
cat("Deep Tree MSE on test sample:", mse.test.tree, "\n")

# Use cross‑validation to form pruned trees
set.seed(seed)
cv.df <- cv.tree(tree.df)

# Visualise trade-off between accuracy and complexity
# This is to see which pruned tree to use
plot(cv.df$size, cv.df$dev, type = "b",
     xlab = "Tree Size",
     ylab = "Cross‑Validated Deviance",
     main = "Cross‑Validation for Tree Size")

# Determine optimal tree size
# This is where the trade-off between accuracy and complexity is minimised
best.size <- cv.df$size[which.min(cv.df$dev)]
cat("Optimal tree size:", best.size, "\n")

# Prune the tree to this optimal size
pruned.df <- prune.tree(tree.df, best = best.size)
summary(pruned.df)

# Plot if not a stub, then calculate train and test MSE for comparison
if (best.size > 1) {
  plot(pruned.df)
  text(pruned.df, pretty = 0, cex = 0.6)
  # Training MSE for pruned tree
  Yhat.train.pruned <- predict(pruned.df, newdata = train.data)
  # Test MSE for pruned tree
  Yhat.test.pruned <- predict(pruned.df, newdata = test.data)
} else {
  cat("Tree pruned to a single node - nothing to plot. \n ")
  # Training MSE for pruned tree
  Yhat.train.pruned <- rep(mean(Y.train), length(Y.train))
  # Test MSE for pruned tree
  Yhat.test.pruned <- rep(mean(Y.train), length(Y.test))
}

mse.train.pruned <- mean((Yhat.train.pruned - Y.train)^2)
cat("Pruned Tree MSE on train sample:", mse.train.pruned, "\n")
mse.test.pruned <- mean((Yhat.test.pruned - Y.test)^2)
cat("Pruned Tree MSE on test sample:", mse.test.pruned, "\n")

################################################################################
# CODE BLOCK 8: OLS VS. TREE MODELS
################################################################################

# Compare MSE for OLS vs. tree to understand model performance
tree_comparison_table <- data.frame(
  Model = c("OLS", "Deep Tree", "Pruned Tree"),
  Train_MSE = c(mse.train, mse.train.tree, mse.train.pruned),
  Test_MSE = c(mse.test, mse.test.tree, mse.test.pruned)
) %>%
  mutate(Difference = Test_MSE - Train_MSE) %>%
  mutate(across(where(is.numeric), ~ round(., 2)))
print(tree_comparison_table)

################################################################################
# CODE BLOCK 9: BAGGING
################################################################################

# Estimate bagging model
set.seed(seed)
bag.df <- randomForest(Y.train ~ .,data = train.data, mtry = 11, ntree = 500, importance = TRUE, keep.inbag = TRUE)
bag.df

# Calculate training MSE for reference
yhat.bag.train <- predict(bag.df, newdata = train.data)
mse.bag.train <- mean((yhat.bag.train - Y.train)^2)
cat("Bagging MSE on training sample:", mse.bag.train, "\n")

# Calculate test MSE for predictive accuracy
yhat.bag.test <- predict(bag.df, newdata = test.data)
mse.bag.test <- mean((yhat.bag.test - Y.test)^2)
cat("Bagging MSE on test sample:", mse.bag.test, "\n")

# Plot variable importance
varImpPlot(bag.df)

# Plot relationship between OOB MSE and trees
plot(bag.df)

################################################################################
# CODE BLOCK 10: RANDOM FOREST
################################################################################

# Estimate random forest model
set.seed(seed)
rf.df <- randomForest(Y.train ~ .,data=train.data, mtry = 4, ntree = 500,
                      importance = TRUE)
# mtry = p/3 is the standard
# 11/3 = 3.667, so we chose 4
rf.df

# Calculate training MSE for reference
yhat.rf.train <- predict(rf.df, newdata = train.data)
mse.rf.train <- mean((yhat.rf.train - Y.train)^2)
cat("Random Forest MSE on training sample:", mse.rf.train, "\n")

# Calculate test MSE for predictive accuracy
yhat.rf.test <- predict(rf.df, newdata = test.data)
mse.rf.test <- mean((yhat.rf.test - Y.test)^2)
cat("Random Forest MSE on test sample:", mse.rf.test, "\n")

# Plot variable importance
varImpPlot(rf.df)

# Plot relationship between OOB MSE and trees
plot(rf.df)

################################################################################
# CODE BLOCK 11: GRADIENT BOOSTING
################################################################################

# Estimate gradient boosting model
set.seed(seed)
gb.df <- gbm(Y.train ~ .,data=train.data,
             distribution = "gaussian",
             n.trees = 200,
             interaction.depth = 4,
             shrinkage = 0.1,
             cv.folds = 5,
             verbose = FALSE)

# Plot prediction error against number of trees
# Also, extract the number of tress with lowest cross-validated error
best.iter <- gbm.perf(gb.df, method = "cv", plot.it = TRUE)

# Calculate training MSE for reference
yhat.gb.train <- predict(gb.df, newdata = train.data, n.trees = best.iter)
mse.gb.train <- mean((yhat.gb.train - Y.train)^2)
cat("Boosting MSE on training sample:", mse.gb.train, "\n")

# Calculate test MSE to evaluate predictive accuracy
yhat.gb.test <- predict(gb.df, newdata = test.data, n.trees = best.iter)
mse.gb.test <- mean((yhat.gb.test - Y.test)^2)
cat("Boosting MSE on test sample:", mse.gb.test, "\n")

################################################################################
# CODE BLOCK 12: OLS VS. TREE VS. ENSEMBLE
################################################################################

# Form MSE comparison table to compare each method
ensemble_comparison_table <- data.frame(
  Model = c("OLS", "Deep Tree", "Pruned Tree",
            "Bagging", "Random Forest", "Gradient Boosting"),
  Train_MSE = c(mse.train, mse.train.tree, mse.train.pruned,
                mse.bag.train, mse.rf.train, mse.gb.train),
  Test_MSE = c(mse.test, mse.test.tree, mse.test.pruned,
               mse.bag.test, mse.rf.test, mse.gb.test)
) %>%
  mutate(Difference = Test_MSE - Train_MSE) %>%
  mutate(across(where(is.numeric), ~ round(., 2)))

print(ensemble_comparison_table)

# PART 2...
################################################################################
# SET-UP
################################################################################

# Clear environment to prevent spillover from Part 1
rm(list = ls())

# Please toggle between seeds 1-5 to reproduce coursework submission results
seed <- 1

################################################################################
# CODE BLOCK 13: DATA INSPECTION AND PREPARATION
################################################################################

# Load group 34 data
load("data.34.RData")
ds <- data_syn

# Remove unneeded data frame
rm(data_syn)

# Check for missing values (there are none)
sum(is.na(ds))

# Transform education_num as instructed
ds$ed_level <- as.numeric(
  ifelse(ds$education_num > 9.5, 3,                                                          
    ifelse(ds$education_num > 8.5, 2,                                                        
      1)))

# Check frequencies to check for no unusual values
table(ds$ed_level)

# Drop education_num to keep data set clean
ds <- ds %>%
  dplyr::select(-education_num)

# Transform marital status as instructed
ds <- ds %>%
  # Create the new 'mar_stat' column based on the existing 'marital_status'
  mutate(
    mar_stat = case_when(
      # NEVER MARRIED
      marital_status %in% c("Never-married") ~ 3,
      # MARRIED
      marital_status %in% c("Married-civ-spouse") ~ 2,
      # WAS MARRIED
      marital_status %in% c("Widowed",
                            "Divorced", 
                            "Separated",
                            "Married-spouse-absent") ~ 1,
      # HANDLE MISSING/UNKNOWN VALUES
      marital_status %in% c("?", "Unknown", "") ~ -99,
      # Catch-all for anything missed as good practice
      TRUE ~ -98))

# Check frequencies to check for no unusual values
table(ds$mar_stat)

# Remove observations where the marital status is unknown (-99 or -98)
ds <- ds %>%
  filter(mar_stat != -99) %>%
  filter(mar_stat != -98)

# Check frequencies to check for no unusual values
table(ds$mar_stat)

# Drop marital_status to keep data set clean
ds <- ds %>%
  dplyr::select(-marital_status)

# Transform occupation as instructed
ds <- ds %>%
  # Create the new 'skill_level' column based on the existing 'occupation'
  mutate(
    skill_level = case_when(
      # HIGH SKILL JOBS
      occupation %in% c("Exec-managerial", "Prof-specialty", "Armed-Forces") ~ 3,
      # MEDIUM SKILL JOBS
      occupation %in% c("Adm-clerical", "Sales", "Tech-support", 
                        "Protective-serv", "Transport-moving", "Craft-repair") ~ 2,
      # LOW SKILL JOBS (and other non-professional)
      occupation %in% c("Other-service", "Machine-op-inspct", 
                        "Handlers-cleaners", "Farming-fishing", "Priv-house-serv") ~ 1,
      # HANDLE MISSING/UNKNOWN VALUES
      occupation %in% c("?", "Unknown", "") ~ -99,
      # Catch-all for anything missed (optional, but good practice)
      TRUE ~ -98))

# Check frequencies to ensure no unusual values
table(ds$skill_level)

# Remove observations where the skill is unknown (-99 or -98)
ds <- ds %>%
  filter(skill_level != -99) %>%
  filter(skill_level != -98)

# Check frequencies to ensure no unusual values
table(ds$skill_level)

# Drop occupation to keep data set clean
ds <- ds %>%
  dplyr::select(-occupation)

# Transform race as instructed
ds <- ds %>%
  # Create the new 'race_transform' column based on the existing 'race'
  mutate(
    race_transform = case_when(
      # WHITE
      race %in% c("White") ~ 3,
      # BLACK
      race %in% c("Black") ~ 2,
      # OTHER
      race %in% c("Asian-Pac-Islander", "Amer-Indian-Eskimo", 
                  "Other") ~ 1,
      # HANDLE MISSING/UNKNOWN VALUES
      race %in% c("?", "Unknown", "") ~ -99,
      # Catch-all for anything missed (optional, but good practice)
      TRUE ~ -98))

# Check frequencies to ensure no unusual values
table(ds$race_transform)

# Remove observations where race is unknown (-99 or -98)
ds <- ds %>%
  filter(race_transform != -99) %>%
  filter(race_transform != -98)

# Check frequencies to ensure no unusual values
table(ds$race_transform)

# Drop race to keep data set clean
ds <- ds %>%
  dplyr::select(-race)

# Transform age as instructed
ds$age_group <- as.numeric(
  ifelse(ds$age > 65, 5,
         ifelse(ds$age > 49, 4,
                ifelse(ds$age > 34, 3,
                       ifelse(ds$age > 24, 2, 1)))))

# Check frequencies to ensure no unusual values
table(ds$age_group)

# Drop age to keep data set clean
ds <- ds %>%
  dplyr::select(-age)

# Transform hours_per_week as instructed
ds$hours <- as.numeric(
  ifelse(ds$hours_per_week > 40, 3,
         ifelse(ds$hours_per_week >= 20, 2,
                1)))

# Check frequencies to ensure no unusual values
table(ds$hours)

# Drop hours_per_week to keep data set clean
ds <- ds %>%
  dplyr::select(-hours_per_week)

# Transform sex for categorical interpretation
ds$sex_num <- as.numeric(ifelse(ds$sex == "Male", 2, ifelse(ds$sex == "Female", 1, NA)) ) 

# Check frequencies to ensure no unusual values
table(ds$sex_num) 

# Drop sex to keep data set clean
ds <- ds %>% dplyr::select(-sex)

# Check for any unusual observations of income
table(ds$income)

# Transform income for categorical interpretation
ds <- ds %>%
  mutate(
    income_level = case_when(
      income == ">50K" ~ 2,
      TRUE ~ 1))

# Drop income to keep data set clean
ds <- ds %>%
  dplyr::select(-income)

# Check frequencies to ensure no unusual values
table(ds$income_level)

# Check data types
str(ds)

# Convert to factors for categorical interpretation
ds$ed_level <- as.factor(ds$ed_level)
ds$mar_stat <- as.factor(ds$mar_stat)
ds$skill_level <- as.factor(ds$skill_level)
ds$race_transform <- as.factor(ds$race_transform)
ds$age_group <- as.factor(ds$age_group)
ds$hours <- as.factor(ds$hours)
ds$sex_num <- as.factor(ds$sex_num)
ds$income_level <- as.factor(ds$income_level)

# Check variable types
str(ds)

# View data
head(ds)

# Form training sample * Please toggle between 0.3 (for main results) and 0.5 (for robustness check)
set.seed(seed)
test.prop = 0.3
train.size = floor(nrow(ds) * (1 - test.prop))
train.sample <- sample(1:nrow(ds), train.size)

# Define explanatory variables
explanatory <- c("ed_level", "mar_stat", "skill_level", 
                    "race_transform", "age_group", "hours", "sex_num")

# Form explanatory and dependent variables in training sample
X.train <- ds[train.sample, explanatory]
Y.train <- ds[train.sample, "income_level"]

# Form and view training data
train.data <- data.frame(Y.train, X.train)
head(train.data)

# Examine training data
str(train.data)
summary(train.data)

# Form explanatory and dependent variables in test sample
X.test <- ds[-train.sample, explanatory]
Y.test <- ds[-train.sample, "income_level"]

# Form and view test data
test.data <- data.frame(Y.test, X.test)
head(test.data)

# Form full data set Y
X <- ds[, explanatory]
Y <- ds[, "income_level"]

# Form and view full data
full.data <- data.frame(Y, X)
head(full.data)

# Examine full data
str(full.data)
summary(full.data)

# Examine test data
str(test.data)
summary(test.data)

################################################################################
# CODE BLOCK 14: STANDARD CLASSIFICATION TECHNIQUES
################################################################################

# Estimate logistic regression based on variable importance from ensemble models
log.model <- glm(Y.train ~ mar_stat + skill_level + age_group + ed_level + race_transform, data = train.data, family = binomial)

# Generate predictions from logistic regression on the train data
log.prob.train <- predict(log.model, type = "response", newdata = train.data)

# Generate predictions from logistic regression on the test data
log.prob.test <- predict(log.model, type = "response", newdata = test.data)

# Convert probabilities into predictions and format as factors
log.pred.train <- as.factor(ifelse(log.prob.train > 0.5, 2, 1))
log.pred.test <- as.factor(ifelse(log.prob.test > 0.5, 2, 1))

# Calculate train PFR for logistic regression
log.fr.train <- 1 - mean(log.pred.train == train.data$Y.train)
print(log.fr.train)

# Calculate test PFR for logistic regression
log.fr.test <- 1 - mean(log.pred.test == test.data$Y.test)
print(log.fr.test)

# Estimate LDA model based on variable importance from ensemble models
lda.model <- lda(Y.train ~ mar_stat + skill_level + age_group + ed_level + race_transform, data = train.data)

# Generate predictions from LDA on train data
lda.pred.train <- predict(lda.model, newdata = train.data)

# Generate predictions from LDA on test data
lda.pred.test <- predict(lda.model, newdata = test.data)

# Calculate train PFR for LDA
lda.class.train <- lda.pred.train$class
lda.fr.train <- 1 - mean(lda.class.train == train.data$Y.train)
print(lda.fr.train)

# Calculate test PFR for LDA
lda.class.test <- lda.pred.test$class
lda.fr.test <- 1 - mean(lda.class.test == test.data$Y.test)
print(lda.fr.test)

# Estimate QDA model based on variable importance from ensemble models
qda.model <- qda(Y.train ~ mar_stat + skill_level + age_group + ed_level + race_transform, data = train.data)

# Generate predictions from QDA on train data
qda.pred.train <- predict(qda.model, newdata = train.data)

# Generate predictions from QDA on test data
qda.pred.test <- predict(qda.model, newdata = test.data)

# Calculate train PFR for QDA
qda.class.train <- qda.pred.train$class
qda.fr.train <- 1 - mean(qda.class.train == train.data$Y.train)
print(qda.fr.train)

# Calculate test PFR for QDA
qda.class.test <- qda.pred.test$class
qda.fr.test <- 1 - mean(qda.class.test == test.data$Y.test)
print(qda.fr.test)

# Estimate NB model based on variable importance from ensemble models
nb.model <- naiveBayes(Y.train ~ mar_stat + skill_level + age_group + ed_level + race_transform, data = train.data)

# Generate predictions from NB on the train data
nb.pred.train <- predict(nb.model, newdata = train.data)

# Generate predictions from NB on the test data
nb.pred.test <- predict(nb.model, newdata = test.data)

# Calculate train PFR for NB
nb.fr.train <- 1 - mean(nb.pred.train == train.data$Y.train)
print(nb.fr.train)

# Calculate test PFR for NB
nb.fr.test <- 1 - mean(nb.pred.test == test.data$Y.test)
print(nb.fr.test)

################################################################################
# CODE BLOCK 15: APPLY TREE MODELS
################################################################################

# Estimate a deep tree
set.seed(seed)
basic.tree <- tree(Y.train ~ ., data = train.data)

# Print summary of the tree
summary(basic.tree)

# Plot the deep tree to see how sample space is partitioned
plot(basic.tree)
text(basic.tree, pretty = 0, cex = 0.4)

# Generate predictions from basic tree on the train data
tree.pred.train <- predict(basic.tree, newdata = train.data, type = "class")

# Generate predictions from basic tree on the test data
tree.pred.test <- predict(basic.tree, newdata = test.data, type = "class")

# Calculate train PFR for basic tree
tree.fr.train <- 1 - mean(tree.pred.train == train.data$Y.train)
print(tree.fr.train)

# Calculate test PFR for basic tree
tree.fr.test <- 1 - mean(tree.pred.test == test.data$Y.test)
print(tree.fr.test)

# Apply cross validation to find optimal tree size
set.seed(seed)
cv.tree.model <- cv.tree(basic.tree, FUN = prune.misclass)
print(cv.tree.model)

# Visualise trade-off between accuracy and complexity
par(mfrow = c(1, 1))
plot(cv.tree.model$size, cv.tree.model$dev, type = "b")

# Prune tree to 3 leaves as explained in write-up
prune.tree <- prune.misclass(basic.tree, best = 3)

# Plot the pruned tree
plot(prune.tree)
text(prune.tree, pretty = 0, cex = 0.4)

# Generate predictions from pruned tree on train data
prune.pred.train <- predict(prune.tree, newdata = train.data, type = "class")

# Generate predictions from pruned tree on test data
prune.pred.test <- predict(prune.tree, newdata = test.data, type = "class")

# Calculate train pfr for pruned tree
prune.fr.train <- 1 - mean(prune.pred.train == train.data$Y.train)
print(prune.fr.train)

# Calculate test pfr for pruned tree
prune.fr.test <- 1 - mean(prune.pred.test == test.data$Y.test)
print(prune.fr.test)

################################################################################
# CODE BLOCK 16: APPLY BAGGING
################################################################################

# Estimate a bagging model
set.seed(seed)
bag.model <- randomForest(Y.train ~ ., data = train.data, 
                          mtry = 7, importance = TRUE, keep.inbag = TRUE, ntree = 500)

# Print summary and plot variable importance
print(bag.model)
varImpPlot(bag.model)

# Generate predictions on train and test data
bag.pred.train <- predict(bag.model, newdata = train.data, type = "class")
bag.pred.test <- predict(bag.model, newdata = test.data, type = "class")

# Calculate PFR on train data for bagging model
bag.fr.train <- 1 - mean(bag.pred.train == train.data$Y.train)
print(bag.fr.train)

# Calculate PFR on test data for bagging model
bag.fr.test <- 1 - mean(bag.pred.test == test.data$Y.test)
print(bag.fr.test)

################################################################################
# CODE BLOCK 17: APPLY RANDOM FOREST
################################################################################

# Estimate a random forest model
set.seed(seed)
rf.model <- randomForest(Y.train ~ ., data = train.data, 
                        mtry = 3, importance = TRUE, keep.inbag = TRUE, ntree = 500)

# Print summary and plot variable importance
print(rf.model)
varImpPlot(rf.model)

# Generate predictions
rf.pred.train <- predict(rf.model, newdata = train.data, type = "class")
rf.pred.test <- predict(rf.model, newdata = test.data, type = "class")

# Calculate PFR for random forest on train data
rf.fr.train <- 1 - mean(rf.pred.train == train.data$Y.train)
print(rf.fr.train)

# Calculate PFR for random forest on test data
rf.fr.test <- 1 - mean(rf.pred.test == test.data$Y.test)
print(rf.fr.test)

################################################################################
# CODE BLOCK: 18 APPLY GRADIENT BOOSTING
################################################################################

# Define numerical targets
train.data.gb <- train.data
test.data.gb <- test.data
train.data.gb$Y.train <- ifelse(train.data.gb$Y.train == "2", 1, 0)
test.data.gb$Y.test <- ifelse(test.data.gb$Y.test == "2", 1, 0)

# Define the candidate values for interaction depth
candidate_depths <- c(3, 4, 5, 6, 7)

# Create an empty data frame to store CV results
gb.cv.results <- data.frame(Depth = candidate_depths,
                         Best_Trees = NA,
                         CV_Error = NA)

# Calculate error for each candidate depth
for (i in seq_along(candidate_depths)) {
  current_depth <- candidate_depths[i]
  
  # Fit the temporary gbm model
  set.seed(seed)
  temp.model <- gbm(Y.train ~ ., data = train.data.gb,
                    distribution = "bernoulli",
                    n.trees = 2000,
                    interaction.depth = current_depth,
                    shrinkage = 0.01,
                    cv.folds = 5,
                    verbose = FALSE)
  
  # Find the best iteration for this specific depth
  # Setting plot.it = FALSE so we don't generate 6 different plots
  best_iter_temp <- gbm.perf(temp.model, method = "cv", plot.it = FALSE)
  
  # Store the optimal number of trees and the minimum CV error
  gb.cv.results$Best_Trees[i] <- best_iter_temp
  gb.cv.results$CV_Error[i] <- temp.model$cv.error[best_iter_temp]
}

# Print the final table of results
print("Cross-Validation Results:")
print(gb.cv.results)

# Identify the best depth based on CV error
best_depth <- gb.cv.results$Depth[which.min(gb.cv.results$CV_Error)]
cat("\nThe optimal interaction.depth based on CV error is:", best_depth, "\n")

# Estimate a gradient boosting model
set.seed(seed)
gb.model <- gbm(Y.train ~ ., data = train.data.gb,
                distribution = "bernoulli",
                n.trees = 2000,
                interaction.depth = best_depth,
                shrinkage = 0.01,
                cv.folds = 5,
                verbose = FALSE)

# Find best iteration via cross-validation
best.iter <- gbm.perf(gb.model, method = "cv", plot.it = TRUE)

# Generate predicted probabilities
gb.prob.train <- predict(gb.model, newdata = train.data.gb, n.trees = best.iter, type = "response")
gb.prob.test <- predict(gb.model, newdata = test.data.gb, n.trees = best.iter, type = "response")

# Convert probabilities into predictions
gb.pred.train <- ifelse(gb.prob.train > 0.5, 1, 0)
gb.pred.test <- ifelse(gb.prob.test > 0.5, 1, 0)

# Calculate PFR for gbm on train data
gb.fr.train <- mean(gb.pred.train != train.data.gb$Y.train)
print(gb.fr.train)

# Calculate PFR for gbm on test data
gb.fr.test <- mean(gb.pred.test != test.data.gb$Y.test)
print(gb.fr.test)

################################################################################
# CODE BLOCK 19: APPLY SHALLOW NEURAL NETWORK
################################################################################

# Manually carry out 10-fold CV to find optimal number of units
# Define the grid of hidden units to test
size_grid <- seq(1, 10, 1)
num_folds <- 10

# Split data into folds
set.seed(seed) 
folds <- createFolds(train.data$Y.train, k = num_folds)

# Data frame to store results
nn.cv.results <- data.frame(size = size_grid, mean_error = NA)

# Cross-Validation Loop
# NOTE: this can take a long time (often a few minutes)
# Please find the optimal unit sizes in the report document
for (i in 1:length(size_grid)) {
  current_size <- size_grid[i]
  fold_errors <- c()
  
  for (j in 1:num_folds) {
    # Split data
    val_idx <- folds[[j]]
    cv_train <- train.data[-val_idx, ]
    cv_val <- train.data[val_idx, ]
    
    # Train model
    temp_model <- nnet(Y.train ~ ., 
                       data = cv_train, 
                       size = current_size, 
                       decay = 0.001, 
                       maxit = 1000, 
                       trace = FALSE)
    
    # Predict on validation fold
    pred <- predict(temp_model, newdata = cv_val, type = "class")
    error <- 1 - mean(pred == cv_val$Y.train)
    fold_errors <- c(fold_errors, error)
  }
  
  # Store average error for this size
  nn.cv.results$mean_error[i] <- mean(fold_errors)
}

# Identify optimal size
optimal_size <- nn.cv.results$size[which.min(nn.cv.results$mean_error)]
cat("\nThe optimal number of units is:", optimal_size, "\n")

# Estimate the neural network
set.seed(seed)
nn.model <- nnet(Y.train ~ ., 
                data = train.data, 
                size = optimal_size, 
                rang = 0.1, 
                decay = 0.001, 
                maxit = 1000, 
                trace = FALSE)

# Generate predictions on train and test data
nn.pred.train <- predict(nn.model, newdata = train.data, type = "class")
nn.pred.test <- predict(nn.model, newdata = test.data, type = "class")

# Calculate pfr for the Neural Network on train data
nn.fr.train <- 1 - mean(nn.pred.train == train.data$Y.train)
print(nn.fr.train)

# Calculate pfr for the Neural Network on test data
nn.fr.test <- 1 - mean(nn.pred.test == test.data$Y.test)
print(nn.fr.test)

# Visualise the Neural Network structure
plotnet(nn.model, 
        alpha.val = 0.4, 
        struct = c(7, optimal_size, 1),
        pos_col = 'red',         
        neg_col = 'blue',        
        circle_col = 'orange')

################################################################################
# CODE BLOCK 20: RESULTS SUMMARY
################################################################################

final.results <- data.frame(
  Method = c("Logistic Regression",
            "LDA",
            "QDA",
            "Naive Bayes",
            "Deep tree", 
            "Pruned tree", 
            "Bagged trees", 
            "Random forest", 
            "Gradient boosting",
            "Shallow neural network"),
  
  Train_PFR = c(log.fr.train,
                lda.fr.train,
                qda.fr.train,
                nb.fr.train,
                tree.fr.train, 
                prune.fr.train, 
                bag.fr.train, 
                rf.fr.train, 
                gb.fr.train,
                nn.fr.train),
  
  Test_PFR = c(log.fr.test, 
              lda.fr.test,
              qda.fr.test,
              nb.fr.test, 
              tree.fr.test, 
              prune.fr.test, 
              bag.fr.test, 
              rf.fr.test, 
              gb.fr.test,
              nn.fr.test)
)

print(final.results)
print(seed)

################################################################################
# CODE BLOCK 21: FINAL LOGISTIC MODEL
################################################################################

# Final Logistic model
log.model <- glm(Y ~ mar_stat + skill_level + age_group + ed_level + race_transform, data = full.data, family = binomial)
print(log.model)

# Sincerely,
# Group 34