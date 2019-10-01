library(healthcareai)

str(pima_diabetes)

#--------------------------------------------------------
# Easy Machine Learning
#--------------------------------------------------------
quick_models <- machine_learn(pima_diabetes, patient_id, outcome = diabetes)
quick_models

predictions <- predict(quick_models)
predictions

plot(predictions)

quick_models %>% 
  predict(outcome_groups = 2) %>% 
  plot()

# Data Profiling
missingness(pima_diabetes) %>% 
  plot()


#--------------------------------------------------------
# Data Preparation
#--------------------------------------------------------
split_data <- split_train_test(d = pima_diabetes,
                               outcome = diabetes,
                               p = 0.8,
                               seed = 1)

prepped_training_data <- prep_data(split_data$train, patient_id, outcome = diabetes,
                                   center = TRUE, scale = TRUE,
                                   collapse_rare_factors = FALSE)


# write.csv(prepped_training_data,'diabetes_prepared.csv')

head(prepped_training_data)

# prep_data object with only center, scale and impute missing values set to true. Rest is set to false (default is true)
while (FALSE) {
  
  prepped_training_data <- prep_data(split_data$train, patient_id, outcome = diabetes,
                                     center = TRUE, scale = TRUE,
                                     collapse_rare_factors = FALSE, 
                                     impute = TRUE, 
                                     remove_near_zero_variance = FALSE, 
                                     add_levels = FALSE, 
                                     logical_to_numeric = FALSE, 
                                     factor_outcome = FALSE)
  
  
}


#--------------------------------------------------------
# Model Training
#--------------------------------------------------------

models <- tune_models(d = prepped_training_data,
                      outcome = diabetes,
                      tune_depth = 25,
                      metric = "PR")

evaluate(models, all_models = TRUE)

models["Random Forest"] %>% 
  plot()


#--------------------------------------------------------
# Faster Model Training
# flash_models use fixed sets of hyperparameter values to train the models
# so you still get a model customized to your data, 
# but without burning the electricity and time to precisely optimize all the details. 
# Here we’ll use models = "RF" to train only a random forest.
# If you want to train a model on fixed hyperparameter values, but you want to choose those values, 
# you can pass them to the hyperparameters argument of tune_models. 
# Run get_hyperparameter_defaults() to see the default values and get a list you can customize.
#--------------------------------------------------------


#--------------------------------------------------------
# Model Interpretation
#--------------------------------------------------------

# In this plot, the low value of weight_class_normal signifies that people with normal weight 
# are less likely to have diabetes. Similarly, plasma glucose is associated with increased risk of 
# diabetes after accounting for other variables.
interpret(models) %>% 
  plot()


# Tree based methods such as random forest and boosted decision trees can’t provide coefficients 
# like regularized regression models can, but they can provide information about how important each 
# feature is for making accurate predictions.
get_variable_importance(models) %>%
  plot()


# The explore function reveals how a model makes its predictions. It takes the most important features 
# in a model, and uses a variety of “counterfactual” observations across those features to see what 
# predictions the model would make at various combinations of the features.
explore(models) %>% 
  plot()

#--------------------------------------------------------
# Prediction
#--------------------------------------------------------

predict(models)

test_predictions <- 
  predict(models, 
          split_data$test, 
          risk_groups = c(low = 30, moderate = 40, high = 20, extreme = 10)
          )

# > Prepping data based on provided recipe
plot(test_predictions)


#--------------------------------------------------------
# Saving, Moving, and Loading Models
#--------------------------------------------------------

save_models(models, file = "my_models.RDS")
models <- load_models("my_models.RDS")

#--------------------------------------------------------
# A Regression Example:
#
# All the examples above have been classification tasks, 
# redicting a yes/no outcome. Here’s an example of a full 
# regression modeling pipeline on a silly problem: 
# predicting individuals’ ages. The code is very similar to classification.
#--------------------------------------------------------

regression_models <- machine_learn(pima_diabetes, patient_id, outcome = diabetes)

summary(regression_models)

# Let’s make a prediction on a hypothetical new patient. Note that the model handles missingness in
# insulin and a new category level in weight_class without a problem (but warns about it).

new_patient <- data.frame(
  pregnancies = 0,
  plasma_glucose = 80,
  diastolic_bp = 55,
  skinfold = 24,
  insulin = NA,
  weight_class = "???",
  pedigree = .2,
  age = 24)

predict(regression_models, new_patient)

