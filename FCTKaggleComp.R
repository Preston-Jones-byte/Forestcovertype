# Libraries ---------------------------------------------------------------

library(tidymodels)
library(vroom)
library(embed)
library(bonsai)
library(tidyverse)
library(themis)
library(discrim)
library(stacks)
library(doParallel)


# Read in Files -----------------------------------------------------------
num_cores <- detectCores()
cl <- makePSOCKcluster(num_cores / 4)
registerDoParallel(cl)

train <- vroom("train.csv")
test <- vroom("test.csv")
train$Cover_Type <- factor(train$Cover_Type)

# EDA ---------------------------------------------------------------------

dplyr::glimpse(train)



# Recipe ------------------------------------------------------------------

# Preprocessing for all models
my_recipe <- recipe(Cover_Type ~ ., data = train) %>%
  step_impute_median(contains("Soil_Type")) %>%  # Impute any missing soil type values
  step_rm(Id) %>%                                # Remove irrelevant ID column
  step_zv(all_predictors()) %>%                  # Remove features with zero variance
  step_normalize(all_numeric_predictors()) 
    


# Boosted Forest ----------------------------------------------------------

boosted_model <- boost_tree(tree_depth = 6,
                            trees = 1000,
                            learn_rate = .22) |> 
  set_engine("lightgbm") |> 
  set_mode("classification")

boosted_wf <- workflow() |> 
  add_recipe(my_recipe)  |> 
  add_model(boosted_model)


fitboosted <- boosted_wf |> fit(data = train)

preds <- predict(fitboosted, new_data =test) |> 
  bind_cols(test) |> 
  rename(Cover_Type =.pred_class) |> 
  select(Id, Cover_Type)

vroom_write(x= preds, file= "./boostedsubmission.csv", delim=",") 



# Random Forest -----------------------------------------------------------


my_mod <- rand_forest(mtry = 15,
                      min_n=2,
                      trees=1000) %>% #Type of model
             set_engine("ranger") %>% # What R function to use
             set_mode("classification")
 
 ## Set Workflow
randf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

fitrf_wf <- randf_wf |> fit(data = train)


#Predictions
preds <- predict(fitrf_wf, new_data =test) |> 
  bind_cols(test) |> 
  rename(Cover_Type =.pred_class) |> 
  select(Id, Cover_Type)

vroom_write(x= preds, file= "./rfsubmission.csv", delim=",") 


# Nueral Network ----------------------------------------------------------

nn_model <- mlp(hidden_units = 10,
                epochs = 50) %>%
  set_engine("keras") %>%
  set_mode("classification")

nn_wf <- workflow() %>%
  add_model(nn_model) %>%
  add_recipe(my_recipe)


# Stack -------------------------------------------------------------------

untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples()

# Define the Folds for Cross-Validation
folds <- vfold_cv(train, v = 5, repeats = 1)

## rf stack -----------------------------------------------------------



# Resample for Stack as a 'tunedModel'
rf_stack <- fit_resamples(randf_wf,
                                  resamples = folds,
                                  metrics = metric_set(accuracy, roc_auc),
                                  control = tunedModel)


## boosted_wf stack ----------------------------------------------------------



# Resample for Stack as a 'tunedModel'
XGboosted_stack <- fit_resamples(boosted_wf,
                                         resamples = folds,
                                         metrics = metric_set(accuracy,roc_auc),
                                         control = tunedModel)

## nb_wf stack ----------------------------------------------------

nn_stack <- fit_resamples(nn_wf,
                          resamples = folds,
                          metrics = metric_set(accuracy, roc_auc),
                          control = tunedModel)

## Combine stack elements ---------------------------------------------------------

my_stack <- stacks() %>%
  add_candidates(XGboosted_stack) %>% 
  add_candidates(rf_stack) %>%
  add_candidates(nn_stack)

# Blend Predictions and Fit Members to Create a Stacked Model
stack_mod <- my_stack %>%
  blend_predictions() %>%
  fit_members()

# Make Predictions from the Stacked Model
stack_preds <- stack_mod %>%
  predict(new_data = test, type = "class")

# Prepare the Predictions for Output
 Preds <-  stack_preds %>%
  bind_cols(test) %>%
  rename(Cover_Type =.pred_class) %>%
  select(Id, Cover_Type)

# Write the Predictions
vroom_write(Preds, "stack_submission.csv", delim = ",")

stopCluster(cl)

