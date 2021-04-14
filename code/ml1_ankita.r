# Packages ####

# resampling, splitting and validation
library(rsample)
# feature engineering or processing
library(recipes)
# specifying models
library(parsnip)
# tuning
library(tune)
# tuning parameters
library(dials)
# measure model performance
library(yardstick)
# variable importance plots
library(vip)
# combining feature engineering and model specification
library(workflows)

# data manipulation
library(dplyr)
library(purr)
library(tidyr)

l# plotting
library(ggplot2)
# parallelism
library(doFuture)
library(parallel)
# timing
library(tictoc)
# viz
library(skimr)

# library(tidymodels)

# Data ####
data(credit_data, package = 'modeldata')
# tibble wraps the data easy to use. Dataframes & tibbles are interchangeable
credit <- credit_data %>% as_tibble()
credit
?modeldata::credit_data

# EDA ####
ggplot2::ggplot(credit_data, ggplot2::aes(Status)) + ggplot2::geom_bar()
ggplot2::ggplot(credit, ggplot2::aes(Status, Amount)) + ggplot2::geom_violin()
ggplot2::ggplot(credit, ggplot2::aes(Status, Age)) + ggplot2::geom_violin()
ggplot2::ggplot(credit, ggplot2::aes(Status, Income)) + ggplot2::geom_violin()

# Want to see changes in age, income with Status. Try with geom_jitter 
ggplot2::ggplot(credit, ggplot2::aes(Age, Income, color=Status)) + ggplot2::geom_point()

# with geom_hex. Package: 'hexbin'
ggplot2::ggplot(credit, ggplot2::aes(Age, Income, color=Status)) + ggplot2::geom_hex()
+ ggplot2::facet_wrap(~ Status) +
    ggplot2::theme(legend.position='bottom')

# Wrapping up in a single line removed the error
ggplot2::ggplot(credit, ggplot2::aes(Age, Income, color=Status)) + ggplot2::geom_hex() + ggplot2::scale_fill_gradient(low='red', high='blue') + ggplot2::facet_wrap(~ Status) + ggplot2::theme(legend.position = 'bottom')

#ggplot2::theme(legend.position='bottom')

# Split the Data ####
# Seed 42 is invalid
set.seed(8261)

# from {sample}
# 80% of the data to be used for Training Set. Sample is stratified according to Status
credit_split <- initial_split(credit, prop=0.8, strata = 'Status')

#Training: 3565; Testing: 889; Total Dataset: 4454
credit_split
class(credit_split)

train <- training(credit_split)
test <- testing(credit_split)

train
test

# Feature Engg or Preprocessing ####
# from {recipes} Packages are put in {} braces
# goal: relate outcome to inputs
# outcomes: response, y, label(CS background), target, output, known, result, dependent variable, event, range
# inputs: predictors, x, features, covariates, variables, data, attributes, independent variable, descriptors, context, subject variables

# 2 Primary ways to deal with imbalanced data: 
# 1) upsample the minority (CS prefers)
# 2) downsample the majority class (Statistician prefers)

# You get one less dummy variable than original columns
# Here we have 4 colors, 3 columns of Dummy Variables. These colors are in rows
library(useful)
colors1 <- tibble(Color=c('blue', 'green', 'blue', 'red', 'red', 
                          'yellow', 'green'))
# one-hot encoding, no dropping the baseline
build.x(~ Color, data=colors1)
# modern tools
build.x(~ Color, data = colors1, contrasts = FALSE)


# Telling Status is outcome ~ . everything else is inputs. 
rec1 <- recipe(Status ~ ., data = train) %>%
    # Does sampling for recipe. Way of specifying to specifically get step_downsample() from themis package, instead of installing the whole package
    # under_ratio will have 1.2 rows of good class for every row of the bad class 
    # not really needed for xgboost
    themis::step_downsample(Status, under_ratio = 1.2) %>%
    # Column by column basis center & scale them.
    # Not really needed for xgboost: will be useful for linear models, neural n/w etc
    step_normalize(Age, Price) %>%
    step_other(all_nominal(), -Status, other = 'Misc') %>%
    step_nzv(all_predictors()) %>% # Removes columns with near 0 variance
    # imputation: filling in missing values, not needed for xgboost
    step_modeimpute(all_nominal(), -Status) %>%
    step_knnimpute(all_numeric()) %>% 
    # Convert categorical variable to Dummy Variable/Indicator. one_hot = TRUE don't
    # drop a baseline. 
    step_dummy(all_nominal(), -Status, one_hot = TRUE) 
rec1

# 2200 rows of data for down sampling 
# Also have dummy variable, no missing values. All done in recipes in just a few steps
rec1 %>% prep() %>% juice() %>% View



# Model Specification ####
# from {parsnip}

# boosted tree
boost_tree()
# BART: dbart
# gbm
# catboost
# xgboost
# LightGBM: May NY R Meetup: meetup.com/nyhackr

# model types
# model modes
# engines
# parameters

parsnip::decision_tree()
parsnip::rand_forest()
parsnip::svm_poly()
parsnip::svm_rbf()
parsnip::linear_reg()
multinom_reg()
logistic_reg()
boost_tree()
surv_reg()

# Highly recommended: gam, multilinearity handled good

linear_reg()
linear_reg() %>% set_engine('glmnet')
linear_reg() %>% set_engine('lm')
linear_reg() %>% set_engine('stan')
linear_reg() %>% set_engine('spark')
linear_reg() %>% set_engine('keras')

show_engines('linear_reg')

show_engines('logistic_reg')

# Deep Learning in R with keras
logistic_reg() %>% set_engine('keras')

show_engines('boost_tree')
show_engines('surv_reg')

boost_tree() %>% set_engine('xgboost')
boost_tree() %>% set_engine('xgboost') %>% set_mode('classification')
boost_tree(mode = 'classification') %>% set_engine('xgboost')

xg_spec1 <- boost_tree(mode = 'classification', trees = 100) %>%
    set_engine('xgboost')
xg_spec1

# Workflows ####

# from {workflows}

flow1 <- workflow() %>%
    add_recipe(rec1) %>%
    add_model(xg_spec1)
flow1



fit1 <- fit(flow1, data = train)
fit1
fit1 %>% class()
fit1 %>% extract_model() %>% class()

# variable importance plot: extract_model comes from{tune}
fit1 %>% extract_model() %>% vip()

# How did we do? Now we need validation Data. Test Data is @very end after 
# we are done with Tuning, Model Fitting etc.
# AIC, accuracy(either right/wrong. Does not specify % of right/wrong), logloss, AUC: Different ways of saying how wrong you are!
# from {yardstick}

# mn_log_loss: Log loss to the Classification Model
# Log_Loss: lower the better, accuracy: higher the better, AUC: between 0 to 1, greater the better
loss_fn <- metric_set(accuracy, roc_auc, mn_log_loss)
loss_fn

# Validation: Analysis portion & assessment portion
# How wrong my Model is? val_split has the training portion and the validation portion
val_split <- validation_split(data=train, prop=0.8, strata='Status')
val_split
val_split$splits[[1]]
val_split$splits[[1]] %>% class()

# from {tune}

val1 <- fit_resamples(object=flow1, resamples=val_split, metrics=loss_fn)
val1
val1$.metrics[[1]]

val1 %>% collect_metrics()
## My Understanding:
# cross-validation: Fit on the training of the val_split and calculate the metrics
# on the val_split$splits. Train on the validation training set and then predict on the rest of the
# validation data to find how well the model training validation worked.

## Jared:
# split training data into training.1 and val
# fit a model on training.1
# use val & model to make predictions
# compare predictions from model with true outcome variable in val
# use AUC, accuracy or log loss for that comparison

# cross-validation: 10 folds is standard, 5: faster
cv_split <- vfold_cv(data = train, v=5, strata = 'Status')
# After all the Training & Predictions on the split data, the Avg of the 5 Folds is taken
cv_split
cv_split %>% class()
cv_split$splits[[1]]
cv_split$splits[[1]]  %>% class()

# validation
val1 <- fit_resamples(object=flow1, resamples=val_split, metrics=loss_fn)
# cross-validation: Train > Test; Train > Test all the time
cv1 <- fit_resamples(object=flow1, resamples=cv_split, metrics=loss_fn)
cv1
cv1$.metrics[[1]]
cv1$.metrics[[2]]

cv1 %>% collect_metrics()

# Fit another Model with different number of trees
xg_spec2 <- boost_tree(mode = 'classification', trees = 300) %>%
    set_engine('xgboost')

xg_spec1
xg_spec2

flow2 <- flow1 %>%
    update_model(xg_spec2)
# 100 Trees
flow1
# 300 Trees
flow2

# validation comparison
val2 <- fit_resamples(object = flow2, resamples = val_split, metrics = loss_fn)
val2
val1

val1 %>% collect_metrics()
val2 %>% collect_metrics()

# Model with learning rate
xg_spec3 <- boost_tree(mode = 'classification', trees = 300, learn_rate = 0.15) %>%
    set_engine('xgboost')

flow3 <- flow2 %>% 
    update_model(xg_spec3)

val3 <- fit_resamples(object = flow3, resamples = val_split, metrics = loss_fn)

val3 %>% collect_metrics()

# More Recipes ####


# xgboost handles missing data all by itself
rec2 <- recipe(Status ~ ., data = train) %>%
    # Does sampling for recipe. Way of specifying to specifically get step_downsample() from themis package,
    # instead of installing the whole package
    themis::step_downsample(Status, under_ratio = 1.2) %>%
    step_other(all_nominal(), -Status, other = 'Misc') %>%
    step_nzv(all_predictors()) %>% # Removes columns with near 0 variance
    step_dummy(all_nominal(), -Status, one_hot = TRUE) 
rec2


flow4 <- flow3 %>%
    update_recipe(rec2)
flow4

val4 <- fit_resamples(flow4, resamples=val_split, metrics=loss_fn)
val4 %>% collect_metrics()


# Imbalanced Data ####
rec3 <- recipe(Status ~ ., data = train) %>%
    step_other(all_nominal(), -Status, other = 'Misc') %>%
    step_nzv(all_predictors()) %>% # Removes columns with near 0 variance
    step_dummy(all_nominal(), -Status, one_hot = TRUE) 
rec3

#scale_pos_weight: ratio of +ve to -ve
scaler <- train %>%
    count(Status) %>%
    pull(n) %>%
    rev() %>%
    reduce(`/`)
2561/1004
scaler

xg_spec5 <- boost_tree(mode = 'classification',
                       trees = 100, learn_rate = 0.15) %>%
    set_engine('xgboost', scale_pos_weight=scaler)
xg_spec5   

flow5 <- flow4 %>%
    update_model(xg_spec5)
flow5

val5 <- fit_resamples(flow5, resamples=val_split, metrics=loss_fn)
# looks like val5 is better than val4
val5 %>% collect_metrics()
val4 %>% collect_metrics()

# Tune Parameters ####

# from {tune} and {dials}

xg_spec6 <- boost_tree(mode = 'classification',
                       learn_rate = 0.15, 
                       tree_depth = 4, 
                       tree=tune()
) %>% 
    set_engine('xgboost', scale_pos_weight=!!scaler)
xg_spec6

flow6 <- flow5 %>%
    update_model(xg_spec6)
flow6

# fails because our workflow has a parameter that needs tuning
val6 <- fit_resamples(flow6, resamples=val_split, metrics=loss_fn)
# neither will fit()

# How do we choose which parameter?
# run in parallel
registerDoFuture()
#cl <- makeCluster(6) # don't try this on your personal laptop
#plan(cluster, workers=cl)


options(tidymodels.dark=TRUE)
tic()
#Sys.sleep(5)


tune6_val <- tune_grid(flow6,
                       resamples = val_split,
                       grid=30,
                       metrics = loss_fn,
                       control = control_grid(verbose = TRUE, allow_par = TRUE)
)
toc()
tune6_val %>% collect_metrics()

# Trees with higher roc-auc is a better choice
tune6_val %>% show_best(metric = 'roc_auc')

#View the metrics for top 30 trees 
tune6_val %>% show_best(metric = 'roc_auc', n=30) %>% View()

tic()
tune6_cv <- tune_grid(
            flow6,
            resamples = cv_split,
            grid = 30,
            metrics = loss_fn,
            control = control_grid(verbose = TRUE, allow_par = TRUE)
)
toc()
# cv got little better this time, more randomness is better
tune6_cv %>% show_best(metric = 'roc_auc')
tune6_val %>% show_best(metric = 'roc_auc')

tune6_cv %>% autoplot()

# Other Tuning Parameters ####
xg_spec7 <- boost_tree(
    mode = 'classification',
    trees = tune(),
    learn_rate = 0.15,
    tree_depth = tune(),
    sample_size = tune()
) %>%
    set_engine('xgboost', scale_pos_weight=!!scaler)

flow7 <- flow6 %>%
    update_model(xg_spec7)
flow7

flow7 %>% parameters()
flow7 %>% parameters() %>% pull(object)

# from {dials}
trees()
trees(range = c(20, 800))

tree_depth()
tree_depth(range = c(2, 8))

sample_size()
sample_size(range = c(5, 20))

# range 30% to 100%
sample_prop()
sample_prop(range = c(0.3, 1))

params7 <- flow7 %>% parameters() %>%
    update(
        trees=trees(range = c(20, 800)),
              tree_depth=tree_depth(range = c(2, 8)),
              sample_size=sample_prop(range = c(0.3, 1))
        
    )
params7 %>% pull(object)


tic()
tune7_val <- tune_grid(
    flow7,
    resamples = val_split,
    param_info = params7,
    grid = 40, # nuances less, number of trees & sample size to randomize a lot
    metrics = loss_fn, 
    control = control_grid(verbose = TRUE, allow_par = TRUE)
)
toc()

tune7_val %>% show_best(metric = 'roc_auc') %>% View()
tune7_val %>% autoplot(metric='roc_auc')

# grid_max_entropy tries to space out the numbers that you are choosing randomly
grid7 <- grid_max_entropy(params7, size = 80)
grid7

tic()
tune7_val.1 <- tune_grid(
    flow7,
    resamples = val_split,
    metrics = loss_fn,
    control = control_grid(verbose = TRUE, allow_par = TRUE),
    grid = grid7
)
toc()

tune7_val.1 %>% show_best(metric = 'roc_auc')

# Bayesian Tuning/Bayesian Optimization. Loss lot of Parallelilism
# Things could take longer
tic()
tune7_val.2 <- tune_bayes(
    flow7,
    resamples = val_split,
    iter = 30,
    metrics = loss_fn,
    param_info = params7,
    control = control_bayes(verbose = TRUE, no_improve = 8)
)
toc()

tune7_val.2 %>% autoplot(metric='roc_auc')
tune7_val.2 %>% show_best(metric='roc_auc')

# Simulated Annealing from {finetune}
# a worst result and from there get other better result
boost_tree(
    mode = 'classification',
    learn_rate = 0.15,
    trees = 157,
    tree_depth = 2,
    sample_size = 0.958
) %>% 
    set_engine('xgboost', scale_pos_weight=!!scaler)


# Finalize Model ####
rec8 <- recipe(Status ~ ., data = train) %>%
    step_other(all_nominal(), -Status, other = 'Misc') %>%
    step_nzv(all_predictors(), freq_cut = tune()) %>% # Removes columns with near 0 variance
    step_dummy(all_nominal(), -Status, one_hot = TRUE) 
rec8

flow8 <- flow7 %>%
    update_recipe(rec8)
flow8

flow8 %>% parameters()

params8 <- flow8 %>%
    parameters() %>%
    update(
        trees=trees(range = c(20, 800)),
        tree_depth=tree_depth(range = c(2, 8)),
        sample_size=sample_prop(range = c(0.3, 1)),
        freq_cut=freq_cut(range = c(5, 25))
    )

# Grid Search again coz faster than Gradient

grid8 <- grid_max_entropy(params8, size = 80)
grid8

tic()
tune8_val <- tune_grid(
    flow8,
    resamples = val_split,
    grid = grid8,
    metrics = loss_fn,
    control = control_grid(verbose = TRUE, allow_par = TRUE)
)
toc()


tune8_val %>% show_best(metric = 'roc_auc')
best_params <- tune8_val %>% select_best(metric = 'roc_auc')

# Fine Tuning in much smaller space
params8.1 <- flow8 %>%
    parameters() %>%
    update(
        trees=trees(range = c(200, 220)),
        tree_depth=tree_depth(range = c(1, 5)),
        sample_size=sample_prop(range = c(0.5, .8)),
        freq_cut=freq_cut(range = c(7, 12))
    )

grid8.1 <- grid_max_entropy(params8.1, size = 80)
tic()
tune8_val.1 <- tune_grid(
    flow8,
    resamples = val_split,
    grid = grid8.1,
    metrics = loss_fn,
    control = control_grid(verbose = TRUE, allow_par = TRUE)
)
toc()
tune8_val.1 %>% show_best(metric = 'roc_auc')

flow8_final <- flow8 %>% finalize_workflow(best_params)
flow8_final

val8 <- fit_resamples(flow8_final, resamples=val_split, metrics=loss_fn)
val8 %>% collect_metrics()

# Last Fit ####
# taking the original training data
results8 <- last_fit(flow8_final, split=credit_split, metrics=loss_fn) 
results8 %>% collect_metrics()

# Fit on Entire Data ####
fit8 <- fit(flow8_final, data = credit)
fit8

# Predictions on New Data ####
# this line won't work unless insert_data_here is a real data.frame
preds8 <- predict(fit8, new_data = insert_data_here)




#ggplot2(credit, aes(x=Status, y=Amount)) + geom_violin()




