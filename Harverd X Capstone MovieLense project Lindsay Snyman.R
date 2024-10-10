# Load necessary libraries 
library(tidyverse)    # For data manipulation and visualization
library(recosystem)   # For matrix factorization and recommender system
library(caret)        # For data partitioning and cross-validation
library(data.table)   # For efficient data handling


#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes for loading required package: tidyverse and package caret
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")


# The Validation subset will be 10% of the MovieLens data.
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
#Make sure userId and movieId in validation set are also in edx subset:
validation <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# A random seed was set for reproducibility
set.seed(2024)

# 30% of the edx dataset was sampled for the analysis
sample_size <- 0.30
edx_sample <- edx %>% sample_frac(sample_size)

# Inspection of the sample dataset
summary(edx_sample)
glimpse(edx_sample)

# A basic EDA: Distribution of Ratings, was performed
ggplot(edx_sample, aes(x = rating)) +
  geom_histogram(binwidth = 0.5, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Ratings in edx Sample Dataset (30%)",
       x = "Rating",
       y = "Count")

#  The number of unique users and movies in the sample is oberved
num_users <- edx_sample %>% distinct(userId) %>% nrow()
num_movies <- edx_sample %>% distinct(movieId) %>% nrow()
cat("Number of unique users in the sample:", num_users, "\n")
cat("Number of unique movies in the sample:", num_movies, "\n")

# The sample dataset is split into training and testing sets (80/20 split)
train_index <- createDataPartition(edx_sample$rating, times = 1, p = 0.8, list = FALSE)
train_set <- edx_sample[train_index, ]
test_set <- edx_sample[-train_index, ]

# Ensure test set contains only users and movies present in the training set
test_set <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Convert data into the format required for `recosystem`
train_data <- data_memory(user_index = train_set$userId,
                          item_index = train_set$movieId,
                          rating = train_set$rating,
                          index1 = TRUE)

test_data <- data_memory(user_index = test_set$userId,
                         item_index = test_set$movieId,
                         rating = test_set$rating,
                         index1 = TRUE)

# Initialized the recommender model from `recosystem`
recommender <- Reco()

# Tune model to find optimal hyperparameters
# Tuning over a different range of parameters for more detailed exploration
tune_result <- recommender$tune(train_data, opts = list(dim = c(10, 20, 40),
                                                        costp_l2 = c(0.01, 0.1, 0.2),
                                                        costq_l2 = c(0.01, 0.1, 0.2),
                                                        lrate = c(0.05, 0.1, 0.2),
                                                        niter = 15,
                                                        nthread = 4))

# Output the best hyperparameters found during tuning
cat("Best tuning parameters:\n")
print(tune_result$min)

# Train the model using the best hyperparameters
recommender$train(train_data, opts = c(tune_result$min, niter = 30))

# Predict ratings on the test set
predicted_ratings <- recommender$predict(test_data, out_memory())

# Function to calculate RMSE
calculate_rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}

# Calculate RMSE for the test set
test_rmse <- calculate_rmse(test_set$rating, predicted_ratings)
cat("RMSE on test set:", test_rmse, "\n")


# Evaluate the model on the validation set
validation_data <- data_memory(user_index = validation$userId,
                               item_index = validation$movieId,
                               rating = validation$rating,
                               index1 = TRUE)

validation_predictions <- recommender$predict(validation_data, out_memory())
validation_rmse <- calculate_rmse(validation$rating, validation_predictions)
cat("RMSE on validation set:", validation_rmse, "\n")

# Visualize actual vs. predicted ratings for the validation set
results_df <- data.frame(actual = validation$rating, predicted = validation_predictions)
ggplot(results_df, aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.4, color = "darkred") +
  geom_abline(slope = 1, intercept = 0, color = "blue") +
  labs(title = "Actual vs Predicted Ratings",
       x = "Actual Rating",
       y = "Predicted Rating") +
  theme_minimal()

# Visualize the distribution of errors
results_df <- results_df %>%
  mutate(error = actual - predicted)

ggplot(results_df, aes(x = error)) +
  geom_histogram(binwidth = 0.2, fill = "purple", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Prediction Errors",
       x = "Prediction Error",
       y = "Count") +
  theme_minimal()

# Summary of model performance
cat("Matrix Factorization using recosystem achieved an RMSE of", validation_rmse, "on the validation set.\n")