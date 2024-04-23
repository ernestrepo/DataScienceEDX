###########################################################################################################################
# Create edx set, validation set (final hold-out test set) - code provided by HarvardX: PH125.9x
###########################################################################################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(scales)
library(ggplot2)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###########################################################################################################################
# Exploratory Analysis
###########################################################################################################################

#Number of rows in dataset
nrow(edx)

#Number of users in dataset
n_distinct(edx$userId)

#Number of movies in dataset
n_distinct(edx$movieId)

#Validate if there is null values
sapply(edx, function(x) sum(is.na(x)))

#Top 10 most rated movies
edx %>% 
  group_by(title) %>%
  summarize(Count_Rating = n()) %>%
  top_n(10,Count_Rating) %>%
  arrange(desc(Count_Rating))

#Plot the distribution of the ratings
ggplot(data = edx, aes(x = rating)) +
  geom_histogram(binwidth = 0.2, fill = "cadetblue3", color = "white") +
scale_y_continuous(breaks = c(1000000, 2000000), labels = c("1", "2")) +
  labs(x = "User Rating", y = "Millions", caption = "Source: edx dataset")

edx %>% group_by(movieId) %>%
  summarise(ave_rating = sum(rating)/n()) %>%
  ggplot(aes(ave_rating)) +
  geom_histogram(bins=30, fill= "cadetblue3", color = "white") +
  labs(x = "Avg. Rating", y = "# of movies", caption = "Source: edx dataset") 

# Plot number of ratings by user in the edx dataset

edx %>% group_by(userId) %>%
  summarise(ave_rating = sum(rating)/n()) %>%
  ggplot(aes(ave_rating)) +
  geom_histogram(bins=30, fill="cadetblue3" ,color = "white") +
  labs(x = "Avg. Rating", y = "# of users", caption = "Source: edx dataset")

edx %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram( bins=30, fill= "cadetblue3",color = "white") +
  scale_x_log10() +
  labs(x = "Users", y = "# of ratings", caption = "Source: edx dataset")

#Exploring genres doing separation with the symbol "|" taking the first
#First genre by the total of genres
edx %>% separate_longer_delim(genres, delim = "|") %>%
  group_by(genres) %>%
  summarise(count = n(), rating = round(mean(rating), 2)) %>%
  arrange(desc(count))

# Group and list top 10 movie titles based on # of ratings
edx %>% group_by(title) %>%
  summarise(n = n()) %>%
  arrange(desc(n)) %>%
  top_n(10)

#Checking if there is a bias by year of rating
edx <- edx %>% mutate(title = str_trim(title)) %>%
  # split title column to two columns: title and year
  extract(title, c("title_temp", "year"), regex = "^(.*) \\(([0-9 \\-]*)\\)$", remove = F) %>%
  # for series take debut date
  mutate(year = if_else(str_length(year) > 4, as.integer(str_split(year, "-", simplify = T)[1]), as.integer(year))) %>%
  # replace title NA's with original title
  mutate(title = if_else(is.na(title_temp), title, title_temp)) %>%
  # drop title_tmp column
  select(-title_temp)
# Plot average rating by year of release in the edx dataset
edx %>% group_by(year) %>%
  summarise(rating = mean(rating)) %>%
  ggplot(aes(year, rating)) +
  geom_point() +
  geom_smooth() +
  labs(x = "Release Year", y = "Avg Rating", caption = "Source: edx dataset")

# Plot # of ratings by year of release in the edx dataset
edx %>% group_by(year) %>%
  summarise(count = n()) %>%
  ggplot(aes(year, count)) +
  geom_line() +
  scale_y_continuous(breaks = seq(0, 800000, 200000), labels = seq(0, 800, 200)) +
  labs(x = "Release Year", y = "# of Ratings", caption = "Source: edx dataset") 

###########################################################################################################################
# Methods - Train and Test set
###########################################################################################################################

#Create training and test set from edx dataframe

set.seed(2024, sample.kind = "Rounding")
test_index<-createDataPartition(y=edx$rating, times = 1, p = 0.2, list = FALSE)
train_set<-edx[-test_index,]
temp <-edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test_set <- temp %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set) 
train_set <- rbind(train_set, removed)

# Remove temporary files to tidy environment
rm(test_index, temp, removed) 

#To save the results a table will be use to compare the final results
objective_rmse <- 0.86490
df_results <- data.frame(Method = "Objective", RMSE = as.character(objective_rmse), Difference = "-")

# Calculate the overall average rating across all movies included in train set
mu_hat <- mean(train_set$rating)

#Calculate RMSE between each rating included in test set and the overall average
naive_rmse<-RMSE(test_set$rating, mu_hat)

#Validating that any number other than mu_hat would result into higher RMSE
predictions <- rep(2.5, nrow(test_set))
RMSE(test_set$rating, predictions)

#Estimating the movie effect (b_i)
movie_avg<-train_set %>% 
  group_by(movieId) %>%
  summarise(b_i= mean(rating - mu_hat))

movie_avg %>% qplot(b_i, geom ="histogram", bins = 30, data = ., color = I("white"),fill = I("cadetblue"))

# Predict ratings adjusting for movie effects
predicted_ratings<-mu_hat+test_set%>%
  left_join(movie_avg, by = "movieId") %>%
  pull(b_i)

# Calculate RMSE based on movie effects model
model_1_rmse<-RMSE(test_set$rating, predicted_ratings)

#Save result and show difference between objective
df_results<-bind_rows(df_results,
                      data.frame(Method = "Movie  Effect", RMSE = as.character(round(model_1_rmse,5)), 
                                 Difference = as.character(round(model_1_rmse,5)-objective_rmse)))

df_results %>% knitr::kable()

# Modeling user effect

train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, fill="cadetblue3",color = "white")


user_avg <- train_set %>%
  left_join(movie_avg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u= mean(rating - mu_hat - b_i))

#Predict rating using movie and user effect  
predicted_ratings<-test_set %>%
  left_join(movie_avg, by="movieId") %>%
  left_join(user_avg, by = "userId") %>%
  mutate(pred=mu_hat + b_u + b_i) %>%
  pull(pred)

#Get the RMSE result from movie and user effect
model_2_rmse<-RMSE(predicted_ratings, test_set$rating)

#Add the RMSE to result table
df_results<-bind_rows(
  df_results,
  data.frame(Method = "Movie  + User Effect", RMSE= as.character(round(model_2_rmse,5))
             ,Difference =as.character(round(model_2_rmse-objective_rmse,5)))
)

df_results %>% knitr::kable()

# Testing the genre effect

genre_avg <- train_set %>%
  left_join(movie_avg, by = "movieId") %>%
  left_join(user_avg, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g= mean(rating - mu_hat - b_i - b_u))

predicted_ratings<-test_set %>%
  left_join(movie_avg, by="movieId") %>%
  left_join(user_avg, by = "userId") %>%
  left_join(genre_avg, by = "genres") %>%
  mutate(pred=mu_hat+b_u+b_i+b_g) %>%
  pull(pred)

model_3_rmse<-RMSE(predicted_ratings, test_set$rating)

#Add the RMSE to result table
df_results<-bind_rows(
  df_results,
  data.frame(Method = "Movie, User, Genre Effect", RMSE= as.character(round(model_3_rmse,5))
             ,Difference =as.character(round(model_3_rmse-objective_rmse,5)))
)

df_results %>% knitr::kable()

# Testing the year effect

year_avg <- train_set %>%
  left_join(movie_avg, by = "movieId") %>%
  left_join(user_avg, by = "userId") %>%
  left_join(genre_avg, by = "genres")%>%
  group_by(year) %>%
  summarize(b_y= mean(rating - mu_hat - b_i - b_u - b_g))

predicted_ratings<-test_set %>%
  left_join(movie_avg, by="movieId") %>%
  left_join(user_avg, by = "userId") %>%
  left_join(genre_avg, by = "genres") %>%
  left_join(year_avg, by = "year")%>%
  mutate(pred=mu_hat+b_u+b_i+b_g+b_y) %>%
  pull(pred)

model_4_rmse<-RMSE(predicted_ratings, test_set$rating)

#Add the RMSE to result table
df_results<-bind_rows(
  df_results,
  data.frame(Method = "Movie, User, Genre, Year Effect", RMSE= as.character(round(model_4_rmse,5))
             ,Difference =as.character(round(model_4_rmse-objective_rmse,5)))
)

df_results %>% knitr::kable()

# Testing the date review effect

dateReview_avg <- train_set %>%
  left_join(movie_avg, by = "movieId") %>%
  left_join(user_avg, by = "userId") %>%
  left_join(genre_avg, by = "genres")%>%
  left_join(year_avg, by = "year")%>%
  mutate(dateReview=round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(dateReview)%>%
  summarize(b_r= mean(rating - mu_hat - b_i - b_u - b_g - b_y))

predicted_ratings<-test_set %>%
  mutate(dateReview=round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(movie_avg, by="movieId") %>%
  left_join(user_avg, by = "userId") %>%
  left_join(genre_avg, by = "genres") %>%
  left_join(year_avg, by = "year")%>%
  left_join(dateReview_avg, by = "dateReview")%>%
  mutate(pred=mu_hat+b_u+b_i+b_g+b_y+b_r) %>%
  pull(pred)

model_5_rmse<-RMSE(predicted_ratings, test_set$rating)

#Add the RMSE to result table
df_results<-bind_rows(
  df_results,
  data.frame(Method = "Movie, User, Genre, Year, Date Review Effect", RMSE= as.character(round(model_5_rmse,5))
             ,Difference =as.character(round(model_5_rmse-objective_rmse,5)))
)

df_results %>% knitr::kable()

#Generate a sequence for Lambda

incremental<-0.1
lambdas <- seq(0,10,incremental)
#remove(b_i, b_u)
rmses<-sapply(lambdas, function(l){
  
  b_i<-train_set %>% 
    group_by(movieId) %>%
    summarise(b_i= sum(rating - mu_hat)/(n()+l))
  
  b_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u= sum(rating - mu_hat- b_i)/(n()+l))
  
  b_g <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g= sum(rating - mu_hat - b_i - b_u)/(n()+l))
  
  b_y <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres")%>%
    group_by(year) %>%
    summarize(b_y= mean(rating - mu_hat - b_i - b_u - b_g)/(n()+l))
  
  b_r <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres")%>%
    left_join(b_y, by = "year")%>%
    mutate(dateReview=round_date(as_datetime(timestamp), unit = "week")) %>%
    group_by(dateReview)%>%
    summarize(b_r= mean(rating - mu_hat - b_i - b_u - b_g - b_y)/(n()+l))
  
  predicted_ratings<-test_set %>%
    mutate(dateReview = round_date(as_datetime(timestamp), unit = "week")) %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_y, by = "year")%>%
    left_join(b_r, by = "dateReview")%>%
    mutate(pred=mu_hat+b_u+b_i+b_g+b_y+b_r) %>%
    pull(pred)
  return(RMSE(predicted_ratings,test_set$rating))
  
})

#Inspect the result to select best parameters
qplot(lambdas, rmses, geom = "line")


#Optimal tuning parameter
lambda<-lambdas[which.min(rmses)]

#Minimum RMSE reached
regularized_rmse<-min(rmses)


#Add the regularized results
df_results<-bind_rows(
  df_results,
  data.frame(Method = "Regularized Movie, User, Genre, Year, Date Review Effect", RMSE= as.character(round(regularized_rmse,5))
             ,Difference =as.character(round(regularized_rmse-objective_rmse,5)))
)

df_results %>% knitr::kable()


###########################################################################################################################
# Final model
###########################################################################################################################

#Adding the "dateReview" as a column to improve performance

edx<-edx %>% 
  mutate(dateReview = round_date(as_datetime(timestamp), unit = "week"))

test_set<-test_set %>%
  mutate(dateReview = round_date(as_datetime(timestamp), unit = "week")) 

b_i<-edx %>% 
  group_by(movieId) %>%
  summarise(b_i= sum(rating - mu_hat)/(n()+lambda))

b_u <- edx %>%
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u= sum(rating - mu_hat- b_i)/(n()+lambda))

b_g <- edx %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g= sum(rating - mu_hat - b_i - b_u)/(n()+lambda))

b_y <- edx %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres")%>%
  group_by(year) %>%
  summarize(b_y= mean(rating - mu_hat - b_i - b_u - b_g)/(n()+lambda))

b_r <- edx %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres")%>%
  left_join(b_y, by = "year")%>%
  group_by(dateReview)%>%
  summarize(b_r= mean(rating - mu_hat - b_i - b_u - b_g - b_y)/(n()+lambda))

predicted_ratings<-test_set %>%
  mutate(dateReview = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_y, by = "year")%>%
  left_join(b_r, by = "dateReview")%>%
  mutate(pred=mu_hat+b_u+b_i+b_g+b_y+b_r) %>%
  pull(pred)

#Calculate final RMSE
final_rmse<-RMSE(test_set$rating, predicted_ratings)

#Objective versus result
df_final_model <- bind_rows(
  data.frame(Method = "Objective", RMSE = as.character(objective_rmse), Difference = "-"),
  data.frame(Method = "Final RMSE", RMSE= as.character(round(final_rmse,5))
             ,Difference =as.character(round(final_rmse-objective_rmse,5)))
  )

df_final_model %>% knitr::kable()

