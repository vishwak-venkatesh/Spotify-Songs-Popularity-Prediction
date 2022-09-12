###### SPOTIFY PROJECT R SCRIPT

# Andrew Gillock
# Snehal Naravane
# Vishwak Venkatesh
# Boran Sheu
# Noah Shimizu
# Muskan Agarwal

rm(list = ls())

## DATA PREPARATION
library(dplyr)
library(stringr)
library(ggplot2)
library(caret)
library(glmnet)
library(lattice)
library(tree)
library(gbm)
library(randomForest)


spotify_data <- read.csv('Spotify-2000.csv')
spotify_data <- subset(spotify_data, select = -c(Index, Title))
names(spotify_data)

# Rename cols for easier usage
names(spotify_data)[names(spotify_data) == 'Top.Genre'] <- 'Genre'
names(spotify_data)[names(spotify_data) == 'Beats.Per.Minute..BPM.'] <- 'BPM'
names(spotify_data)[names(spotify_data) == 'Loudness..dB.'] <- 'Loudness'
names(spotify_data)[names(spotify_data) == 'Length..Duration.'] <- 'Length'

# Extract last word of genre and group
spotify_data['genre_grouped'] <- (word(spotify_data$Genre, -1))
genre_cnts <- spotify_data %>% 
  group_by(genre_grouped) %>% 
  summarise(cnt = n()/nrow(spotify_data)*100) %>% 
  arrange(desc(cnt))

# Convert duration to numeric (remove commas before)
spotify_data$Length <- gsub(",","",spotify_data$Length)
spotify_data <- transform(spotify_data, Length = as.numeric(Length))

# Dropping genres with less than 2% songs
low_cnts_genre <- genre_cnts[genre_cnts$cnt < 2,]$genre_grouped 
spotify_data[spotify_data$genre_grouped %in% low_cnts_genre,]$genre_grouped <- 'Others'

# Drop genre and artist
spotify_data <- subset(spotify_data, select = -c(Genre))

spotify_data <- subset(spotify_data, select = -c(Artist))

##########
##SLIDE 6
##########

#get summary statistics
library(psych) #for describe()
spotify_data %>% select(-Index) %>% 
  select_if(is.numeric) %>% 
  describe() %>% 
  select(-skew, -trimmed, -n, -vars, -kurtosis, -mad)

##########
##SLIDE 7
##########

#create popularity hist
spotify_data %>% ggplot(aes(x = Popularity)) + 
  geom_histogram(bins = 20, color = 'black', fill = '#1E90FF') + 
  ggtitle('Popularity Score Histogram') + 
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5)) #move title to center

##########
##SLIDE 8
##########

#create correlation heatmap
library(reshape2)
cor <- spotify_data %>% select_if(is.numeric) %>% cor() %>% round(2)

#get upper triangle of heatmap
get_upper_tri <- function(cor){
  cor[lower.tri(cor)]<- NA
  return(cor)
}

#now we make it look fancy
upper_tri <- get_upper_tri(cor)

new_cor <- melt(upper_tri, na.rm = TRUE)
ggplot(new_cor, aes(Var2, Var1, fill = value)) + 
  geom_tile() + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 70, vjust = 0.65), 
        legend.title = element_blank(), 
        plot.title = element_text(hjust = 0.5)
        ) + 
  xlab(NULL) + ylab(NULL) +
  geom_text(aes(Var2, Var1, label = value), color = "white", size = 3) +
  ggtitle('Correlation Heatmap')

half_cor <- melt(cor)
head(half_cor)

##########
##SLIDE 9
##########

##plot predictor relationships
#loudness v energy
spotify_data %>% ggplot(aes(x = Loudness, y = Energy)) + 
  geom_point() + geom_smooth(se = FALSE) + theme_minimal()

#valence v danceability
spotify_data %>% ggplot(aes(x = Valence, y = Danceability)) + 
  geom_point() + geom_smooth(se = FALSE) + theme_minimal()

#acousticness v energy
spotify_data %>% ggplot(aes(x = Acousticness, y = Energy)) + 
  geom_point() + geom_smooth(se = FALSE) + theme_minimal()

##########
##SLIDE 10
##########

#plot average song popularity after grouping by year of release
spotify_data %>% 
  group_by(Year) %>% #group all songs by year
  summarize(mean_pop = mean(Popularity)) %>% #take mean popularity of each year
  ggplot(aes(x = Year, y = mean_pop)) + 
  geom_point() + 
  theme_minimal() + 
  xlab('Year') + ylab('Mean Popularity') + 
  ggtitle('Average Song Popularity by Year of Release') +
  theme(plot.title = element_text(hjust = 0.5)) #move title to center of plot

##########
##SLIDE 11
##########

spotify_data2 <- read.csv('Spotify-2000.csv')

## USING ORIGINAL DATASET FOR COMPARISON, repeat initial wrangling
# Rename cols for easier usage
names(spotify_data2)[names(spotify_data2) == 'Top.Genre'] <- 'Genre'
names(spotify_data2)[names(spotify_data2) == 'Beats.Per.Minute..BPM.'] <- 'BPM'
names(spotify_data2)[names(spotify_data2) == 'Loudness..dB.'] <- 'Loudness'
names(spotify_data2)[names(spotify_data2) == 'Length..Duration.'] <- 'Length'

#get frequencies of artists
artist_table <- spotify_data2 %>% group_by(Artist) %>% 
  summarize(Percentage = n()/nrow(.)) %>% 
  arrange(desc(Percentage))
head(artist_table, 10)

#get frequences of genres (prior to grouping)
genre_table <- spotify_data2 %>% group_by(Genre) %>% 
  summarize(Percentage = n()/nrow(.)) %>% 
  arrange(desc(Percentage))
head(genre_table, 20)

##########
##SLIDE 13
##########

#get counts for each genre
spotify_data2['genre_grouped'] <- (word(spotify_data2$Genre, -1))
genre_cnts <- spotify_data2 %>% 
  group_by(genre_grouped) %>% 
  summarise(cnt = n()/nrow(spotify_data)*100) %>% 
  arrange(desc(cnt))

#create bar chart of genre frequency
library(forcats)
spotify_data2 %>% mutate(
  genre = fct_lump_n(fct_infreq(genre_grouped), 5), #get top 5 genres
  highlight = fct_other(
    genre,
    keep = 'Other', #other genres go in 'Other' category
    other_level = 'Named'
  )
) %>% ggplot() +
  aes(y = fct_rev(genre), #descending order
      fill = fct_rev(highlight), 
      label = ..count..) + 
  geom_bar() +
  scale_x_continuous(name = 'Number of Songs') + 
  scale_y_discrete(name = NULL) +
  scale_fill_manual(
    values = c(Named = 'gray50', Other = '#98545F'), guide = 'none'
  ) + ggtitle('Frequency of Genre in Top 2000 Songs') +
  theme_minimal()

##########
##SLIDE 14
##########

#create scatterplot
spotify_data %>% 
  filter(genre_grouped %in% c('metal', 'pop')) %>% #filter for only metal and pop
  ggplot(aes(
    x = Valence, y = Energy, 
    color = genre_grouped) #color by genre
    ) + geom_point() + 
  ggtitle('Scatterplot of Valence & Energy Scores for Metal & Pop Songs') + 
  theme_minimal() + 
  theme(legend.title = element_blank(), legend.position = 'top') + #remove legend title, move legend to top
  theme(plot.title = element_text(hjust = 0.5)) #move title to center

#create boxplot
spotify_data %>% 
  filter(genre_grouped %in% c('rock', 'standards')) %>% #filter for only rock and standards
  ggplot(aes(
    x = genre_grouped, y = Energy, 
    fill = genre_grouped)
    ) + geom_boxplot() +
  ggtitle('Boxplot of Energy Scores for Rock & Standards Songs') +
  theme_minimal() + 
  theme(legend.position = 'none') + #remove legend 
  scale_x_discrete(name = NULL) + #remove x-axis label
  theme(plot.title = element_text(hjust = 0.5)) #move title to center

##########
##SLIDE 15
##########

#plot mean popularity of rock music
spotify_data %>% 
  filter(genre_grouped == 'rock') %>% #filter for only rock genre
  group_by(Year) %>% 
  summarize(mean_popularity = mean(Popularity)) %>% #get mean popularity
  ggplot(aes(x = Year, y = mean_popularity)) + 
  geom_point() + geom_smooth(method = 'lm', se = FALSE) + 
  theme_minimal() + ggtitle('Popularity Trend of Rock') +
  ylab('Mean Popularity') +
  theme(plot.title = element_text(hjust = 0.5)) #move title to center


##########
##SLIDE 17 Table
##########
head(spotify_data)
spotify_data1 <- subset(spotify_data, select = -c(genre_grouped))
# Training & test data
set.seed(1)
split <- sample(seq_len(nrow(spotify_data1)), size = floor(0.75*nrow(spotify_data1)) )
train <- spotify_data1[split,]
test <- spotify_data1[-split,]

#### Single Predictors ####

# Initializing the regression results lists
train_reg_1_summ = list()
train_reg_1_r2 = list()
train_reg_1_adjr2 = list()
train_reg_1_stderr = list()
train_reg_1_sigma = list()
train_reg_1_coeff = list()
train_reg_1_stderr = list()
train_reg_1_tval = list()
train_reg_1_pval = list()
train_reg_1_RMSE = list()

test_predict_1 =  list()

# Number of rows
spot_rows = dim(spotify_data1)[1]
# Number of columns
spot_cols = dim(spotify_data1)[2]

# Simple Linear Regression
for (colns in (1:(spot_cols-1))){
  train_reg_1_1 = lm(paste('train$Popularity~train$',colnames(train[colns])))
  test_predict_1[length(test_predict_1) + 1] = (predict(train_reg_1_1, test))
  train_reg_1_summ[[length(train_reg_1_summ) + 1]] = summary(train_reg_1_1)
  train_reg_1_r2[[length(train_reg_1_r2) + 1]] = summary(train_reg_1_1)$r.squared
  train_reg_1_adjr2[[length(train_reg_1_adjr2) + 1]] = summary(train_reg_1_1)$adj.r.squared
  train_reg_1_sigma[[length(train_reg_1_sigma) + 1]] = summary(train_reg_1_1)$sigma
  train_reg_1_coeff[[length(train_reg_1_coeff) + 1]] = summary(train_reg_1_1)$coefficients
  train_reg_1_stderr[[length(train_reg_1_stderr) + 1]] = summary(train_reg_1_1)$coefficients[,2]
  train_reg_1_tval[[length(train_reg_1_tval) + 1]] = summary(train_reg_1_1)$coefficients[,3]
  train_reg_1_pval[[length(train_reg_1_pval) + 1]] = summary(train_reg_1_1)$coefficients[,4]
  res_len = length(train_reg_1_1$residuals)
  res_sum = 0
  for (iter in (1:res_len)){
    res_sq = train_reg_1_1$residuals[iter] ^ 2
    res_sum = res_sum + res_sq
  }
  res_mean = res_sum / res_len
  train_reg_1_RMSE[length(train_reg_1_RMSE) + 1] = sqrt(res_mean)
}

# train_reg_1_coeff

metrics_matrix <- matrix(data = NA, nrow = 6, ncol = 10)
colnames(metrics_matrix) = names(train)[1:(spot_cols-1)]
rownames(metrics_matrix) = c('Estimate','Std Error', 't-value', 'p-value', 'R squared', 'Adj R squared')

metrics_matrix
summ_list <- list(train_reg_1_coeff, train_reg_1_stderr, train_reg_1_tval, train_reg_1_pval, train_reg_1_r2, train_reg_1_adjr2)

for(i in 1:nrow(metrics_matrix)){
  for(j in 1:(spot_cols -1)){
    if(i<=4){
      metrics_matrix[i,j] = round(summ_list[i][[1]][[j]][2],4)
    } else {
      metrics_matrix[i,j] = round(summ_list[i][[1]][[j]][1],4)
    }
  }
}
metrics_matrix

##########
##SLIDE 18 
##########
train <- spotify_data[split,]
test <- spotify_data[-split,]

lin_reg <- lm(Popularity ~ ., data = train)
summary(lin_reg)
# Check RMSE with test data
predictions <- predict(lin_reg, newdata = test) # Change model name
rmse <- sqrt(mean((test$Popularity - predictions)^2))
rmse

##########
##SLIDE 21 
##########

# Create dummy variables (for genre_grouped)
spotify_data <- data.frame(spotify_data[,1:(length(spotify_data)-2)],
                           model.matrix(~spotify_data$genre_grouped)[,-1],spotify_data$Popularity) 
names(spotify_data)[names(spotify_data) == 'spotify_data.Popularity'] <- 'Popularity'

#Interactions with all columns
spotify_data <- data.frame(model.matrix(~.^2, spotify_data[,1:length(spotify_data)-1])[,-1], spotify_data$Popularity)
names(spotify_data)[names(spotify_data) == 'spotify_data.Popularity'] <- 'Popularity'

train <- spotify_data[split,]
test <- spotify_data[-split,]

# Forward/Backward/Step regression
null = lm(Popularity~1, data=train)
full = lm(Popularity~., data=train)

# BEST MODEL
regBack = step(full, direction="backward", k=log(length(train)))
summary(regBack)
# Check RMSE with test data
predictions <- predict(regBack, newdata = test) # Change model name
rmse <- sqrt(mean((test$Popularity - predictions)^2))
rmse

# Residual plots
plot(fitted(regBack), resid(regBack), xlab = "Fitted Values", ylab = "Residuals", main = "Residuals vs Fitted Values")
abline(0,0)

##########
##SLIDE 22 
##########

# Evaluation Function
# Compute R^2 from true and predicted values
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2) 
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  ) 
}

# Process data -- scaling
y_train <- train$Popularity
X_train <- data.matrix(subset(train, select = -c(Popularity)))
y_test <- test$Popularity
X_test <- data.matrix(subset(test, select = -c(Popularity)))
normParam <- preProcess(X_train) #scaling x_train. safe mean and variance in nomParam.
X_train <- predict(normParam, X_train)
X_test <- predict(normParam, X_test)
# Lasso Regression
cv_model <- cv.glmnet(X_train, y_train, nfolds = 10, alpha = 1) #nfolds=number of time cross-validation. alpah=lasso
#find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda
best_lasso <- glmnet(X_train, y_train, alpha = 1, lambda = best_lambda) #train model
coef(best_lasso)
# Get Evaluation -- Train
predictions_train <- predict(best_lasso, s = best_lambda, newx = X_train)
eval_results(y_train, predictions_train, train)
# Get Evaluation -- Test
predictions_test <- predict(best_lasso, s = best_lambda, newx = X_test)
eval_results(y_test, predictions_test, test)

# Ridge Regression
cv_model <- cv.glmnet(X_train, y_train, nfolds = 10, alpha = 0)
#find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda
# Plot MSE-lambda relation
#plot(cv_model)
# Construct the model
best_ridge <- glmnet(X_train, y_train, alpha = 0, lambda = best_lambda)
coef(best_ridge)
# Get Evaluation -- Train
predictions_train <- predict(best_ridge, s = best_lambda, newx = X_train)
eval_results(y_train, predictions_train, train)
# Get Evaluation -- Test
predictions_test <- predict(best_ridge, s = best_lambda, newx = X_test)
eval_results(y_test, predictions_test, test)

###Plot
Lasso.Fit = glmnet(X_train,y_train)
Ridge.Fit = glmnet(X_train,y_train,alpha=0)

CV.L = cv.glmnet(X_train, y_train,alpha=1)
CV.R = cv.glmnet(X_train, y_train,alpha=0)

LamR = CV.R$lambda.1se
LamL = CV.L$lambda.1se


par(mfrow=c(1,2))
plot(log(CV.R$lambda),sqrt(CV.R$cvm),main="Ridge CV (k=10)",xlab="log(lambda)",ylab = "RMSE",col=4,type="b",cex.lab=1.2)
abline(v=log(LamR),lty=2,col=2,lwd=2)
plot(log(CV.L$lambda),sqrt(CV.L$cvm),main="LASSO CV (k=10)",xlab="log(lambda)",ylab = "RMSE",col=4,type="b",cex.lab=1.2)
abline(v=log(LamL),lty=2,col=2,lwd=2)

## Bagging, Random Forest and Boosting ##

# Training & test data
set.seed(1)
spotify_data$Popularity <- as.numeric(spotify_data$Popularity)
split <- sample(seq_len(nrow(spotify_data)), size = floor(0.75*nrow(spotify_data)) )
spotify_train <- spotify_data[split,]
spotify_test <- spotify_data[-split,]

spotify_test$genre_grouped <- as.factor(spotify_test$genre_grouped)
spotify_train$genre_grouped <- as.factor(spotify_train$genre_grouped)


####    Random Forest    ####

rf.spotify <- randomForest(Popularity ~ ., spotify_train, mtry = 5, importance = TRUE, ntree = 1000)

yhat.bag <- predict(rf.spotify, newdata = spotify_test)
sqrt(mean((yhat.bag - spotify_test$Popularity)^2))

importance(rf.spotify)
varImpPlot(rf.spotify)  

####    Bagging   ####

set.seed(1)
bag.spot <- randomForest(
  Popularity ~ . , data = spotify_train,
  mtry = 10, importance= TRUE,
  ntree = 900
)

predicted.bag <- predict(bag.spot, newdata = spotify_test)
error <- mean((predicted.bag - spotify_test$Popularity)^2)
sqrt(error) 

importance(bag.spot)
varImpPlot(bag.spot)  


####    Boosting    ####

boost.spotify_data <- gbm(Popularity ~ ., spotify_train,
                          distribution = "gaussian", n.trees = 40,
                          interaction.depth = 4, shrinkage = 0.2, verbose = F)
###
summary(boost.spotify_data)
###
###
yhat.boost <- predict(boost.spotify_data,
                      newdata = spotify_test, n.trees = 40)
rmse <- sqrt(mean((yhat.boost - spotify_test$Popularity)^2))
rmse



