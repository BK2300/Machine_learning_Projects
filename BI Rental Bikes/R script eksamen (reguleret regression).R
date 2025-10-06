# Load pakkerne
library(tidyverse)   # Data wrangling, filtrering, visualisering
library(lubridate)   # Håndtering af datoer og tid
library(rpart)       # Beslutningstræer
library(rpart.plot)  # Visualisering af beslutningstræer
library(caret)       # Dataopdeling, modeltræning og evaluering
library(ggplot2)     # Ekstra datavisualisering

# Indlæs datasættet
rental_data <- read_csv(file.choose())

# Tjek de første rækker af data
head(rental_data)

# Konverter tekstvariabler til faktorer
rental_data <- rental_data %>%
  mutate(
    season = as.factor(season),
    month = as.factor(month),
    weekday = as.factor(weekday),
    holiday = as.factor(holiday),
    weather = as.factor(weather)
  )

# Kør summary igen for at se de opdaterede resultater
summary(rental_data)


#############################################################
                        #Overblik
#############################################################


# Smelt data til lang format for at kunne lave ét samlet boxplot
rental_data_long <- rental_data %>%
  select(fraction, temperature, feelslike, windspeed, humidity) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")

# Lav ét samlet boxplot
ggplot(rental_data_long, aes(x = Variable, y = Value, fill = Variable)) + 
  geom_boxplot() +
  theme_minimal() +
  ggtitle("Boxplots af vigtige numeriske variabler") +
  theme(legend.position = "none")  # Skjul legend for et renere plot

ggplot(rental_data, aes(y = fraction)) + 
  geom_boxplot(fill = "red") + 
  theme_minimal() +
  ggtitle("Boxplot af fraction (andel af lejlighedsvis brugere)")


# Lav histogrammer for de vigtigste numeriske variabler
ggplot(rental_data_long, aes(x = Value, fill = Variable)) + 
  geom_histogram(bins = 30, alpha = 0.6, position = "identity") +
  facet_wrap(~Variable, scales = "free") +
  theme_minimal() +
  ggtitle("Histogrammer af vigtige numeriske variabler")


# Smelt data for scatterplots (beholder fraction som y)
scatter_data_long <- rental_data %>%
  select(fraction, temperature, feelslike, windspeed, humidity) %>%
  pivot_longer(cols = -fraction, names_to = "Variable", values_to = "Value")

# Scatterplot af fraction mod de andre variabler
ggplot(scatter_data_long, aes(x = Value, y = fraction, color = Variable)) + 
  geom_point(alpha = 0.5) +
  facet_wrap(~Variable, scales = "free_x") +
  theme_minimal() +
  ggtitle("Scatterplots: Sammenhæng mellem fraction og vigtige variabler")


# Beregn korrelationen for alle numeriske variabler
cor_matrix <- cor(rental_data %>% select_if(is.numeric), use = "complete.obs")

# Vis korrelationsmatrix
print(cor_matrix)

library(ggcorrplot)

# Plot korrelationsmatrix som heatmap
ggcorrplot(cor_matrix, lab = TRUE, lab_size = 3)


library(car) # For Anova-funktion

# ANOVA-test for hver kategorisk variabel mod fraction
anova_results <- list()

categorical_vars <- c("season", "month", "weekday", "holiday", "weather")

for (var in categorical_vars) {
  model <- lm(fraction ~ get(var), data = rental_data)
  anova_results[[var]] <- Anova(model, type=2) # Type 2 ANOVA for mere robuste resultater
}

# Udskriv resultaterne
anova_results



#########################################################
                   #reguleret regression
#########################################################
# Load pakker
library(glmnet)       # Ridge og Lasso regression
library(caret)        # Modeltræning og evaluering
library(dplyr)        # Data manipulation
library(ggplot2)      # Visualisering
library(tidyverse)    # Samlet pakke til datahåndtering
library(ggcorrplot)   # Korrelationsplots

# Fjern kategoriske variabler (da Ridge/Lasso kun håndterer numeriske data)
numerical_data <- rental_data %>% 
  select(-season, -month, -weekday, -holiday, -weather)

# Opdel data i træning (80%) og test (20%)
set.seed(123)
trainIndex <- createDataPartition(numerical_data$fraction, p = 0.8, list = FALSE)
train_data <- numerical_data[trainIndex, ]
test_data  <- numerical_data[-trainIndex, ]

# Standardiser numeriske variable (vigtigt for Ridge/Lasso)
preprocess_params <- preProcess(train_data, method = c("center", "scale"))
train_scaled <- predict(preprocess_params, train_data)
test_scaled  <- predict(preprocess_params, test_data)

# Definer X (uafhængige variable) og y (afhængig variabel)
X_train <- as.matrix(train_scaled %>% select(-fraction))
y_train <- train_scaled$fraction

X_test <- as.matrix(test_scaled %>% select(-fraction))
y_test <- test_scaled$fraction

# Træn Ridge Regression model (alpha = 0 for Ridge)
set.seed(123)
ridge_model <- cv.glmnet(X_train, y_train, alpha = 0)

# Find den optimale lambda-værdi
best_lambda_ridge <- ridge_model$lambda.min
print(paste("Optimal lambda for Ridge:", round(best_lambda_ridge, 4)))

# Forudsigelser på testdata
ridge_predictions <- predict(ridge_model, s = best_lambda_ridge, newx = X_test)

# Beregn RMSE for Ridge
ridge_rmse <- sqrt(mean((ridge_predictions - y_test)^2))
print(paste("Ridge RMSE:", round(ridge_rmse, 4)))



# Træn Lasso Regression model (alpha = 1 for Lasso)
set.seed(123)
lasso_model <- cv.glmnet(X_train, y_train, alpha = 1)

# Find den optimale lambda-værdi
best_lambda_lasso <- lasso_model$lambda.min
print(paste("Optimal lambda for Lasso:", round(best_lambda_lasso, 4)))

# Forudsigelser på testdata
lasso_predictions <- predict(lasso_model, s = best_lambda_lasso, newx = X_test)

# Beregn RMSE for Lasso
lasso_rmse <- sqrt(mean((lasso_predictions - y_test)^2))
print(paste("Lasso RMSE:", round(lasso_rmse, 4)))

# Sammenlign RMSE-værdierne
rmse_results <- data.frame(
  Model = c("Ridge Regression", "Lasso Regression"),
  RMSE = c(ridge_rmse, lasso_rmse)
)

print(rmse_results) # Vis resultaterne i en tabel



# Load nødvendige pakker
library(glmnet)
library(caret)

# Forbered data til Elastic Net
set.seed(123)
x <- model.matrix(fraction ~ temperature + feelslike + humidity + windspeed + season + 
                    weekday + holiday + weather + hour, rental_data)[, -1]  # Fjern intercept
y <- rental_data$fraction

# Opdel data i træning og test
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
x_train <- x[trainIndex, ]
x_test  <- x[-trainIndex, ]
y_train <- y[trainIndex]
y_test  <- y[-trainIndex]

# Definer lambda-værdier til Elastic Net
lambda_seq <- 10^seq(-4, 1, length = 100)  # Fra 0.0001 til 10

# Cross-validation for at finde optimal alpha og lambda
set.seed(123)
cv_elastic <- train(
  x = x_train, y = y_train,
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10  # Prøver forskellige alpha-værdier (blanding af Ridge og Lasso)
)

# Optimal lambda og alpha
best_lambda <- cv_elastic$bestTune$lambda
best_alpha <- cv_elastic$bestTune$alpha

print(paste("Optimal lambda for Elastic Net:", round(best_lambda, 4)))
print(paste("Optimal alpha for Elastic Net:", round(best_alpha, 2)))

# Træn den bedste Elastic Net-model
elastic_model <- glmnet(x_train, y_train, alpha = best_alpha, lambda = best_lambda)

# Forudsigelser på testdata
elastic_predictions <- predict(elastic_model, s = best_lambda, newx = x_test)

# Beregn RMSE
elastic_rmse <- sqrt(mean((elastic_predictions - y_test)^2))
print(paste("Elastic Net RMSE:", round(elastic_rmse, 4)))




# Lav forudsigelser med Elastic Net
elastic_predictions <- predict(elastic_model, newx = as.matrix(x_test), s = best_lambda)

# Konverter til numerisk format
elastic_predictions <- as.numeric(elastic_predictions)

# Plot forudsigelser vs. faktiske værdier
plot(y_test, elastic_predictions, 
     xlab = "Faktiske værdier", 
     ylab = "Forudsagte værdier",
     main = "Elastic Net: Faktiske vs. Forudsagte værdier",
     col = "black", pch = 16, cex = 0.6)

# Tilføj en rød linje for perfekt forudsigelse
abline(0, 1, col = "red", lwd = 2)

# Beregn MAE for Elastic Net
elastic_mae <- mean(abs(elastic_predictions - y_test))
print(paste("Elastic Net MAE:", round(elastic_mae, 4)))


# Ridge regression koefficientsti (alpha = 0)
plot(ridge_model, xvar = "lambda", label = TRUE, main = "Ridge Regression Koefficientsti")

# Lasso regression koefficientsti (alpha = 1)
plot(lasso_model, xvar = "lambda", label = TRUE, main = "Lasso Regression Koefficientsti")


# Hent koefficienter fra Lasso-model
lasso_coef <- coef(lasso_model, s = best_lambda_lasso)

# Udskriv variabler med koefficienter ≠ 0
important_features <- rownames(lasso_coef)[lasso_coef[,1] != 0]
print("Udvalgte features i Lasso:")
print(important_features)
















