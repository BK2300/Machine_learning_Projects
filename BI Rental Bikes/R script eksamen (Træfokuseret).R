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
ggcorrplot(cor_matrix)


#########################################################################


## 📌 Indlæs nødvendige pakker
library(rpart)
library(rpart.plot)
library(DescTools)  # RMSE-beregning
library(dplyr)
library(plotmo)     # Partial Dependence Plots
library(ipred)      # Bagging
library(randomForest) # Random Forest
library(caret)      # Optimering af hyperparametre

## 📌 Opdel data i træning og test (80% træning, 20% test)
set.seed(123)
trainIndex <- createDataPartition(rental_data$fraction, p = 0.8, list = FALSE)
train_data <- rental_data[trainIndex, ]
test_data  <- rental_data[-trainIndex, ]

## 📌 Fjern NA-værdier fra begge datasæt
train_data <- na.omit(train_data)
test_data  <- na.omit(test_data)  # Vigtigt at gøre det for begge!

## 📌 Konverter kategoriske variabler til faktorer (for begge datasæt)
train_data <- train_data %>%
  mutate(across(c(season, weekday, holiday, weather), as.factor))

test_data <- test_data %>%
  mutate(across(c(season, weekday, holiday, weather), as.factor))

## 🔥 Beslutningstræ uden krydsvalidering 🔥
tree_model <- rpart(fraction ~ temperature + feelslike + humidity + windspeed + 
                      season + weekday + holiday + weather + hour,
                    data = train_data, method = "anova", 
                    control = list(cp = 0.01, maxdepth = 5))  

# 📌 Find den optimale CP-værdi og beskær træet
optimal_cp <- tree_model$cptable[which.min(tree_model$cptable[, "xerror"]), "CP"]
pruned_tree <- prune(tree_model, cp = optimal_cp)

# 📌 Visualisér det beskårne træ
rpart.plot(pruned_tree, type = 3, extra = 101, under = TRUE, tweak = 1.2, main = "Beskåret Beslutningstræ")

# 📌 Partial Dependence Plots
plotmo(pruned_tree, pmethod = "partdep", degree1 = TRUE, main = "Partial Dependence Plots")

# 📌 Lav forudsigelser og beregn RMSE for beslutningstræet
predictions <- predict(pruned_tree, newdata = test_data)
rmse_tree <- sqrt(mean((predictions - test_data$fraction)^2, na.rm = TRUE))
print(paste("Beslutningstræ RMSE:", round(rmse_tree, 4)))

# 📌 Plot forudsigelser vs. faktiske værdier for beslutningstræ
plot(test_data$fraction, predictions, xlab = "Faktiske værdier", ylab = "Forudsagte værdier",
     main = "Beslutningstræ: Faktiske vs. Forudsagte værdier")
abline(0,1, col="red")


## 🔥 Bagging af beslutningstræer 🔥
set.seed(123)
bagged_tree_model <- bagging(fraction ~ temperature + feelslike + humidity + windspeed +
                               season + weekday + holiday + weather + hour,
                             data = train_data, nbagg = 50) 

# 📌 Lav forudsigelser og beregn RMSE for bagging
bagging_predictions <- predict(bagged_tree_model, newdata = test_data)
rmse_bagging <- sqrt(mean((bagging_predictions - test_data$fraction)^2))
print(paste("Bagging RMSE:", round(rmse_bagging, 4)))

# 📌 Plot forudsigelser vs. faktiske værdier for bagging
plot(test_data$fraction, bagging_predictions, xlab = "Faktiske værdier", ylab = "Forudsagte værdier",
     main = "Bagging: Faktiske vs. Forudsagte værdier")
abline(0,1, col="red")


## 🔥 Random Forest Model 🔥
set.seed(123)
rf_model <- randomForest(fraction ~ temperature + feelslike + humidity + windspeed +
                           season + weekday + holiday + weather + hour,
                         data = train_data, ntree = 100, mtry = 3, importance = TRUE)

# 📌 Lav forudsigelser og beregn RMSE for Random Forest
rf_predictions <- predict(rf_model, newdata = test_data)
rmse_rf <- sqrt(mean((rf_predictions - test_data$fraction)^2))
print(paste("Random Forest RMSE:", round(rmse_rf, 4))) #(MEGET LANGSOM, TAG EN PAUSE PÅ 1MIN!)

# 📌 Plot forudsigelser vs. faktiske værdier for Random Forest
plot(test_data$fraction, rf_predictions, xlab = "Faktiske værdier", ylab = "Forudsagte værdier",
     main = "Random Forest: Faktiske vs. Forudsagte værdier")
abline(0,1, col="red")

# 📌 Visualisér vigtighed af variabler
varImpPlot(rf_model)

# 📌 Sammenlign RMSE for alle modeller
rmse_results <- data.frame(
  Model = c("Beslutningstræ", "Bagging", "Random Forest"),
  RMSE = c(rmse_tree, rmse_bagging, rmse_rf)
)

print(rmse_results)  # Sammenligning af modelpræcision


###################################################################
                # MSE

# Beregn Mean Squared Error (MSE) for beslutningstræet
mse_tree <- mean((predictions - test_data$fraction)^2)

# Beregn MSE for Bagging
mse_bagging <- mean((bagging_predictions - test_data$fraction)^2)

# Beregn MSE for Random Forest
mse_rf <- mean((rf_predictions - test_data$fraction)^2)

# Print alle MSE-værdierne
print(paste("MSE for beslutningstræ:", round(mse_tree, 4)))
print(paste("MSE for Bagging:", round(mse_bagging, 4)))
print(paste("MSE for Random Forest:", round(mse_rf, 4)))

# Saml resultaterne i en tabel for bedre overskuelighed
mse_results <- data.frame(
  Model = c("Beslutningstræ", "Bagging", "Random Forest"),
  MSE = c(mse_tree, mse_bagging, mse_rf)
)

# Vis resultaterne i en tabel
print(mse_results)


###############################################################
                   #feature importance

# Feature Importance for Random Forest
rf_importance <- as.data.frame(importance(rf_model))
rf_importance$Variable <- rownames(rf_importance) # Tilføj variabelnavne
rf_importance <- rf_importance %>% arrange(desc(IncNodePurity)) # Sorter efter betydning
print("Feature Importance for Random Forest:")
print(rf_importance)

# Visualiser Feature Importance for Random Forest
library(ggplot2)
ggplot(rf_importance, aes(x = reorder(Variable, IncNodePurity), y = IncNodePurity)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() + 
  theme_minimal() +
  labs(title = "Feature Importance for Random Forest",
       x = "Variabler", y = "Betydning (IncNodePurity)")
######################################################################################
                                     #MAE 
# Load nødvendig pakke
library(Metrics)  # Indeholder mae()-funktionen

# Beregn MAE for beslutningstræet
mae_tree <- mae(test_data$fraction, predictions)
print(paste("MAE for Beslutningstræ:", round(mae_tree, 4)))

# Beregn MAE for Bagging
mae_bagging <- mae(test_data$fraction, bagging_predictions)
print(paste("MAE for Bagging:", round(mae_bagging, 4)))

# Beregn MAE for Random Forest
mae_rf <- mae(test_data$fraction, rf_predictions)
print(paste("MAE for Random Forest:", round(mae_rf, 4)))

# Saml resultaterne i en tabel
mae_results <- data.frame(
  Model = c("Beslutningstræ", "Bagging", "Random Forest"),
  MAE = c(mae_tree, mae_bagging, mae_rf)
)

# Vis resultaterne
print(mae_results)

































