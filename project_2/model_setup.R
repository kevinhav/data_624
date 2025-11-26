library(tidyverse)
library(randomForest)
library(rpart)
library(readxl)
library()


# ============================================================================
# ABC Beverage pH Prediction - Model Setup
# Comparing: Remove Missing Values vs KNN Imputation
# ============================================================================

# 1. LOAD AND EXPLORE DATA
# ============================================================================

df <- read_excel("data/train_data.xlsx")

cat("Original data dimensions:", nrow(df), "rows x", ncol(df), "columns\n")
cat("\nMissing values by column:\n")
missing_summary <- colSums(is.na(df))
print(missing_summary[missing_summary > 0])
cat("\nTotal missing values:", sum(is.na(df)), "\n")

# 2. CLEAN COLUMN NAMES
# ============================================================================

clean_names <- function(data) {
  data %>%
    rename_with(~str_replace_all(., " ", "_"))
}

df_cleaned_names <- df %>% clean_names()

# 3. APPROACH 1: REMOVE MISSING VALUES (ORIGINAL)
# ============================================================================

cat("\n========== APPROACH 1: REMOVE MISSING VALUES ==========\n")

df_removed <- df_cleaned_names %>%
  drop_na() %>%
  mutate(Brand_Code = as.numeric(as.factor(Brand_Code)))

cat("Data dimensions after removal:", nrow(df_removed), "rows x", ncol(df_removed), "columns\n")
cat("Rows removed:", nrow(df_cleaned_names) - nrow(df_removed), "\n")
cat("Data retained:", round(100 * nrow(df_removed) / nrow(df_cleaned_names), 1), "%\n")

# 4. APPROACH 2: KNN IMPUTATION
# ============================================================================

cat("\n========== APPROACH 2: KNN IMPUTATION (k=5) ==========\n")

# Convert Brand_Code to numeric before imputation
df_for_impute <- df_cleaned_names %>%
  mutate(Brand_Code = as.numeric(as.factor(Brand_Code)))

# Apply KNN imputation (k=5 nearest neighbors)
df_imputed <- kNN(df_for_impute, k = 5, imp_var = FALSE) %>%
  as_tibble()

cat("Data dimensions after imputation:", nrow(df_imputed), "rows x", ncol(df_imputed), "columns\n")
cat("All missing values filled with KNN neighbors\n")
cat("Data retained: 100%\n")

# 5. TARGET VARIABLE SUMMARY
# ============================================================================

cat("\nTarget Variable (PH) Summary:\n")
cat("  Removed approach:  n =", nrow(df_removed), "\n")
print(summary(df_removed$PH))

cat("\n  Imputed approach:  n =", nrow(df_imputed), "\n")
print(summary(df_imputed$PH))

# ============================================================================
# FUNCTION TO BUILD AND EVALUATE MODELS
# ============================================================================

build_models <- function(data, approach_name) {
  
  cat(sprintf("\n========== BUILDING MODELS: %s ==========\n", approach_name))
  
  # LINEAR REGRESSION
  cat("\n1. Linear Regression\n")
  lm_mod <- lm(PH ~ ., data = data)
  lm_pred <- predict(lm_mod, data)
  lm_rmse <- sqrt(mean((data$PH - lm_pred)^2))
  lm_mae <- mean(abs(data$PH - lm_pred))
  lm_r2 <- 1 - (sum((data$PH - lm_pred)^2) / sum((data$PH - mean(data$PH))^2))
  cat(sprintf("   RMSE: %.4f | MAE: %.4f | R²: %.4f\n", lm_rmse, lm_mae, lm_r2))
  
  # DECISION TREE
  cat("\n2. Decision Tree\n")
  dt_mod <- rpart(PH ~ ., data = data, method = "anova", cp = 0.01)
  dt_pred <- predict(dt_mod, data)
  dt_rmse <- sqrt(mean((data$PH - dt_pred)^2))
  dt_mae <- mean(abs(data$PH - dt_pred))
  dt_r2 <- 1 - (sum((data$PH - dt_pred)^2) / sum((data$PH - mean(data$PH))^2))
  cat(sprintf("   RMSE: %.4f | MAE: %.4f | R²: %.4f\n", dt_rmse, dt_mae, dt_r2))
  
  # RANDOM FOREST
  cat("\n3. Random Forest\n")
  rf_mod <- randomForest(PH ~ ., 
                         data = data,
                         ntree = 500,
                         importance = TRUE,
                         seed = 42)
  rf_pred <- predict(rf_mod, data)
  rf_rmse <- sqrt(mean((data$PH - rf_pred)^2))
  rf_mae <- mean(abs(data$PH - rf_pred))
  rf_r2 <- 1 - (sum((data$PH - rf_pred)^2) / sum((data$PH - mean(data$PH))^2))
  cat(sprintf("   RMSE: %.4f | MAE: %.4f | R²: %.4f\n", rf_rmse, rf_mae, rf_r2))
  
  # Feature importance
  rf_importance <- rf_mod$importance %>%
    as_tibble(rownames = "Feature") %>%
    pivot_longer(cols = -Feature, 
                 names_to = "Importance_Type", 
                 values_to = "Importance_Score") %>%
    filter(Importance_Type == "%IncMSE") %>%
    select(Feature, Importance_Score) %>%
    arrange(desc(Importance_Score))
  
  # Return results
  list(
    lm = list(model = lm_mod, rmse = lm_rmse, mae = lm_mae, r2 = lm_r2),
    dt = list(model = dt_mod, rmse = dt_rmse, mae = dt_mae, r2 = dt_r2),
    rf = list(model = rf_mod, rmse = rf_rmse, mae = rf_mae, r2 = rf_r2),
    importance = rf_importance,
    n_obs = nrow(data)
  )
}

# 6. BUILD MODELS FOR BOTH APPROACHES
# ============================================================================

results_removed <- build_models(df_removed, "REMOVED MISSING VALUES")
results_imputed <- build_models(df_imputed, "KNN IMPUTATION")

# 7. COMPARISON TABLE
# ============================================================================

cat("\n========== APPROACH COMPARISON ==========\n\n")

comparison_df <- tibble(
  Approach = c("Removed", "Removed", "Removed", 
               "KNN Imputed", "KNN Imputed", "KNN Imputed"),
  Model = c("Linear Regression", "Decision Tree", "Random Forest",
            "Linear Regression", "Decision Tree", "Random Forest"),
  N_Observations = c(results_removed$n_obs, results_removed$n_obs, results_removed$n_obs,
                     results_imputed$n_obs, results_imputed$n_obs, results_imputed$n_obs),
  RMSE = c(results_removed$lm$rmse, results_removed$dt$rmse, results_removed$rf$rmse,
           results_imputed$lm$rmse, results_imputed$dt$rmse, results_imputed$rf$rmse),
  MAE = c(results_removed$lm$mae, results_removed$dt$mae, results_removed$rf$mae,
          results_imputed$lm$mae, results_imputed$dt$mae, results_imputed$rf$mae),
  R_Squared = c(results_removed$lm$r2, results_removed$dt$r2, results_removed$rf$r2,
                results_imputed$lm$r2, results_imputed$dt$r2, results_imputed$rf$r2)
) %>%
  mutate(across(where(is.numeric) & !N_Observations, ~round(., 4)))

print(comparison_df)

# 8. BEST MODEL SUMMARY
# ============================================================================

cat("\n========== BEST MODELS BY APPROACH ==========\n")

best_removed <- comparison_df %>%
  filter(Approach == "Removed") %>%
  slice_max(R_Squared, n = 1) %>%
  select(Model, RMSE, MAE, R_Squared)

best_imputed <- comparison_df %>%
  filter(Approach == "KNN Imputed") %>%
  slice_max(R_Squared, n = 1) %>%
  select(Model, RMSE, MAE, R_Squared)

cat("\nBest Model (Removed Approach):\n")
print(best_removed)

cat("\nBest Model (KNN Imputation Approach):\n")
print(best_imputed)

# 9. VISUALIZATION - FEATURE IMPORTANCE COMPARISON
# ============================================================================

cat("\nGenerating feature importance plots...\n")

p_removed <- results_removed$importance %>%
  slice(1:15) %>%
  ggplot(aes(x = reorder(Feature, Importance_Score), y = Importance_Score)) +
  geom_col(fill = "#E63946", alpha = 0.8) +
  coord_flip() +
  labs(title = "Removed Missing Values",
       subtitle = "Top 15 Features - Mean Decrease in MSE",
       x = "Feature",
       y = "Importance Score") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 11))

p_imputed <- results_imputed$importance %>%
  slice(1:15) %>%
  ggplot(aes(x = reorder(Feature, Importance_Score), y = Importance_Score)) +
  geom_col(fill = "#2E86AB", alpha = 0.8) +
  coord_flip() +
  labs(title = "KNN Imputation",
       subtitle = "Top 15 Features - Mean Decrease in MSE",
       x = "Feature",
       y = "Importance Score") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 11))

p_combined <- gridExtra::grid.arrange(p_removed, p_imputed, ncol = 2)
ggsave("feature_importance_comparison.png", p_combined, width = 14, height = 8, dpi = 300)
cat("Saved: feature_importance_comparison.png\n")

# 10. SAVE MODELS AND KEY OUTPUTS
# ============================================================================

# Save all models
saveRDS(results_removed$lm$model, "lm_model_removed.rds")
saveRDS(results_removed$dt$model, "dt_model_removed.rds")
saveRDS(results_removed$rf$model, "rf_model_removed.rds")

saveRDS(results_imputed$lm$model, "lm_model_imputed.rds")
saveRDS(results_imputed$dt$model, "dt_model_imputed.rds")
saveRDS(results_imputed$rf$model, "rf_model_imputed.rds")

# Save datasets for export script
saveRDS(df_removed, "df_removed.rds")
saveRDS(df_imputed, "df_imputed.rds")

cat("\n========== FILES SAVED ==========\n")
cat("Models (Removed):\n")
cat("  - lm_model_removed.rds\n")
cat("  - dt_model_removed.rds\n")
cat("  - rf_model_removed.rds\n\n")
cat("Models (Imputed):\n")
cat("  - lm_model_imputed.rds\n")
cat("  - dt_model_imputed.rds\n")
cat("  - rf_model_imputed.rds\n\n")
cat("Datasets:\n")
cat("  - df_removed.rds\n")
cat("  - df_imputed.rds\n")

# 11. SUMMARY AND RECOMMENDATION
# ============================================================================

cat("\n========== SUMMARY AND RECOMMENDATION ==========\n")

removed_r2 <- comparison_df %>%
  filter(Approach == "Removed") %>%
  pull(R_Squared) %>%
  max()

imputed_r2 <- comparison_df %>%
  filter(Approach == "KNN Imputed") %>%
  pull(R_Squared) %>%
  max()

improvement <- ((imputed_r2 - removed_r2) / removed_r2) * 100

cat(sprintf("\nRemoved Approach Best R²: %.4f\n", removed_r2))
cat(sprintf("KNN Imputation Best R²: %.4f\n", imputed_r2))
cat(sprintf("Improvement: %.2f%%\n", improvement))

if (imputed_r2 > removed_r2) {
  cat(sprintf("\n✓ RECOMMENDATION: Use KNN IMPUTATION approach\n"))
  cat(sprintf("  - Retains all %d observations (+%.1f%% data)\n", 
              results_imputed$n_obs,
              100 * (results_imputed$n_obs - results_removed$n_obs) / results_removed$n_obs))
  cat(sprintf("  - Improves R² by %.2f%%\n", improvement))
  cat("  - Better feature coverage for model training\n")
} else {
  cat(sprintf("\n✓ RECOMMENDATION: Use REMOVED approach\n"))
  cat(sprintf("  - Simpler workflow with no imputation bias\n"))
  cat(sprintf("  - Working with %d complete observations\n", results_removed$n_obs))
}