# Water Flow Time Series Analysis and Forecasting
# This script performs hourly aggregation, stationarity testing, and forecasting

# Load Required Libraries --------------------------------------------------
library(tidyverse)
library(readxl)
library(tsibble)
library(fable)
library(feasts)
library(imputeTS)
library(writexl)
library(lubridate)

# Data Import --------------------------------------------------------------
cat("Loading data...\n")
pipe1_raw <- read_excel("project_1/Waterflow_Pipe1.xlsx")
pipe2_raw <- read_excel("project_1/Waterflow_Pipe2.xlsx")

# Time-Based Hourly Aggregation --------------------------------------------
cat("Aggregating to hourly intervals...\n")

aggregate_hourly <- function(df, pipe_name) {
  df %>%
    rename(datetime = `Date Time`, waterflow = WaterFlow) %>%
    mutate(datetime = floor_date(datetime, unit = "hour")) %>%
    group_by(datetime) %>%
    summarise(waterflow = mean(waterflow, na.rm = TRUE), .groups = "drop") %>%
    mutate(pipe = pipe_name)
}

pipe1_hourly <- aggregate_hourly(pipe1_raw, "Pipe1")
pipe2_hourly <- aggregate_hourly(pipe2_raw, "Pipe2")

# Fill Missing Hours -------------------------------------------------------
cat("Filling missing hours and imputing values...\n")

create_complete_series <- function(df) {
  full_hours <- tibble(
    datetime = seq(
      from = floor_date(min(df$datetime), "hour"),
      to = floor_date(max(df$datetime), "hour"),
      by = "hour"
    )
  )
  
  df_complete <- full_hours %>%
    left_join(df, by = "datetime") %>%
    mutate(pipe = first(na.omit(df$pipe)))
  
  df_complete$waterflow <- na_interpolation(df_complete$waterflow, option = "linear")
  
  return(df_complete)
}

pipe1_complete <- create_complete_series(pipe1_hourly)
pipe2_complete <- create_complete_series(pipe2_hourly)

# Convert to tsibble -------------------------------------------------------
pipe1_ts <- pipe1_complete %>% as_tsibble(index = datetime)
pipe2_ts <- pipe2_complete %>% as_tsibble(index = datetime)

# Stationarity Testing -----------------------------------------------------
cat("Testing for stationarity...\n")

test_stationarity <- function(ts_data, pipe_name) {
  unit_root <- ts_data %>% features(waterflow, unitroot_kpss)
  
  cat("\n", pipe_name, "KPSS Test p-value:", unit_root$kpss_pvalue, "\n")
  
  if (unit_root$kpss_pvalue > 0.05) {
    cat("Result: Series is STATIONARY\n")
  } else {
    cat("Result: Series is NON-STATIONARY\n")
  }
  
  return(unit_root$kpss_pvalue > 0.05)
}

pipe1_stationary <- test_stationarity(pipe1_ts, "Pipe 1")
pipe2_stationary <- test_stationarity(pipe2_ts, "Pipe 2")

# Model Fitting ------------------------------------------------------------
cat("\nFitting forecasting models...\n")

pipe1_fit <- pipe1_ts %>%
  model(
    mean_model = MEAN(waterflow),
    naive = NAIVE(waterflow),
    snaive = SNAIVE(waterflow ~ lag("day")),
    drift = RW(waterflow ~ drift()),
    ets = ETS(waterflow),
    arima = ARIMA(waterflow)
  )

pipe2_fit <- pipe2_ts %>%
  model(
    mean_model = MEAN(waterflow),
    naive = NAIVE(waterflow),
    snaive = SNAIVE(waterflow ~ lag("day")),
    drift = RW(waterflow ~ drift()),
    ets = ETS(waterflow),
    arima = ARIMA(waterflow)
  )

# Model Accuracy -----------------------------------------------------------
cat("\nEvaluating model accuracy...\n")

pipe1_accuracy <- accuracy(pipe1_fit) %>% arrange(RMSE)
pipe2_accuracy <- accuracy(pipe2_fit) %>% arrange(RMSE)

cat("\nPipe 1 Model Accuracy:\n")
print(pipe1_accuracy)

cat("\nPipe 2 Model Accuracy:\n")
print(pipe2_accuracy)

best_pipe1 <- pipe1_accuracy %>% slice(1) %>% pull(.model)
best_pipe2 <- pipe2_accuracy %>% slice(1) %>% pull(.model)

cat("\nBest model for Pipe 1:", best_pipe1, "\n")
cat("Best model for Pipe 2:", best_pipe2, "\n")

# One Week Forecast --------------------------------------------------------
cat("\nGenerating one week (168 hours) forecast...\n")

h_forecast <- 168

pipe1_forecast <- pipe1_fit %>%
  select(!!best_pipe1) %>%
  forecast(h = h_forecast)

pipe2_forecast <- pipe2_fit %>%
  select(!!best_pipe2) %>%
  forecast(h = h_forecast)

# Prepare Export -----------------------------------------------------------
prepare_forecast_export <- function(forecast_obj, pipe_name) {
  forecast_obj %>%
    as_tibble() %>%
    select(datetime, .mean, waterflow) %>%
    rename(
      DateTime = datetime,
      Forecast = .mean,
      DistributionInfo = waterflow
    ) %>%
    mutate(
      Pipe = pipe_name,
      DistributionInfo = as.character(DistributionInfo)
    ) %>%
    select(Pipe, DateTime, Forecast, DistributionInfo)
}

pipe1_export <- prepare_forecast_export(pipe1_forecast, "Pipe1")
pipe2_export <- prepare_forecast_export(pipe2_forecast, "Pipe2")
combined_forecast <- bind_rows(pipe1_export, pipe2_export)

# Export to Excel ----------------------------------------------------------
cat("\nExporting results to Excel...\n")

write_xlsx(
  list(
    "Combined_Forecast" = combined_forecast,
    "Pipe1_Forecast" = pipe1_export,
    "Pipe2_Forecast" = pipe2_export,
    "Pipe1_Accuracy" = pipe1_accuracy,
    "Pipe2_Accuracy" = pipe2_accuracy,
    "Pipe1_Hourly_Data" = pipe1_ts,
    "Pipe2_Hourly_Data" = pipe2_ts
  ),
  path = "waterflow_forecast.xlsx"
)

cat("\n✓ Analysis complete! Results saved to: waterflow_forecast.xlsx\n")
cat("\nForecast Summary:\n")
cat("Pipe 1 - Mean:", round(mean(pipe1_export$Forecast), 2), 
    "| Range:", round(min(pipe1_export$Forecast), 2), "-", 
    round(max(pipe1_export$Forecast), 2), "\n")
cat("Pipe 2 - Mean:", round(mean(pipe2_export$Forecast), 2), 
    "| Range:", round(min(pipe2_export$Forecast), 2), "-", 
    round(max(pipe2_export$Forecast), 2), "\n")