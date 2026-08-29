# Project README: Electricity Demand Forecasting with LightGBM and LSTM

This notebook demonstrates a comprehensive predictive modeling workflow for electricity demand forecasting. The goal is to accurately predict future electricity demand using historical data and various features, comparing the performance and interpretability of two distinct machine learning models: LightGBM (a gradient boosting model) and LSTM (a recurrent neural network).

## Table of Contents

1.  **Setup and Data Loading:**
    *   **Purpose:** Install necessary Python libraries (e.g., LightGBM, SHAP, PyTorch, Pandas) and load the raw electricity demand dataset. The data is expected to be provided in a `.zip` file containing a CSV.
    *   **Key Actions:** Installs dependencies and reads the primary `COLUMBUS_COMBINED.csv` file into a Pandas DataFrame.

2.  **Feature Engineering:**
    *   **Purpose:** Transform raw time-series data into a rich set of features that can help the models learn complex patterns and relationships.
    *   **Key Actions:**
        *   **Timestamp Parsing:** Converts raw time strings into datetime objects and sets them as the DataFrame index.
        *   **Missing Value Handling:** Interpolates any missing demand values.
        *   **Calendar Features:** Extracts cyclical components like hour of day, day of week, month, and a flag for weekends.
        *   **Cyclical Trigonometric Transformations:** Applies sine and cosine transformations to 'Hour' and 'Month' to capture their periodic nature without imposing artificial order.
        *   **Autoregressive Lag Features:** Creates lagged demand values (e.g., previous hour, same hour yesterday, same hour last week) to capture temporal dependencies.
        *   **Rolling Statistics:** Computes rolling mean and standard deviation of demand over a 24-hour window.
        *   **Data Cleaning:** Removes initial rows with NaNs introduced by lag features.

3.  **Model Training (LightGBM):**
    *   **Purpose:** Train a LightGBM Regressor, a highly efficient gradient boosting framework, to predict electricity demand.
    *   **Key Actions:**
        *   **Data Split:** Divides the dataset into training and testing sets chronologically (80/20 split).
        *   **Model Initialization & Training:** Configures and trains a `LGBMRegressor` model with early stopping to prevent overfitting.
        *   **Prediction:** Generates demand predictions on the test set.

4.  **Model Training (LSTM):**
    *   **Purpose:** Train a Long Short-Term Memory (LSTM) neural network, suitable for sequence data, to predict electricity demand.
    *   **Key Actions:**
        *   **Data Scaling:** Normalizes features and targets using `StandardScaler` for optimal neural network performance.
        *   **Sequence Generation:** Transforms the time-series data into sequences (e.g., 24-hour windows) for LSTM input.
        *   **PyTorch DataLoaders:** Prepares data for efficient batch training with PyTorch.
        *   **LSTM Model Definition:** Defines a custom LSTM neural network architecture.
        *   **Training Loop:** Trains the LSTM model using Adam optimizer and Mean Squared Error loss over multiple epochs.
        *   **Inference:** Generates demand predictions on the test set and inverse-scales them to the original units.

5.  **Model Evaluation and Results:**
    *   **Purpose:** Quantitatively assess and compare the performance of both the LightGBM and LSTM models using key regression metrics.
    *   **Key Actions:**
        *   **Metric Calculation:** Computes Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE) for both models.
        *   **Results Comparison:** Presents the evaluation metrics in a structured DataFrame to highlight the performance of each model. (e.g., *Initial results indicate LightGBM generally outperforms the current LSTM configuration in MAE and MAPE, suggesting better accuracy for this specific dataset and feature set*.)

6.  **Model Interpretability (SHAP):**
    *   **Purpose:** Use SHAP (SHapley Additive exPlanations) to understand how each feature contributes to the predictions of both models.
    *   **Key Actions:**
        *   **LightGBM SHAP:** Calculates SHAP values for the LightGBM model and visualizes global feature importance (beeswarm plot) and local explanations for peak hours (waterfall plot).
        *   **LSTM SHAP:** Adapts SHAP `KernelExplainer` for the LSTM model, including an inference wrapper and aggregation of SHAP values across time steps. Visualizes global and local explanations for the LSTM model similar to LightGBM.
