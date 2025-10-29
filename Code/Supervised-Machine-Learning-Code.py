# Title: Supervised Machine Learning Module
# Author: Alexander Zakrzeski
# Date: October 28, 2025

# Load to import, clean, and wrangle data
import os
import pandas as pd
import polars as pl

# Load to analyze associations and run statistical tests
import statsmodels.api as sm 
from dython.nominal import associations
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols

# Load to train, test, and evaluate machine learning models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             ConfusionMatrixDisplay, f1_score, 
                             mean_absolute_error, precision_score, recall_score,
                             roc_auc_score, root_mean_squared_error)
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Set the working directory
os.chdir("/Users/atz5/Desktop/Machine-Learning-In-Python/Data")

# Part 1: K-Nearest Neighbors

# Section 1.1: Data Preprocessing

# Load the data from the CSV file, rename columns, and filter
hd1 = (
    pl.read_csv("Heart-Disease-1-Data.csv")
      .rename(str.lower)
      .rename({"chestpaintype": "chest_pain_type",
               "restingbp": "resting_bp",
               "fastingbs": "fasting_bs",
               "restingecg": "resting_ecg",
               "maxhr": "max_hr",
               "exerciseangina": "exercise_angina",
               "oldpeak": "old_peak",
               "heartdisease": "heart_disease"})
      .filter(pl.col("age").is_between(35, 75) & (pl.col("resting_bp") >= 80) &
              pl.col("cholesterol").is_between(100, 500) & 
              (pl.col("old_peak") >= 0))
    )

# Change data types, select columns, and convert to a Pandas DataFrame
hd1_cat = (
    hd1.with_columns([
        pl.col(c).cast(pl.Utf8).alias(c) 
        for c in ["fasting_bs", "heart_disease"]
        ])
       .select(pl.col(pl.Utf8))
       .to_pandas()
    )

# Select columns
hd1_num = hd1.select("age", "resting_bp", "cholesterol", "max_hr", "old_peak", 
                     "heart_disease")

# Section 1.2: Exploratory Data Analysis

# For each variable perform a chi-square test and then calculate Cramer's V
hd1_cat_results = []

for col in ["sex", "chest_pain_type", "fasting_bs", "resting_ecg", 
            "exercise_angina", "st_slope"]:
    
    chi2, p, dof, expected = chi2_contingency(
        pd.crosstab(hd1_cat[col], hd1_cat["heart_disease"])
        )    
    p = "<0.001" if p < 0.001 else str(round(p, 3))
    
    cramers_v = (
        associations(hd1_cat[[col, "heart_disease"]], compute_only = True) 
        ["corr"].loc[col, "heart_disease"].round(2)
        )
    
    hd1_cat_results.append({"variable": col, 
                            "p_value": p, 
                            "cramers_v": cramers_v})

# Create a DataFrame, modify values in a column, and sort rows    
hd1_cat_results = (
    pl.DataFrame(hd1_cat_results)
      .with_columns(
          pl.when(pl.col("variable") == "sex")       
            .then(pl.lit("Sex"))
            .when(pl.col("variable") == "chest_pain_type")
            .then(pl.lit("Chest Pain Type"))
            .when(pl.col("variable") == "fasting_bs")
            .then(pl.lit("Fasting Blood Sugar"))
            .when(pl.col("variable") == "resting_ecg")
            .then(pl.lit("Resting ECG"))
            .when(pl.col("variable") == "exercise_angina")
            .then(pl.lit("Exercise Angina"))
            .when(pl.col("variable") == "st_slope")
            .then(pl.lit("ST Slope"))
            .alias("variable")  
          )
      .sort("cramers_v", descending = True)
    )

# Create a DataFrame and for each variable perform a correlation test
hd1_num_results = pl.DataFrame({
    "variable": [col for col in ["age", "resting_bp", "cholesterol", "max_hr", 
                                 "old_peak"]],
    "correlation": [hd1.select(pl.corr(col, "heart_disease").round(2)).item() 
                    for col in ["age", "resting_bp", "cholesterol", "max_hr", 
                                "old_peak"]]
    # Modify values in a column and sort rows  
    }).with_columns(
        pl.when(pl.col("variable") == "age")
          .then(pl.lit("Age"))
          .when(pl.col("variable") == "resting_bp")
          .then(pl.lit("Resting Blood Pressure")) 
          .when(pl.col("variable") == "cholesterol") 
          .then(pl.lit("Cholesterol"))
          .when(pl.col("variable") == "max_hr")
          .then(pl.lit("Maximum Heart Rate"))
          .when(pl.col("variable") == "old_peak")
          .then(pl.lit("ST Depression")) 
          .alias("variable") 
     ).sort("correlation", descending = True)

# Section 1.3: Machine Learning Model

# Drop columns, create dummy variables, and rename columns
hd1 = (    
    hd1.drop("resting_bp", "cholesterol", "fasting_bs", "resting_ecg")
       .to_dummies(columns = ["sex", "chest_pain_type", "exercise_angina", 
                              "st_slope"])
      .rename(str.lower)
    )

# Perform a train-test split
hd1_x_train, hd1_x_test, hd1_y_train, hd1_y_test = train_test_split(
    hd1.drop("heart_disease"), hd1.select("heart_disease").to_series(), 
    test_size = 0.2, random_state = 123
    )

# Perform min-max scaling
hd1_scaler = MinMaxScaler()
hd1_x_train = hd1_scaler.fit_transform(hd1_x_train)

# Tune hyperparameters with cross-validation to find the best hyperparameters
hd1_best_hp = GridSearchCV(
    estimator = KNeighborsClassifier(),
    param_grid = {"n_neighbors": [19, 20, 21],
                  "weights": ["distance", "uniform"],
                  "metric": ["euclidean", "manhattan"]},
    scoring = "accuracy",
    cv = KFold(n_splits = 5, shuffle = True, random_state = 123)
    ).fit(hd1_x_train, hd1_y_train).best_params_  

# Fit the model to the training data
hd1_knn_fit = KNeighborsClassifier(
    n_neighbors = hd1_best_hp["n_neighbors"], 
    weights = hd1_best_hp["weights"], 
    metric = hd1_best_hp["metric"]
    ).fit(hd1_x_train, hd1_y_train)

# Perform min-max scaling
hd1_x_test = hd1_scaler.transform(hd1_x_test)

# Get the accuracy on the test data
round(hd1_knn_fit.score(hd1_x_test, hd1_y_test), 2)

# Part 2: Linear Regression

# Section 2.1: Data Preprocessing

# Load the data from the CSV file
mc = (
    pl.read_csv("Medical-Cost-Data.csv")
      # Create new columns and modify values of existing columns 
      .with_columns(
          pl.when(pl.col("sex") == "male")
            .then(1)
            .otherwise(0)
            .alias("male"),
          pl.when(pl.col("smoker") == "yes")
            .then(1)
            .otherwise(0)
            .alias("smoker"),
          pl.col("region").alias("region_orig"),
          pl.col("charges").log().alias("log_charges")     
     ).to_dummies("region")
      # Rename columns and select columns in the appropriate order
      .rename({"region_northeast": "northeast", 
               "region_northwest": "northwest", 
               "region_southeast": "southeast", 
               "region_southwest": "southwest", 
               "region_orig": "region"})
      .select("age", "male", "bmi", "children", "smoker", "region", "northwest", 
              "southeast", "southwest", "charges", "log_charges")  
    )

# Section 2.2: Exploratory Data Analysis

# Perform a one-way ANOVA
sm.stats.anova_lm(ols(f"log_charges ~ region", data = mc.to_pandas()).fit(), 
                  typ = 1)

# Create a DataFrame and for each variable perform a correlation test
mc_corr_results = pl.DataFrame({
    "variable": [col for col in ["age", "male", "bmi", "children", "smoker"]],
    "correlation": [mc.select(pl.corr(col, "log_charges").round(2)).item()
                    for col in ["age", "male", "bmi", "children", "smoker"]]
    # Modify values in a column and sort rows
    }).with_columns(
        pl.when(pl.col("variable") == "age")
          .then(pl.lit("Age"))
          .when(pl.col("variable") == "male")
          .then(pl.lit("Male"))
          .when(pl.col("variable") == "bmi")
          .then(pl.lit("BMI"))
          .when(pl.col("variable") == "children")
          .then(pl.lit("Children"))
          .when(pl.col("variable") == "smoker")
          .then(pl.lit("Smoker"))
          .alias("variable") 
     ).sort("correlation", descending = True)

# Select columns that are used in the model
mc = mc.select("age", "children", "smoker", "log_charges")

# Section 2.3: Machine Learning Model
       
# Perform a train-test split
mc_x_train, mc_x_test, mc_y_train, mc_y_test = train_test_split(
    mc.drop("log_charges"), mc.select("log_charges").to_series(), 
    test_size = 0.2, random_state = 123
    )

# Fit the model to the training data
mc_lr_fit = LinearRegression().fit(mc_x_train, mc_y_train)

# Create a DataFrame containing the performance and error metrics
pl.DataFrame({
    "Model": "Linear Regression",
    "R\u00b2": format(mc_lr_fit.score(mc_x_test, mc_y_test), ".3f"),
    "RMSE": "$" + format(root_mean_squared_error(
        mc_y_test.exp(),
        pl.Series(mc_lr_fit.predict(mc_x_test)).exp() *
        (mc_y_train - mc_lr_fit.predict(mc_x_train)).exp().mean()
        ), ",.0f"),
    "MAE": "$" + format(mean_absolute_error(
        mc_y_test.exp(),
        pl.Series(mc_lr_fit.predict(mc_x_test)).exp() *
        (mc_y_train - mc_lr_fit.predict(mc_x_train)).exp().mean()
        ), ",.0f") 
    })

# Part 3: Logistic Regression

# Section 3.1: Data Preprocessing

# Load the data from the CSV file, rename columns, filter, and change data types
hd2 = (
    pl.read_csv("Heart-Disease-2-Data.csv", infer_schema_length = 175)
      .rename({"trestbps": "trest_bps",
               "restecg": "rest_ecg", 
               "thalach": "thal_ach", 
               "oldpeak": "old_peak"})
      .filter(pl.col("age").is_between(35, 75) & (pl.col("chol") < 500) & 
              (pl.col("ca") != "?") & (pl.col("thal") != "?"))
      .with_columns([
          pl.col(c).cast(pl.Utf8).alias(c)
          for c in ["cp", "rest_ecg", "slope", "ca", "thal"]
          ])
      # Drop a column
      .drop("")
    )

# Change data types, select columns, and convert to a Pandas DataFrame   
hd2_cat = (
    hd2.with_columns([
        pl.col(c).cast(pl.Utf8).alias(c)
        for c in ["sex", "fbs", "exang", "present"]
        ])
       .select(pl.col(pl.Utf8))
       .to_pandas() 
    )

# Select columns
hd2_num = hd2.select("age", "trest_bps", "chol", "thal_ach", "old_peak", 
                     "present")    

# Section 3.2: Exploratory Data Analysis

# For each variable perform a chi-square test and then calculate Cramer's V
hd2_cat_results = []

for col in ["sex", "cp", "fbs", "rest_ecg", "exang", "slope", "ca", "thal"]:
    chi2, p, dof, expected = chi2_contingency(
        pd.crosstab(hd2_cat[col], hd2_cat["present"])
        )    
    p = "<0.001" if p < 0.001 else str(round(p, 3))
    
    cramers_v = (
        associations(hd2_cat[[col, "present"]], compute_only = True) 
        ["corr"].loc[col, "present"].round(2)
        )
    
    hd2_cat_results.append({"variable": col, 
                            "p_value": p, 
                            "cramers_v": cramers_v})

# Create a DataFrame, modify values in a column, and sort rows
hd2_cat_results = (
    pl.DataFrame(hd2_cat_results)
      .with_columns(
          pl.when(pl.col("variable") == "sex")       
            .then(pl.lit("Sex"))
            .when(pl.col("variable") == "cp")
            .then(pl.lit("Chest Pain Type"))
            .when(pl.col("variable") == "fbs")
            .then(pl.lit("Fasting Blood Sugar"))
            .when(pl.col("variable") == "rest_ecg")
            .then(pl.lit("Resting ECG"))
            .when(pl.col("variable") == "exang")
            .then(pl.lit("Exercise Angina"))
            .when(pl.col("variable") == "slope")
            .then(pl.lit("ST Slope"))
            .when(pl.col("variable") == "ca")
            .then(pl.lit("Major Vessels"))
            .when(pl.col("variable") == "thal")
            .then(pl.lit("Thalassemia"))
            .alias("variable")  
          )
      .sort("cramers_v", descending = True)
    ) 

# Create a DataFrame and for each variable perform a correlation test
hd2_num_results = pl.DataFrame({
    "variable": [col for col in ["age", "trest_bps", "chol", "thal_ach", 
                                 "old_peak"]],
    "correlation": [hd2.select(pl.corr(col, "present").round(2)).item() 
                    for col in ["age", "trest_bps", "chol", "thal_ach", 
                                "old_peak"]]
    # Modify values in a column and sort rows
    }).with_columns(
        pl.when(pl.col("variable") == "age")
          .then(pl.lit("Age"))
          .when(pl.col("variable") == "trest_bps")
          .then(pl.lit("Resting Blood Pressure")) 
          .when(pl.col("variable") == "chol") 
          .then(pl.lit("Cholesterol"))
          .when(pl.col("variable") == "thal_ach")
          .then(pl.lit("Maximum Heart Rate"))
          .when(pl.col("variable") == "old_peak")
          .then(pl.lit("ST Depression"))
          .alias("variable")    
     ).sort("correlation", descending = True)

# Drop columns, create dummy variables, and rename columns
hd2 = (
    hd2.drop("age", "sex", "trest_bps", "chol", "fbs", "rest_ecg")
       .to_dummies(columns = ["cp", "exang", "slope", "ca", "thal"], 
                   drop_first = True)
       .rename({"ca_1.0": "ca_1", 
                "ca_2.0": "ca_2", 
                "ca_3.0": "ca_3", 
                "thal_3.0": "thal_3", 
                "thal_7.0": "thal_7"})
    )

# Standardize the numeric variables
hd2_scaled = StandardScaler().fit_transform(hd2.select("thal_ach", "old_peak"))
hd2 = hd2.with_columns([
    pl.Series(col_name, hd2_scaled[:, i]) 
              for i, col_name in enumerate(["thal_ach", "old_peak"])
    ])

# Section 3.3: Machine Learning Model

# Perform a train-test split
hd2_x_train, hd2_x_test, hd2_y_train, hd2_y_test = train_test_split(
    hd2.drop("present"), hd2.select("present").to_series(), 
    test_size = 0.2, random_state = 123
    )

# Fit the model to the training data
hd2_logit_fit = LogisticRegression().fit(hd2_x_train, hd2_y_train)

# Get predictions of the model on the test data
hd2_pred = hd2_logit_fit.predict(hd2_x_test)

# Create a confusion matrix based on the predictions
ConfusionMatrixDisplay(
    confusion_matrix(hd2_y_test, hd2_logit_fit.predict(hd2_x_test)) 
    ).plot(cmap = "Blues", colorbar = False)

# Create a DataFrame containing the performance metrics
pl.DataFrame({
    "Model": "Logistic Regression",
    "Accuracy": format(accuracy_score(hd2_y_test, hd2_pred) * 100, ".0f") + "%",
    "Precision": format(precision_score(hd2_y_test, hd2_pred) * 100, 
                        ".0f") + "%",
    "Recall": format(recall_score(hd2_y_test, hd2_pred) * 100, ".0f") + "%",
    "F1 Score": format(f1_score(hd2_y_test, hd2_pred) * 100, ".0f") + "%",
    "ROC AUC Score": format(roc_auc_score(hd2_y_test, hd2_pred) * 100, 
                            ".0f") + "%"
    })

# Part 5: Random Forest
# Section 5.1: Data Preprocessing
# Section 5.2: Exploratory Data Analysis
# Section 5.3: Machine Learning Model