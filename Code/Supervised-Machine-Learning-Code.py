# Title: Supervised Machine Learning Module
# Author: Alexander Zakrzeski
# Date: August 25, 2025

# Load to import, clean, and wrangle data
import os
import pandas as pd
import polars as pl

# Load to analyze associations and run statistical tests 
from dython.nominal import associations
from scipy.stats import chi2_contingency

# Load to train, test, and evaluate machine learning models
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# Set the working directory
os.chdir("/Users/atz5/Desktop/Machine-Learning-In-Python/Data")

# Part 1: K-Nearest Neighbors

# Section 1.1: Data Preprocessing

# Load the data from the CSV file, rename columns, and filter
hd = (
    pl.read_csv("Heart-Disease-Data.csv")
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

# Change data types, select columns, and convert to a Pandas dataframe
hd_cat = (
    hd.with_columns([
        pl.col(c).cast(pl.Utf8).alias(c) 
        for c in ["fasting_bs", "heart_disease"]
        ])
      .select(pl.col(pl.Utf8))
      .to_pandas()
    )

# Select columns
hd_num = hd.select("age", "resting_bp", "cholesterol", "max_hr", "old_peak", 
                   "heart_disease")

# Section 1.2: Exploratory Data Analysis

# For each variable perform a chi-square test and then calculate Cramer's V
hd_cat_results = []

for col in ["sex", "chest_pain_type", "fasting_bs", "resting_ecg", 
            "exercise_angina", "st_slope"]:
    
    chi2, p, dof, expected = chi2_contingency(
        pd.crosstab(hd_cat[col], hd_cat["heart_disease"])
        )    
    p = "<0.001" if p < 0.001 else str(round(p, 3))
    
    cramers_v = (
        associations(hd_cat[[col, "heart_disease"]], compute_only = True) 
        ["corr"].loc[col, "heart_disease"].round(2)
        )
    
    hd_cat_results.append({"variable": col, 
                           "p_value": p, 
                           "cramers_v": cramers_v})

# Create a dataframe, modify values in a column, and sort rows    
hd_cat_results = (
    pl.DataFrame(hd_cat_results)
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

# Create a dataframe and for each variable perform a correlation test
hd_num_results = pl.DataFrame({
    "variable": [col for col in ["age", "resting_bp", "cholesterol", "max_hr", 
                                 "old_peak"]],
    "correlation": [hd.select(pl.corr(col, "heart_disease").round(2)).item() 
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

# Part 4: Machine Learning Model

# Drop columns, create dummy variables, and rename columns
hd = (    
    hd.drop("resting_bp", "cholesterol", "fasting_bs", "resting_ecg")
      .to_dummies(columns = ["sex", "chest_pain_type", "exercise_angina", 
                             "st_slope"])
      .rename(str.lower)
    )

# Perform a train-test split
hd_x_train, hd_x_test, hd_y_train, hd_y_test = train_test_split(
    hd.drop("heart_disease"), hd.select("heart_disease").to_series(), 
    test_size = 0.2, random_state = 123
    )

# Perform min-max scaling of the numeric variables 
hd_scaler = MinMaxScaler()
hd_x_train = hd_scaler.fit_transform(
    hd_x_train.select("age", "max_hr", "old_peak")
    )

# Tune hyperparameters with cross-validation to find the best hyperparameters
hd_best_hp = GridSearchCV(
    estimator = KNeighborsClassifier(),
    param_grid = {"n_neighbors": [19, 20, 21],
                  "weights": ["distance", "uniform"],
                  "metric": ["euclidean", "manhattan"]},
    scoring = "accuracy",
    cv = KFold(n_splits = 5, shuffle = True, random_state = 123)
    ).fit(hd_x_train, hd_y_train).best_params_  

# Fit the model to the training data
hd_knn_fit = KNeighborsClassifier(
    n_neighbors = hd_best_hp["n_neighbors"], 
    weights = hd_best_hp["weights"], 
    metric = hd_best_hp["metric"]
    ).fit(hd_x_train, hd_y_train)

# Perform min-max scaling of the numeric variables
hd_x_test = hd_scaler.transform(
    hd_x_test.select("age", "max_hr", "old_peak")
    )

# Get the accuracy on the test data
round(hd_knn_fit.score(hd_x_test, hd_y_test), 2)