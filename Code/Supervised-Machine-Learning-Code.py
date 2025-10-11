# Title: Supervised Machine Learning Module
# Author: Alexander Zakrzeski
# Date: October 10, 2025

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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# Set the working directory
os.chdir("/Users/atz5/Desktop/Machine-Learning-In-Python/Data")

# Part 1: K-Nearest Neighbors

# Section 1.1: Data Preprocessing

# Load the data from the CSV file, rename columns, and filter
hd = (
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

# Create a DataFrame, modify values in a column, and sort rows    
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

# Create a DataFrame and for each variable perform a correlation test
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

# Section 1.3: Machine Learning Model

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

# Perform min-max scaling
hd_scaler = MinMaxScaler()
hd_x_train = hd_scaler.fit_transform(hd_x_train)

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

# Perform min-max scaling
hd_x_test = hd_scaler.transform(hd_x_test)

# Get the accuracy on the test data
round(hd_knn_fit.score(hd_x_test, hd_y_test), 2)

# Clear the global environment
globals().clear()

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

# Clear the global environment
globals().clear()

# Part 3: Logistic Regression

# Section 3.1: Data Preprocessing

hd = (
    pl.read_csv("Heart-Disease-2-Data.csv", infer_schema_length = 175)
      .rename({"sex": "male", 
               "trestbps": "trest_bps", 
               "restecg": "rest_ecg", 
               "thalach": "thal_ach", 
               "oldpeak": "old_peak"})
      .filter(pl.col("age").is_between(35, 75))
      .drop("") 
    )

hd_cat = hd.select("male") 

hd_num = hd.select("age") 





# Section 3.2: Exploratory Data Analysis

# Section 3.3: Machine Learning Model


["cp", "trest_bps", "chol", "fbs", "rest_ecg", "thal_ach", "exang", "old_peak", 
 "slope", "ca", "thal", "present"]

################################################################################
# Load the data from the CSV file, rename columns, and filter
hd = (
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
################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



auto["high_price"] = 0
auto.loc[auto["price"] > 15000, "high_price"] = 1

X = auto.drop(["price", "high_price"], axis = 1)
y = auto["high_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.20, random_state = 731
    )

auto.plot.scatter(x = "horsepower", y = "high_price")
plt.show()

beta0 = -7
beta1 = 0.05

auto["z"] = beta0 + beta1 * auto["horsepower"]
auto["hz"] = 1 / (1 + np.exp(-auto["z"]))

auto.plot.scatter(x = "z", y = "hz")
plt.show()

prob = np.mean(y_train)
odds = prob / (1 - prob)

auto["L"] = 0

auto.loc[auto["high_price"] == 1, "L"] = -np.log(auto["hz"])
auto.loc[auto["high_price"] == 0, "L"] = -np.log(1 - auto["hz"])

loss = sum(auto["L"])

model = LogisticRegression()
X_sub = X_train[["horsepower"]]
model.fit(X_sub, y_train)

intercept = model.intercept_
odds = np.exp(intercept)

log_or = model.coef_[0,0]
odds_ratio = np.exp(log_or)

X_sub = X_train[["horsepower", "highway_mpg"]]
model.fit(X_sub, y_train)

horsepower_or = np.exp(model.coef_[0, 0])
highway_mpg_or = np.exp(model.coef_[0, 1])

model.predict_proba(X_sub)

X = X_train[["horsepower"]]
model = LogisticRegression()
model.fit(X, y_train)
accuracy = model.score(X, y_train)

predictions = model.predict(X)

tp = sum((y_train == 1) & (predictions == 1))
fn = sum((y_train == 1) & (predictions == 0))
recall = tp / (tp + fn)

tn = sum((y_train == 0) & (predictions == 0))
fp = sum((y_train == 0) & (predictions == 1))
specificity = tn / (tn + fp)

tp = sum((y_train == 1) & predictions == 1)
fp = sum((y_train == 0) & predictions == 1)
precision = tp / (tp + fp)

tn = sum((y_train == 0) & (predictions == 0))
fn = sum((y_train == 1) & (predictions == 0))
npv = tn / (tn + fn)

X1 = X_train[["length", "horsepower"]]
X2 = X_train[["stroke", "compression_ratio"]]

X1_test = X_test[["length", "horsepower"]]
X2_test = X_test[["stroke", "compression_ratio"]]

model1 = LogisticRegression()
model2 = LogisticRegression()

model1.fit(X1, y_train)
model2.fit(X2, y_train)

test_accuracy1 = model1.score(X1_test, y_test)
test_accuracy2 = model2.score(X2_test, y_test)

summary = auto.groupby("high_price").agg(
	{
    	"horsepower": "mean",
      	"width": "mean",
        "stroke": "mean",
        "compression_ratio": "mean"
    }
)

X1 = X_train[["horsepower"]]
X2 = X_train[["horsepower", "compression_ratio"]]
X3 = X_train[["horsepower", "compression_ratio", "city_mpg"]]

X1_test = X_test[["horsepower"]]
X2_test = X_test[["horsepower", "compression_ratio"]]
X3_test = X_test[["horsepower", "compression_ratio", "city_mpg"]]

model1 = LogisticRegression()
model2 = LogisticRegression()
model3 = LogisticRegression()

model1.fit(X1, y_train)
model2.fit(X2, y_train)
model3.fit(X3, y_train)


train_accuracies = [
    model1.score(X1, y_train),
    model2.score(X2, y_train),
    model3.score(X3, y_train)
]

test_accuracies = [
    model1.score(X1_test, y_test),
    model2.score(X2_test, y_test),
    model3.score(X3_test, y_test)
]

X1 = X_train[["engine_size", "horsepower"]]

model =  LogisticRegression()
model.fit(X1, y_train)

accuracy = model.score(X_test[["engine_size", "horsepower"]], y_test)
test_predictions = model.predict(X_test[["engine_size", "horsepower"]])
confusion = confusion_matrix(y_test, test_predictions)