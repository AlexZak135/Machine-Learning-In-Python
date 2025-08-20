# Title: Supervised Machine Learning Module
# Author: Alexander Zakrzeski
# Date: August 19, 2025

import numpy as np
import os
import pandas as pd
import polars as pl

from dython.nominal import associations
from scipy.stats import chi2_contingency

os.chdir("/Users/atz5/Desktop/Machine-Learning-In-Python/Data")

# Part 1: K-Nearest Neighbors

hd = (
    pl.read_csv("Heart-Disease-Data.csv")
      .rename(str.lower)
      .rename({"chestpaintype": "chest_pain_type",
               "fastingbs": "fasting_bs", 
                "restingecg": "resting_ecg",
                "maxhr": "max_hr",   
                "exerciseangina": "exercise_angina",
                "oldpeak": "old_peak",
                "heartdisease": "heart_disease"})
      .filter(pl.col("age").is_between(35, 75) & (pl.col("restingbp") >= 80) &
              pl.col("cholesterol").is_between(100, 500) & 
              (pl.col("old_peak") >= 0))
    )

hd_cat = (
    hd.select("sex", "chest_pain_type", "fasting_bs", "resting_ecg", 
              "exercise_angina", "st_slope", "heart_disease")
      .with_columns([
          pl.col(col).cast(pl.Utf8).alias(col) 
          for col in ["fasting_bs", "heart_disease"]
          ])
      .to_pandas()
    )

hd_num = (
    hd.select("age", "restingbp", "cholesterol", "max_hr", "old_peak", 
              "heart_disease")
      .with_columns(
          pl.col("heart_disease").cast(pl.Utf8).alias("heart_disease")
          )
    )

hd_results = []

for col in ["sex", "chest_pain_type", "fasting_bs", "resting_ecg", 
            "exercise_angina", "st_slope"]:
    chi2, p, dof, expected = chi2_contingency(
        pd.crosstab(hd_cat[col], hd_cat["heart_disease"])
        )
    
    if p < 0.001:
        p = "<0.001"
    else:
        p = str(round(p, 3))
    
    cramers_v = (
        associations(hd_cat[[col, "heart_disease"]], 
                     nom_nom_assoc = "cramer", 
                     cramers_v_bias_correction = True, 
                     compute_only = True)
        ["corr"].loc[col, "heart_disease"].round(2)
        )
    
    hd_results.append({"variable": col, 
                       "p_value": p, 
                       "cramers_v": cramers_v})
    
hd_results = pl.DataFrame(hd_results)





hd.select("age").describe()
hd.select(pl.corr("age", "heart_disease", method = "pearson"))
hd.select("restingbp").describe()
hd.select(pl.corr("restingbp", "heart_disease", method = "pearson"))
hd.select("cholesterol").describe()
hd.select(pl.corr("cholesterol", "heart_disease", method = "pearson"))
hd.select("maxhr").describe()
hd.select(pl.corr("max_hr", "heart_disease", method = "pearson"))
hd.select("oldpeak").describe()
hd.select(pl.corr("old_peak", "heart_disease", method = "pearson"))

hd.select(pl.col("heartdisease").value_counts())




################################################################################
hd = pl.read_csv("Heart-Disease-Data.csv").rename(str.lower)
################################################################################


hd.select(pl.corr(pl.col("age").log(), pl.col("heartdisease"), method = "pearson"))














print("Chi-square statistic:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
print("Cramer's V:", cramers_v)





# Perform chi-square tests
chi2_contingency(pd.crosstab(test_inputs["same_state"], test_inputs["late"]))
chi2_contingency(pd.crosstab(test_inputs["pickup_period"], test_inputs["late"]))
chi2_contingency(pd.crosstab(test_inputs["actual_arrive_period"], 
                             test_inputs["late"]))
# Convert to a Pandas dataframe






################################################################################

Age: age of the patient [years]
Sex: sex of the patient [M: Male, F: Female]
ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
RestingBP: resting blood pressure [mm Hg]
Cholesterol: serum cholesterol [mm/dl]
FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
Oldpeak: oldpeak = ST [Numeric value measured in depression]
ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
HeartDisease: output class [1: heart disease, 0: Normal]

import polars.selectors as cs
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC



# Part 1: Machine Learning Workflow

bc = pl.from_pandas(load_breast_cancer(as_frame = True).frame)

bc_x_train, bc_x_test, bc_y_train, bc_y_test = train_test_split(
    bc.drop("target"), bc.select("target").to_series(), test_size = 0.15, 
    random_state = 417
    )

bc_model_1 = LinearSVC(penalty = "l2", loss = "hinge", C = 10, 
                       random_state = 417)
bc_model_1.fit(bc_x_train, bc_y_train)
bc_model_1.score(bc_x_test, bc_y_test)

bc_model_2 = LinearSVC(penalty = "l2", loss = "squared_hinge", C = 10, 
                       max_iter = 3_500, random_state = 417)
bc_model_2.fit(bc_x_train, bc_y_train)
bc_model_2.score(bc_x_test, bc_y_test)

# Part 2: K-Nearest Neighbors

sp = (
    pl.read_parquet("Subscription-Prediction.parquet")
      .with_row_index("index")
      .to_dummies(columns = "marital")
      .with_columns(
          pl.when(pl.col("y") == "yes")
            .then(1)
            .otherwise(0)
            .alias("y")
          )
    )
sp.group_by("y").len()

sp_train = sp.sample(fraction = 0.85, seed = 417)
sp_test = (
    sp.filter(~pl.col("index").is_in(sp_train.select(pl.col("index"))))
      .drop("index")
    ) 
sp_train = sp_train.drop("index")

sp_x_train = sp_train.drop("y")
sp_x_test = sp_test.drop("y")
sp_y_train = sp_train.select("y").to_series()
sp_y_test = sp_test.select("y").to_series()

def knn(df_x_train, feature, single_test_input, k, df_y_train):
    df_x_train = df_x_train.with_columns(
        (pl.col(feature) - single_test_input[feature]).abs().alias("distance")
        )
    
    indices = (
        df_x_train.with_row_index("index").sort("distance").head(k)["index"]
                  .to_list()
        )
    
    prediction = pl.Series(sorted(df_y_train[indices])).mode()[0]
    
    return prediction

print(f"Prediction: {knn(sp_x_train, "age", sp_x_test.row(215, named = True), 3, 
                     sp_y_train)}")
print(f"Actual: {sp_y_test[215]}")

sp_x_test = sp_x_test.with_columns(
    pl.Series("age_predicted_y", [knn(sp_x_train, "age", row, 3, sp_y_train) 
                                  for row in sp_x_test.iter_rows(named = True)])
    )
sp_x_test = sp_x_test.with_columns(
    pl.Series("campaign_predicted_y", 
              [knn(sp_x_train, "campaign", row, 3, sp_y_train) 
               for row in sp_x_test.iter_rows(named = True)])
    )

print(f"Accuracy: {round(sp_x_test.select(
    ((pl.col("age_predicted_y") == pl.Series(sp_y_test)).mean() * 100)
        .alias("accuracy")
    ).item(), 2)}%")
print(f"Accuracy: {round(sp_x_test.select(
    ((pl.col("campaign_predicted_y") == pl.Series(sp_y_test)).mean() * 100)
        .alias("accuracy")
    ).item(), 2)}%")

def knn(df_x_train, features, single_test_input, k, df_y_train):
    df_x_train = (
        df_x_train.with_columns([
            (pl.col(feature) - single_test_input[feature]) ** 2 
            for feature in features
            ])    
                  .with_columns(
            pl.sum_horizontal([pl.col(feature) for feature in features]).sqrt()
              .alias("distance")
            )
        )
    
    indices = (df_x_train.with_row_index("index").sort("distance")
                         .head(k)["index"].to_list())
    
    prediction = pl.Series(sorted(df_y_train[indices])).mode()[0] 
    
    return prediction 

print(f"Prediction: {knn(sp_x_train, ["age", "campaign", "marital_married", 
                                      "marital_single"], 
                         sp_x_test.row(215, named = True), 3, sp_y_train)}")
print(f"Actual: {sp_y_test[215]}")

sp_x_test = sp_x_test.with_columns(
    pl.Series("predicted_y", 
              [knn(sp_x_train, ["age", "campaign", "marital_married", 
                                "marital_single"], row, 3, sp_y_train) 
               for row in sp_x_test.iter_rows(named = True)])
    )

print(f"Accuracy: {round(sp_x_test.select(
    ((pl.col("predicted_y") == pl.Series(sp_y_test)).mean() * 100)
        .alias("accuracy")
    ).item(), 2)}%")

age_min_max = [sp_x_train.select(pl.col("age")).min().item(), 
               sp_x_train.select(pl.col("age")).max().item()]
campaign_min_max = [sp_x_train.select(pl.col("campaign")).min().item(), 
                    sp_x_train.select(pl.col("campaign")).max().item()]

sp_x_train = sp_x_train.with_columns(
    ((pl.col("age") - age_min_max[0]) / (age_min_max[1] - age_min_max[0])) 
        .alias("age"),
    ((pl.col("campaign") - campaign_min_max[0]) / 
     (campaign_min_max[1] - campaign_min_max[0]))
        .alias("campaign")
    )

sp_x_test = sp_x_test.with_columns(
    ((pl.col("age") - age_min_max[0]) / (age_min_max[1] - age_min_max[0])) 
        .alias("age"),
    ((pl.col("campaign") - campaign_min_max[0]) / 
     (campaign_min_max[1] - campaign_min_max[0]))
        .alias("campaign")
    )    

sp_x_test = sp_x_test.with_columns(
    pl.Series("predicted_y",
              [knn(sp_x_train, ["age", "campaign", "marital_married", 
                                "marital_single"], row, 3, sp_y_train) 
               for row in sp_x_test.iter_rows(named = True)])
    )

print(f"Accuracy: {round(sp_x_test.select(
    ((pl.col("predicted_y") == pl.Series(sp_y_test)).mean() * 100)
        .alias("accuracy")
    ).item(), 2)}%")

sp = (
    pl.read_parquet("Subscription-Prediction.parquet")
      .to_dummies(columns = ["marital", "default"]) 
      .drop("marital_divorced", "default_no")
      .with_columns(
          pl.when(pl.col("y") == "yes")
            .then(1)
            .otherwise(0)
            .alias("y")
          )
    )

sp_x_remain, sp_x_val, sp_y_remain, sp_y_val = train_test_split(
    sp.drop("y"), sp.select("y").to_series(), test_size = 0.20, 
    random_state = 417
    ) 

sp_x_train, sp_x_test, sp_y_train, sp_y_test = train_test_split(
    sp_x_remain, sp_y_remain, test_size = 0.25, random_state = 417
    ) 

scaler = MinMaxScaler()

sp_x_train = scaler.fit_transform(
    sp_x_train[["marital_married", "marital_single", "marital_unknown", 
                "default_unknown", "age", "duration"]]
    )

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(sp_x_train, sp_y_train)

sp_x_val = scaler.transform(
    sp_x_val[["marital_married", "marital_single", "marital_unknown", 
              "default_unknown", "age", "duration"]]
    )

knn.score(sp_x_val, sp_y_val)

knn = KNeighborsClassifier(n_neighbors = 2_000)
knn.fit(sp_x_train, sp_y_train)

knn.score(sp_x_val, sp_y_val)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(sp_x_train, sp_y_train)

knn.score(sp_x_val, sp_y_val)

sp_x_test = scaler.transform(
    sp_x_test[["marital_married", "marital_single", "marital_unknown", 
               "default_unknown", "age", "duration"]]
    )

knn.score(sp_x_test, sp_y_test)

sp = (
    pl.read_parquet("Subscription-Prediction.parquet")
      .with_columns(
          pl.when(pl.col("y") == "yes")
            .then(1)
            .otherwise(0)
            .alias("y")
          )
    )

correlations = pl.DataFrame({
    "variable": [col for col in sp.select(cs.numeric()).columns if col != "y"],
    "correlation": [sp.select(pl.corr("y", col).abs()).item() 
                    for col in sp.select(cs.numeric()).columns if col != "y"]
    }).sort("correlation", descending = True).limit(5)
top_five = correlations.get_column("variable").to_list()  

sp_x_remain, sp_x_val, sp_y_remain, sp_y_val = train_test_split(
    sp.select(top_five), sp.get_column("y"), test_size = 0.20, 
    random_state = 417
    ) 

sp_x_train, sp_x_test, sp_y_train, sp_y_test = train_test_split(
    sp_x_remain, sp_y_remain, test_size = 0.25, random_state = 417
    ) 

scaler = MinMaxScaler()

sp_x_train = scaler.fit_transform(sp_x_train)
sp_x_val = scaler.transform(sp_x_val)
sp_x_test = scaler.transform(sp_x_test)

accuracies = {}

for neighbors in range(1, 6):
    knn = KNeighborsClassifier(n_neighbors = neighbors)
    knn.fit(sp_x_train, sp_y_train)
    accuracy = knn.score(sp_x_val, sp_y_val) 
    accuracies[neighbors] = accuracy

print(accuracies)

for neighbors in range(1, 6):
    knn = KNeighborsClassifier(n_neighbors = neighbors, weights = "distance", 
                               p = 5)
    knn.fit(sp_x_train, sp_y_train)
    accuracy = knn.score(sp_x_val, sp_y_val) 
    accuracies[neighbors] = accuracy

print(accuracies)

knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, 
                        {"n_neighbors": range(1, 10), 
                         "metric": ["minkowski", "manhattan"]}, 
                        scoring = "accuracy")
knn_grid.fit(sp_x_train, sp_y_train)
print(best_score_)
print(best_params_)

knn_grid.best_estimator_.score(sp_x_val, sp_y_val)
knn_grid.best_estimator_.score(sp_x_test, sp_y_test)
