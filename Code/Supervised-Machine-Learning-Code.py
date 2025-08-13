# Title: Supervised Machine Learning Module
# Author: Alexander Zakrzeski
# Date: August 12, 2025

import os
import polars as pl
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

os.chdir("/Users/atz5/Desktop/Machine-Learning-In-Python/Data")

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