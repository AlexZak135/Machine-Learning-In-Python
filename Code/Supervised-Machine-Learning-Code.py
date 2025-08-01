# Title: Supervised Machine Learning Module
# Author: Alexander Zakrzeski
# Date: July 31, 2025

import os
import polars as pl
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
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
