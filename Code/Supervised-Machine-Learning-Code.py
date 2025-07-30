# Title: Supervised Machine Learning Module
# Author: Alexander Zakrzeski
# Date: July 29, 2025

# Load to import, clean, and wrangle data
import polars as pl
from sklearn.datasets import load_breast_cancer

# Load to train, test, and evaluate machine learning models
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Part 1: Machine Learning Workflow

bc = pl.from_pandas(load_breast_cancer(as_frame = True).frame)

bc_x_train, bc_x_test, bc_y_train, bc_y_test = train_test_split(
    bc.drop("target"), bc.select("target").to_series(), test_size = 0.15, 
    random_state = 417
    )

bc_model = LinearSVC(penalty = "l2", loss = "hinge", C = 10, random_state = 417)
bc_model.fit(bc_x_train, bc_y_train)
bc_model.score(bc_x_test, bc_y_test)