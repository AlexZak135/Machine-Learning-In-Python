# Title: Supervised Machine Learning Module
# Author: Alexander Zakrzeski
# Date: July 28, 2025

# Load to import, clean, and wrangle data
import polars as pl
from sklearn.datasets import load_breast_cancer

# Load to train, test, and evaluate machine learning models

# Part 1: Machine Learning Workflow

bc = pl.from_pandas(load_breast_cancer(as_frame = True).frame)