# Title: Unsupervised Machine Learning Module 
# Author: Alexander Zakrzeski
# Date: September 18, 2025

import numpy as np
import os
import polars as pl

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

os.chdir("/Users/atz5/Desktop/Machine-Learning-In-Python/Data")

customers = (
    pl.read_csv("Customer-Segmentation-Data.csv")
      .filter(pl.col("marital_status") != "Unknown")
      .to_dummies(["gender", "education_level", "marital_status"]) 
      .rename({"gender_F": "female", 
               "gender_M": "male",
               "education_level_College": "college",
               "education_level_Doctorate": "doctorate",
               "education_level_Graduate": "graduate",
               "education_level_High School": "high school",
               "education_level_Post-Graduate": "post-graduate",
               "education_level_Uneducated": "uneducated",
               "marital_status_Divorced": "divorced", 
               "marital_status_Married": "married", 
               "marital_status_Single": "single"})
    )

correlations = (
    customers.select("age", "dependent_count", "estimated_income", 
                     "months_on_book", "total_relationship_count", 
                     "months_inactive_12_mon", "credit_limit", 
                     "total_trans_amount", "total_trans_count", 
                     "avg_utilization_ratio")
             .to_pandas()
             .corr()
             .reset_index()
             .rename(columns = {"index": "variable"})
    )

customers = customers.drop("months_on_book", "total_trans_count")

num_cols = ["age", "dependent_count", "estimated_income", 
            "total_relationship_count", "months_inactive_12_mon", 
            "credit_limit", "total_trans_amount", "avg_utilization_ratio"]
scaled = StandardScaler().fit_transform(customers.select(num_cols))
customers_scaled = customers.with_columns([
    pl.Series(name, scaled[:, i]) for i, name in enumerate(num_cols)
    ])

def plot_elbow_curve(df, max_clusters):
    inertias = [] 
    
    for k in range(1, max_clusters + 1): 
        model = KMeans(n_clusters = k, random_state = 123)
        model.fit(df)
        inertias.append(round(model.inertia_, 2))
        
    plt.figure(figsize = (12, 8))
    plt.plot(range(1, max_clusters + 1), inertias, marker = "o") 
    plt.xticks(ticks = range(1, max_clusters + 1), 
               labels = range(1, max_clusters + 1))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    plt.title("Inertia vs. Number of Clusters")
    plt.tight_layout()
    plt.show()
    
    return inertias

plot_elbow_curve(customers_scaled.drop("customer_id"), 10)

customers = customers.with_columns(
    pl.Series("cluster", 
              (KMeans(n_clusters = 4, random_state = 123)
               .fit_predict(customers_scaled.drop("customer_id"))) + 1)
    )
    






X = customers_scaled.drop("customer_id").to_numpy()
ks = range(2, 11)
sil = []
for k in ks:
    labels = KMeans(n_clusters=k, random_state=123).fit_predict(X)
    sil.append(silhouette_score(X, labels))
k_sil = ks[int(np.argmax(sil))]
print("Best k by Silhouette:", k_sil)




sil = []
for k in range(2, 11):
    labels = KMeans(n_clusters=k, random_state=123).fit_predict(customers_scaled.drop("customer_id"))
    sil.append(silhouette_score(customers_scaled.drop("customer_id"), labels))
print("Best k by Silhouette:", ks[int(np.argmax(sil))])
    
    
    

################################################################################
age: customer age in years.
gender: customer gender (M or F).
dependent_count: number of dependents of each customer.
education_level: level of education ("High School", "Graduate", etc.).
marital_status: marital status ("Single", "Married", etc.).
estimated_income: the estimated income for the customer projected by the data science team.
total_relationship_count: number of times the customer contacted the company.
months_inactive_12_mon: number of months the customer did not use the credit card in the last 12 months.
credit_limit: customer's credit limit.
total_trans_amount: the overall amount of money spent on the card by the customer.
avg_utilization_ratio: daily average utilization ratio.
################################################################################
################################################################################



import seaborn as sns

print(customers["Cluster"].value_counts())

fig = plt.figure(figsize = (20, 10))

for i, column in enumerate(["Age", "Annual Income", "Spending Score"]):
    df_plot = customers.groupby("Cluster")[column].mean()
    ax = fig.add_subplot(2, 2, i + 1)
    ax.bar(df_plot.index, df_plot, color = sns.color_palette("Set1"), 
           alpha = 0.6)
    ax.set_title(f"Average {column.title()} per Cluster", alpha = 0.5)
    ax.xaxis.grid(False)
    
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 2, figsize = (16, 8))
sns.scatterplot(x = "Age", y = "Annual Income", hue = "Cluster", 
                data = customers, palette = "Set1", alpha = 0.4, ax = axs[0][0])
sns.scatterplot(x = "Age", y = "Spending Score", hue = "Cluster", 
                data = customers, palette = "Set1", alpha = 0.4, ax = axs[0][1],
                legend = False)
sns.scatterplot(x = "Annual Income", y = "Spending Score", hue = "Cluster", 
                data = customers, palette = "Set1", alpha = 0.4, ax = axs[1][0])

plt.tight_layout()
plt.show()

plot_df = pd.crosstab(
    index = customers["Cluster"], columns = customers["Gender"], 
    values = customers["Gender"], aggfunc = "size", normalize = "index"
    )

fig, ax = plt.subplots(figsize = (12, 6))
plot_df.plot.bar(stacked = True, ax = ax, alpha = 0.6)
ax.set_title(f"% Gender per Cluster", alpha = 0.5)
ax.set_ylim(0, 1.4)
ax.legend(frameon = False)
ax.xaxis.grid(False)

plt.tight_layout()
plt.show()