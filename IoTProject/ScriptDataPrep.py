from IPython.core.interactiveshell import InteractiveShell
from scipy.cluster.hierarchy import linkage, dendrogram

InteractiveShell.ast_node_interactivity = "all"
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib;

matplotlib.use('Qt5Agg')
plt.rcParams['figure.figsize'] = (10, 5)
import missingno as msno
from datetime import date
import warnings

warnings.filterwarnings("ignore")
import xgboost as xgb
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import LocalOutlierFactor

"""
ÖNCELİK	     TARİH	  ARIZA GÖZLEM SÜRESİ (DK)
MAJOR ARIZA	11.09.20	115  
MİNÖR ARIZA	06.10.20	20
MAJOR ARIZA	10.10.20	150
MAJÖR ARIZA	13.10.20	150
MİNÖR ARIZA	18.10.20	300
MAJÖR ARIZA	30.10.20	40
MİNÖR ARIZA	02.11.20	40
"""
sensor_info = {"vibx": "X RMS Velocity in MM vibration horizontal",
               "yatay titreşimler, sıkıştırma probu ile zemin arasındaki sürtünmeden kaynaklanır,"
               "vibz": "Z RMS Velocity in MM vibration veritical",
               "spm": "Slide açı değeri",
               "temp": "Temperature in Celcius",
               "zacc": "Z Peak Acceleration",
               "zfreq": "Z High Frequency RMS high freq RMS acceleration",
               "xkurt": "X Kurtosis the spectral kurtosis for vibration signals",
               "crestfactor": "X Crest Factor=zacc/zfreq"
               }


# Load Data
def load_data(data=r"data/MotorVerisi01.csv"):
    dataframe = pd.read_csv(data, low_memory=True)
    return dataframe


df_ = load_data()

df_.columns = [col.upper() for col in df_.columns]


# Data Understanding
def check_df(dataframe):
    print("##################### Data Shape #####################")
    print(dataframe.shape)
    print(" ")
    print("##################### Data Info #####################")
    print(dataframe.info())
    print(" ")
    print("##################### Data Missing Values #####################")
    print(dataframe.isnull().sum())
    print(" ")
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99]).T)


check_df(df_)


# Data type categorize
def convert_time(dataframe):  # object convert to datetime

    dataframe["TIME"] = pd.to_datetime(dataframe["TIME"]).dt.tz_localize(None)
    return dataframe


df_ = convert_time(df_)

df = df_.copy()


# Nan Values Analyse

def missing_values_table(dataframe, na_name=False):
    na_col = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 1]
    n_miss = dataframe[na_col].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_col].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["NaN", "Ratio"])
    print(missing_df, end="\n")

    if na_name:
        return na_col


missing_values_table(df, True)

# Nan Values Visualization

msno.bar(df)
plt.show();

# NaN Values Heatmap

msno.heatmap(df)
plt.show()


# Drop Columns

def drop_columns(dataframe, col1="NAME", col2="PARTNO", col3="SPM", col4="BALANCERBASINCI"):
    dataframe = dataframe.drop([col1, col2, col3, col4], axis=1)
    return dataframe


df = drop_columns(df)


# Drop NaN Values

def drop_NaN(dataframe):
    dataframe = dataframe.dropna(axis=0)
    return dataframe


df = drop_NaN(df)


# Resample TIME columns

def hours_time_resample(dataframe, col="TIME"):
    dataframe = dataframe.set_index(col)
    dataframe_hours = dataframe.resample("1H").mean()
    return dataframe_hours


def minutes_time_resample(dataframe, col="TIME"):
    dataframe = dataframe.set_index(col)
    dataframe_minutes = dataframe.resample("1Min").mean()
    return dataframe_minutes


def days_time_resample(dataframe, col="TIME"):
    dataframe = dataframe.set_index(col)
    dataframe_days = dataframe.resample("1D").mean()
    return dataframe_days


df_hours = hours_time_resample(df)
df_minutes = minutes_time_resample(df)
df_days = days_time_resample(df)

check_df(df_hours)  # every column has 984 NaN values
check_df(df_minutes)  # every column has 67152 NaN values
check_df(df_days)  # every column has 24 NaN values

df_hours = df_hours.bfill()
df_hours = df_hours.asfreq("H")

df_minutes = df_minutes.bfill()
df_minutes = df_minutes.asfreq("T")

df_days = df_days.bfill()
df_days = df_days.asfreq("D")

missing_values_table(df_hours)  # There is not NaN value
missing_values_table(df_minutes)  # There is not NaN value
missing_values_table(df_days)  # There is not NaN value

df_hours.reset_index().to_excel(r"Hours_MotorVerisi.xlsx", index=False)
df_minutes.reset_index().to_excel(r"Minutes_MotorVerisi.xlsx", index=False)
df_days.reset_index().to_excel(r"Days_MotorVerisi.xlsx", index=False)

# Local Outlier Factor

plt.style.use("seaborn-dark")
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df_hours)

df_scores = clf.negative_outlier_factor_
df_scores[0:5]

np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 100], style='.-')
plt.show()

th = np.sort(df_scores)[28]

# FOR AB TESTING


df_h_lof_values = df_hours[df_scores < th]
df_lof_in_time = df_h_lof_values.index  # Finding lof date
df_h_lof_values.reset_index().to_excel(r"Hours_LOFvalues.xlsx", index=False)  # test group

control_group = df_hours.sample(35, random_state=63)
control_group = control_group.reset_index()
control_group.to_excel(r"Hours_ControlGroup.xlsx", index=False)

#  Exploratory Data Analysis - EDA

num_cols = [col for col in df_hours if str(df_hours.dtypes) != "O"]


def plot_columns(dataframe, numerical):
    plt.figure(figsize=(16, 6))
    dataframe[numerical].plot()


for col in num_cols:
    plot_columns(df_h_lof_values, col)  # LOF values plotting


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        sns.boxplot(x=dataframe[numerical_col], data=dataframe)
        plt.show(block=True)


num_summary(df_hours, num_cols)

for col in num_cols:
    num_summary(df_hours, col, plot=True)

for col in num_cols:
    plot_columns(df_hours, col)

# Correlation/ Heatmap

corr_df_hours = df_hours.corr(method='spearman')
sns.heatmap(corr_df_hours, annot=True, linewidths=0.4, annot_kws={'size': 10})

plt.xticks(rotation=90)
plt.yticks(rotation=0)

# Cluster Map

fig = sns.clustermap(corr_df_hours, row_cluster=True, col_cluster=True, figsize=(8, 8))

plt.setp(fig.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
plt.setp(fig.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.savefig('Motor_ClusterMap.png')

# Clustering - KMeans
df_KMeans = df_hours.copy()
# Elbow method breakdown
ssd = []
K = range(1, 12)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df_KMeans)
    print(kmeans.inertia_)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Residuals Distance")
plt.title("Optimum Number of Clusters")
plt.show()

# Cluster selection
kmeans = KMeans(n_clusters=3).fit(df_KMeans)
df_KMeans["SegmentKmeans"] = kmeans.labels_

# df_KMeans.reset_index().to_excel(r"SegmentKMeans_MotorVerisi.xlsx", index=False)

# Clustering - Hierarchical
df_Hierarch = df_hours.copy()

# Dendrogram Method
plt.figure(figsize=(10, 7))
plt.title('Dendrogram')

linkage_method = linkage(df_Hierarch, method='ward', metric='euclidean')
Dendrogram = dendrogram(linkage_method)

# Cluster selection
cluster_Hierarch = AgglomerativeClustering(n_clusters=3, linkage='ward', affinity='euclidean')
cluster_Hierarch.fit(df_Hierarch)
df_Hierarch["SegmentHierarch"] = cluster_Hierarch.labels_

# Scatter Plot
plt.figure(figsize=(5, 5))
plt.scatter(df_Hierarch["VIBX"], df_Hierarch["TEMP"], c=cluster_Hierarch.fit_predict(df_Hierarch[["VIBX", "TEMP"]]),
            cmap='rainbow')
plt.show()

# df_Hierarch.reset_index().to_excel(r"SegmentHierarch_MotorVerisi.xlsx", index=False)

