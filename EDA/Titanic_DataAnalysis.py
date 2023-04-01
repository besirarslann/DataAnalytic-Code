
import numpy as np
import seaborn as sns
import pandas as pd
#pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#print("##################### Loading Dataset using seaborn library #####################")
def load_data(data="titanic"):
    dataframe = sns.load_dataset(data)
    return dataframe
df = load_data()
df.head()
df.columns = [col for col in df.columns]
df.columns


#print("##################### Sex-Count Distrubation #####################")
def gropby_sex(dataframe, x= "sex"):
    return dataframe.groupby([x]).agg({x:"count"})
print(gropby_sex(df))

#print("##################### Number of Unique #####################")
def number_of_unique(dataframe):
    return dataframe.nunique()
number_of_unique(df)


#print("##################### Unique values of Pclass column #####################")
def pclass_unique(dataframe):
    return dataframe.unique()
pclass_unique(df["pclass"])


#print("##################### Unique values of Pclass and Parch columns #####################")
def pclass_parch_unique(dataframe):
    return dataframe.nunique()
pclass_parch_unique(df[["pclass", "parch"]])


def change_to_category(dataframe):
    print("Embarked dtype : ",dataframe.dtype)
    print("*********************************************************")
    return dataframe.astype("category")
df["embarked"] = change_to_category(df["embarked"])


#print("##################### Containing C in the embarked column #####################")
def embarked_C(dataframe):
    return dataframe[dataframe["embarked"] == "C"]
embarked_C(df)


#print("############################################## Age less than 30 and sex Female ##############################################")
def age_sex_less30_Female(dataframe):
    return dataframe[(dataframe["age"] < 30) & (dataframe["sex"] == "female")]
age_sex_less30_Female(df)


#print("############################################## Fare and Age columns, whose greater than 500 and 70   ##############################################")
def fare_age_grater_500_70(dataframe):
    return dataframe[ (dataframe["fare"] > 500) | (dataframe["age"] > 70) ]
fare_age_grater_500_70(df)


#print("############################################## Drop column -> who   ##############################################")
def drop_column(dataframe):
    return dataframe.drop("who", axis=1) # axis=1 for our column-based
drop_column(df)


#print("############################################## Mean assignment by sex(female,male)   ##############################################")
def mean_assign_by_sex(dataframe):
    dataframe.loc[(dataframe["age"].isnull()) & (dataframe["sex"]=="female"), "age"] = dataframe.groupby("sex")["age"].mean()["female"]
    dataframe.loc[(dataframe["age"].isnull()) & (dataframe["sex"]=="male"), "age"] = dataframe.groupby("sex")["age"].mean()["male"]
    return dataframe
mean_assign_by_sex(df)
print("Checking for null in data after assigning mean :", df["age"].isnull().sum())


#print("############################################## Segmentation by Age column   ##############################################")
bins = [0, 18, 28, 38, 48, 58, 68, str(int(df["age"].max()))]
my_labels = ["0_18" , "19_28", "29_38", "39_48", "49_58", "59_68", "69_" + str(int(df["age"].max()))]
df["age_cat"] = len(bins)

def age_categorical(dataframe):
    dataframe["age_cat"] = pd.cut(dataframe["age"], bins, labels=my_labels)
age_categorical(df)
df.head()


#print("############################################## Fillna mode deck column   ##############################################")
def fillna_with_mode(dataframe):
    dataframe["deck"] = dataframe["deck"].fillna(dataframe["deck"].mode()[0])
    return dataframe
fillna_with_mode(df)


#print("####################### Find the sum, count, mean values of the survived variable in the breakdown of the pclass and gender variables #######################")
def groupby_pclass_sex(dataframe):
    return dataframe.groupby(["pclass", "sex"]).agg({"survived" : ["sum","count","mean"]}).reset_index()
groupby_pclass_sex(df)


#print("################## Write a function that returns 1 for whose age is under 30, 0 for whose age is greater than or equal to 30 ################")
def create_age_flag(dataframe):
    dataframe["age_flag"] = dataframe["age"].apply(lambda x : 1 if (x < 30) else 0)
    return dataframe
create_age_flag(df)