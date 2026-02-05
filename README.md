<H3>Thanushree M</H3>
<H3>212224240169</H3>
<H3>EX. NO.1</H3>
<H3>05-02-2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
# import libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Read the dataset from drive
df = pd.read_csv("/content/Churn_Modelling.csv")
print(df)

# split the dataset
X = df.iloc[:, :-1].values
print(X)

y = df.iloc[:, -1].values
print(y)

# Finding Missing Values
print(df.isnull().sum())

# Handling Missing values
num_cols = df.select_dtypes(include='number').columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean().round(1))
print(df.isnull().sum())

y = df.iloc[:, -1].values
print(y)


# Check for Duplicates
df.duplicated()

# Detect Outliers
print(df.describe())

# When we normalize the dataset it brings the value of all the features
# between 0 and 1 so that all the columns are in the same range,
# and thus there is no dominant feature.

scaler = MinMaxScaler()
numeric_cols = df.select_dtypes(include=['number']).columns
df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
print(df_scaled)


# splitting the data for training & Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 'test_size=0.2' means 20% test data and 80% train data

print(X_train)
print(len(X_train))

print(X_test)
print(len(X_test))

```


## OUTPUT:
# Read the dataset from drive
<img width="681" height="784" alt="image" src="https://github.com/user-attachments/assets/6b88f291-64a9-42c0-ba88-85a949bbea72" />


# split the dataset
<img width="387" height="208" alt="image" src="https://github.com/user-attachments/assets/f8a7f06b-c920-4369-832e-57c54ece1d78" />






<img width="344" height="86" alt="image" src="https://github.com/user-attachments/assets/31b80042-6127-498f-a586-b2316373caaa" />





# Finding Missing Values

<img width="269" height="313" alt="image" src="https://github.com/user-attachments/assets/3c2ff6f0-17a6-45d7-a3c9-ee0f991c78d0" />




# Handling Missing values

<img width="558" height="370" alt="image" src="https://github.com/user-attachments/assets/0196d8c2-0769-4fb0-9109-79c30a32b9c2" />






<img width="280" height="92" alt="image" src="https://github.com/user-attachments/assets/54a98376-1bb9-4596-b3bd-f51d7f43c521" />






# Check for Duplicates


<img width="303" height="502" alt="image" src="https://github.com/user-attachments/assets/842da2a1-42ca-47e6-864f-cfdd314609b9" />



# Detect Outliers


<img width="636" height="570" alt="image" src="https://github.com/user-attachments/assets/eaf8f4ca-b945-45c3-b440-ed80e11babfb" />



# When we normalize the dataset it brings the value of all the features
# between 0 and 1 so that all the columns are in the same range,
# and thus there is no dominant feature.



<img width="710" height="582" alt="image" src="https://github.com/user-attachments/assets/cd0bc9f7-ac6e-4be2-9082-e565aabb5a8d" />


# splitting the data for training & Testing
# 'test_size=0.2' means 20% test data and 80% train data

<img width="573" height="214" alt="image" src="https://github.com/user-attachments/assets/41dc1d81-8d75-42f4-85d3-f88222158364" />



<img width="682" height="206" alt="image" src="https://github.com/user-attachments/assets/7d61bff5-5cd6-4048-92fa-c9b014630235" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


