import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
auto_mpg = fetch_ucirepo(id=9) 
  
# data (as pandas dataframes) 
X = auto_mpg.data.features 
y = auto_mpg.data.targets 

df = pd.concat([X, y], axis=1)

df = df.dropna()  # Drop rows with missing values
df = df.drop_duplicates()  # Drop duplicate rows
df = df.reset_index(drop=True)  # Reset index after dropping rows

X = df.drop(columns=['mpg'])
y = df['mpg']

scalar = StandardScaler()
X = scalar.fit_transform(X)



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(y_train)
