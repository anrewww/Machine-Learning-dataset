from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

  
# fetch dataset 
auto_mpg = fetch_ucirepo(id=9) 
  
# data (as pandas dataframes) 
X = auto_mpg.data.features 
y = auto_mpg.data.targets 
  
# Combine features and target into a single dataframe
df = pd.concat([X, y], axis=1)


# Remove null/NA rows
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

# Display the first few rows of the dataframe
print(df.columns.tolist())


