import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Separate features and target variable
X = df.drop(columns=['mpg', 'origin'])
y = df['mpg']


# Split the dataset into training and testing sets
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

# targets --> numpy arrays
y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)


# Adding theta_0 (bias term) to the feature matrix
def generateXvector(X_train):
    vectorX = np.c_[np.ones((len(X_train), 1)), X_train]
    return vectorX

# Initial guess for theta
def theta_init(X_train):
    theta = np.random.randn(len(X_train[0])+1, 1)
    return theta


# Set iterations and learning rate
iterations = X_train.shape[0]
learningrate = 0.01

# Gradient descent function
def Multivariable_Linear_Regression(X_train,y_train,learningrate, iterations):
    """Find the multivarite regression model for the data set
         Parameters:
          X: independent variables matrix
          y: dependent variables matrix
          learningrate: learningrate of Gradient Descent
          iterations: the number of iterations
        Return value: the final theta vector and the plot of cost function
    """
    y_new = np.reshape(y_train, (len(y_train), 1))   # Reshape y to be a column vector
    cost_lst = []
    vectorX = generateXvector(X_train)
    theta = theta_init(X_train)
    m = len(X_train)
    for i in range(iterations):
        gradients = 1/m * vectorX.T.dot(vectorX.dot(theta) - y_new)
        theta = theta - learningrate * gradients
        y_pred = vectorX.dot(theta)
        cost_value = 1/(2*len(y_train))*((y_pred - y_new)**2) 
        #Calculate the loss for each training instance
        total = 0
        for i in range(len(y_train)):
            total += cost_value[i][0] 
            #Calculate the cost function for each iteration
        cost_lst.append(total)
    plt.plot(np.arange(1,iterations),cost_lst[1:], color = 'red')
    plt.title('Cost function Graph')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost')
    plt.show()
    return theta




