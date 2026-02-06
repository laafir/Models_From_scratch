import numpy as np


#LinearRegression
class LinearRegression:

  #initializing the hyperparameter learning rate and no of iteration

  def __init__(self,learning_rate,no_of_iteration):
    self.learning_rate = learning_rate
    self.no_of_iteration = no_of_iteration


  def fit(self, X, Y):

    # convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y).reshape(-1)


    self.m, self.n = X.shape

    # initializing weight and bias
    self.w = np.zeros(self.n)
    self.b = 0

    self.X = X
    self.Y = Y

    # gradient descent
    for i in range(self.no_of_iteration):
        self.update_weights()



  def update_weights(self):
    y_prediction = self.predict(self.X)

    #calculate Gradient
    dw = -(2*(self.X.T).dot(self.Y - y_prediction))/self.m
    db = - 2 * np.sum(self.Y - y_prediction)/self.m

    #updating W and B
    self.w = self.w - self.learning_rate*dw
    self.b = self.b - self.learning_rate*db

  def predict(self,X):
    #line equation
    return X.dot(self.w) + self.b

#LogisticRegression
   
class LogisticRegression:

  #declaring hyper parameters
  def __init__(self,learning_rate=0.002,no_of_iterations=1000):
    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations

  def fit(self,X,Y):
    self.X = X
    self.Y = Y
    
    #rows and columns
    self.m,self.n = X.shape 

    #initializing weights and bias
    self.b = 0
    self.w = np.zeros(self.n)
     
    #calling no of time the updation should happen
    for i in range(self.no_of_iterations):
      self.update_weight()
  
  def update_weight(self):
    #gradient descent
    linear_model = np.dot(self.X, self.w) + self.b
    y_pred = 1 / (1 + np.exp(-linear_model))

    dw = 1/self.m * np.dot(self.X.T,(y_pred-self.Y))
    db = 1/self.m * np.sum(y_pred-self.Y)

    #updating w and b
    self.w = self.w - self.learning_rate*dw
    self.b = self.b - self.learning_rate*db
  
  def predict(self,X):
    linear_model = np.dot(X, self.w) + self.b
    y_pred = 1 / (1 + np.exp(-linear_model))
    y_pred = np.where(y_pred>0.5,1,0)
    return y_pred

#Support Vector Machine

class SVMC:
  # hyperparameters
  def __init__(self, learning_rate=0.001, no_of_iteration=1000, lambda_parameter=0.01):
    self.learning_rate = learning_rate
    self.no_of_iteration = no_of_iteration
    self.lambda_parameter = lambda_parameter

  def fit(self, X, Y):
    self.X = X
    self.Y = Y

    # rows and columns
    self.m, self.n = X.shape

    # initialize weights and bias
    self.w = np.zeros(self.n)
    self.b = 0

    for _ in range(self.no_of_iteration):
      self.update_weight()

  def update_weight(self):
    for idx, x_i in enumerate(self.X):
      condition = self.Y[idx] * (np.dot(x_i, self.w) - self.b) >= 1

      if condition:
        dw = 2 * self.lambda_parameter * self.w
        db = 0
      else:
        dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, self.Y[idx])
        db = self.Y[idx]

      # updating weight and bias
      self.w = self.w - self.learning_rate * dw
      self.b = self.b - self.learning_rate * db

  def predict(self, X):
    linear_output = np.dot(X, self.w) - self.b
    return np.sign(linear_output)

#Lasso Regression
import numpy as np

class LassoRegression:
    
    def __init__(self, no_of_iteration, learning_rate, lambda_parameter):
        self.no_of_iteration = no_of_iteration
        self.learning_rate = learning_rate
        self.lambda_parameter = lambda_parameter

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        # rows and columns
        self.m, self.n = X.shape

        # initializing weights and bias
        self.w = np.zeros(self.n)
        self.b = 0

        for _ in range(self.no_of_iteration):
            self.update_weight()

    def update_weight(self):
        y_pred = self.predict(self.X)

        dw = np.zeros(self.n)

        for j in range(self.n):
            error_term = np.sum(self.X[:, j] * (self.Y - y_pred))

            if self.w[j] > 0:
                dw[j] = (-2 / self.m) * (error_term + self.lambda_parameter)
            else:
                dw[j] = (-2 / self.m) * (error_term - self.lambda_parameter)

        db = (-2 / self.m) * np.sum(self.Y - y_pred)

        # gradient descent update
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        return X.dot(self.w) + self.b