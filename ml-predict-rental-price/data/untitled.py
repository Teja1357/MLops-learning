#import all libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Data Processing 
rentalDF = pd.read_csv('data/rental_1000.csv')

#use featurization for model development
X = rentalDF[['rooms','sqft']].values  #Features
y = rentalDF['Price'].values           #Labels

#Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#Model Training
model = LinearRegression().fit(X, y)

#Model Prediction
predict_rental_price = model.(X_test[0].reshape(1,-1))[0]
Print("The actual rental price for room count=",X_test[0][0],"and","Area in sqft" =,X_test[0][1],"is =",y_test[0])
Print("The predicted rental price for rooms with count=",X_test[0][0],"and","Area in sqft" =,X_test[0][1],predict_rental_price)
