#import all libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

#create Data frame for Data processing
rentalDF = pd.read_csv("data/rental_1000.csv")

#Data transformation (Featurization - use features for model Development)
X = rentalDF[['rooms','sqft']].values  #Features
y = rentalDF['price'].values           #Labels

#Split the Date into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#Model Training
model = LinearRegression().fit(X, y)

#Save the Model 
joblib.dump(model,'rental_price_model.joblib')

# Load the model
model = joblib.load('rental_price_model.joblib')

# Model Prediction
y_pred = model.predict(X_test)

#Compute RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")

#Example Prediction for a specific test sample
sample_index = 0  #Change this index to test different sample
predict_rental_price = model.predict([X_test[sample_index]])[0]
print(f"The Actual Rental Price for Rooms count={X_test[sample_index][0]} and Area in sqft={X_test[sample_index][1]} is={y_test[sample_index]}")
print(f"The Predicted Rental Price for Rooms count={X_test[sample_index][0]} and Area in sqft={X_test[sample_index][1]} is={predict_rental_price}")


#rooms_count = int(input("Enter the number of rooms:="))
#area_sqft   = float(input("Enter the Area in Sqft:=")) 

#user_input = np.array([[rooms_count,area_sqft]])

#predict_rental_price = model.predict(user_input)[0]

#print(f"The predicted Rental price for rooms count={rooms_count} and Area in Sqft={area_sqft} is={predict_rental_price}")

