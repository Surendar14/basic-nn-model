# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
A neural network with multiple hidden layers and multiple nodes in each hidden layer is known as a deep learning system or a deep neural network. Here the basic neural network model has been created with one input layer, one hidden layer and one output layer.The number of neurons(UNITS) in each layer varies the 1st input layer has 16 units and hidden layer has 8 units and output layer has one unit.

In this basic NN Model, we have used "relu" activation function in input and hidden layer, relu(RECTIFIED LINEAR UNIT) Activation function is a piece-wise linear function that will output the input directly if it is positive and zero if it is negative.

## Neural Network Model
 <img src="https://user-images.githubusercontent.com/75235747/187438155-2415924c-e313-48c3-b54d-1faed1bdeb57.png" width="600">

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('Ex01').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df.head()
df.dtypes
df=df.astype({'A':'int'})
df=df.astype({'B':'float'})
df.dtypes
from sklearn.model_selection import train_test_split
X=df[['A']].values
Y=df[['B']].values
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=20)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train_scaled=scaler.transform(x_train)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
ai_brain = Sequential([
    Dense(2,activation='relu'),
    Dense(1,activation='relu')
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(x=x_train_scaled,y=y_train,epochs=20000)
loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()
x_test_scaled=scaler.transform(x_test)
ai_brain.evaluate(x_test_scaled,y_test)
input=[[100]]
input_scaled=scaler.transform(input)
ai_brain.predict(input_scaled)
```
## Dataset Information
![dataset](https://user-images.githubusercontent.com/75235747/187439549-b24f51a6-305f-4a25-a20a-488783517605.png)

## OUTPUT
### Training Loss Vs Iteration Plot
![plot](https://user-images.githubusercontent.com/75235747/187439938-92eddc78-b2bd-4a73-b7cc-8e5ac229f656.png)

### Test Data Root Mean Squared Error
![o1](https://user-images.githubusercontent.com/75235747/187440422-cda7feaf-504f-4bfa-96d0-591bb4b777c0.png)

### New Sample Data Prediction
![o2](https://user-images.githubusercontent.com/75235747/187440769-573cbb52-64fa-4b5f-a55d-c7efd1714211.png)

## RESULT
A Basic neural network regression model for the given dataset is developed successfully.
