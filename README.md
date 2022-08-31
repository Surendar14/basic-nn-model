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
