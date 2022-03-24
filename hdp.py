import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data=pd.read_csv('heartdataset.csv')
x=data.drop(columns='target',axis=1)
y=data['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
model=LogisticRegression()
model.fit(x_train,y_train)
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print('accuracy on training data:',training_data_accuracy)
x_test_prediction=model.predict(x_test)
testing_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('accuracy on testing data:',testing_data_accuracy)
input_data=(38,1,2,138,175,0,1,173,0,0,2,4,2)
#change input to numpyarray

input_data_as_numpy_array=np.asarray(input_data)

#reshape numpy array as we are predicting for only one instance

input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)
if(prediction==[0]):
    print(prediction,'no heart diseases')
else:
    print(prediction,'heart is defective')