import numpy as np
import pickle

# loading the saved model
loaded_model1 = pickle.load(open('model.pkl', 'rb'))


## Building a Predictive System 
input_data = (36,0,10,21,2515,4,11,110,97.07,6.98,15.0,7,2,0)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) 

prediction = loaded_model1.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not productive')
else:
  print('The person is productive')