import numpy as np
import pickle
import streamlit as st


#loading the saved model
loaded_model = pickle.load(open('model.pkl','rb'))


#Creating a function for Prediction

def Labour_Prediction(Age, Designation, Experience, Working_Hours, RemainingWorkingHours, OverTimeHours, ActualWorkingHours, BeatsPerMinute, Body_Temperature, Skin_Sensor, RR, Motion_Productivity, Motion_Indication, Noise_Detection): 
    input_data = [[Age, Designation, Experience, Working_Hours, RemainingWorkingHours, OverTimeHours, ActualWorkingHours, BeatsPerMinute, Body_Temperature, Skin_Sensor, RR, Motion_Productivity, Motion_Indication, Noise_Detection ]]

    

    # changing the input_data to numpy array
    #input_data_as_numpy_array = np.asarray(input_data)
  
    # reshape the array as we are predicting for one instance
    #input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)()

    prediction = loaded_model.predict(input_data)
    print(prediction)

    if (prediction[0] == 0):
     return 'The person is not productive'
    else:
     return 'The person is productive'
 
    
 
def main():
    
    #Giving a title
    st.title('Labour prediction Web App')
    
    
    #getting the input data from user
    
    Age = st.text_input('Various Age Group')
    Designation = st.text_input('Designation')
    Experience = st.text_input('Work_Experience')
    Working_Hours = st.text_input('Worked_Time')
    RemainingWorkingHours = st.text_input('Remaining Time')
    OverTimeHours = st.text_input('Extra Time Worked')
    ActualWorkingHours = st.text_input('Actual Worked Time')
    BeatsPerMinute = st.text_input('Heart Beat')
    Body_Temperature = st.text_input('Temperature')
    Skin_Sensor = st.text_input('Skin')
    RR = st.text_input('Respiration')
    Motion_Productivity = st.text_input('Movement')
    Motion_Indication = st.text_input('Indicator')
    Noise_Detection = st.text_input('Noise Alert')
    
    
    #Codes for Prediction
    Productivity = ''
    
    #Creating a button for prediction
    if st.button('Productivity Test result'):
        Productivity = Labour_Prediction(Age, Designation, Experience, Working_Hours, RemainingWorkingHours, OverTimeHours, ActualWorkingHours, BeatsPerMinute, Body_Temperature, Skin_Sensor, RR, Motion_Productivity, Motion_Indication, Noise_Detection )
    
    st.success(Productivity)
    
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    