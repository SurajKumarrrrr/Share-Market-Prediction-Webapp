import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

st.write("""
# Stock Price Predictor Using LSTM:
Stock values is very valuable but extremely hard to predict correctly for any human being on their own. This project seeks to solve the problem of Stock Prices Prediction by utilizes Deep Learning models, Long-Short Term Memory (LSTM) Neural Network algorithm, to predict future stock values.\n
""")

st.sidebar.write("""
# Stock Price Prediction:
Predict Stock Price For The Next 30 Days
""")



uploaded_file = st.sidebar.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    st.write("""
# Dataset Sample:
Taken From Quandl\n
""")
    bytes_data = uploaded_file.getvalue()
    data=bytes_data.decode("utf-8")
    df=pd.read_csv(uploaded_file)
    

    st.dataframe(df.head())
    # st.line_chart(df)

# prof_image = Image.open('Created By Picture.png')
# st.sidebar.image(prof_image)




# import pandas as pd
# df=pd.read_csv('AAPL.csv')
# st.write(df.head())
    user_list=df['symbol'].unique().tolist()
    #user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")
    selected_user = st.sidebar.selectbox("Show Analysis wrt",user_list)
    if st.sidebar.button("Show Analysis"):
        df1=df.reset_index()['high']
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler(feature_range=(0,1))
        df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

        training_size=int(len(df1)*0.65)
        test_size=len(df1)-training_size
        train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
        x_input=test_data[341:].reshape(1,-1)
        

        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import LSTM

        model=Sequential()
        model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
        model.add(LSTM(50,return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam')

        from numpy import array

        lst_output=[]
        n_steps=100
        i=0
        while(i<30):
    
            if(len(temp_input)>100):
        #print(temp_input)
                x_input=np.array(temp_input[1:])
                print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
        # print(x_input)
                yhat = model.predict(x_input, verbose=0)
                print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
        #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1
        day_new=np.arange(1,101)
        day_pred=np.arange(101,131)
        fig, ax = plt.subplots()

        ax.plot(day_new,scaler.inverse_transform(df1[1158:]))
        ax.plot(day_pred,scaler.inverse_transform(lst_output))
        plt.xticks(rotation='horizontal')
        st.pyplot(fig)
        st.write("""
# Prediction Graph:
Predict Values For The Next 30 Days\n
""")

        graph_image = Image.open('10dayspredict.png')
        st.image(graph_image,width=500)
        

    # if st.sidebar.button("Show Analysis"):