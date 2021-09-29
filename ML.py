#!/usr/bin/env python
# coding: utf-8

# In[12]:

#get_ipython().system('pip install pandas')


# In[50]:

# In[13]:

#get_ipython().system('pip install matplotlib')


# In[14]:

#get_ipython().system('pip install seaborn')

# Importing the Dependencies

# In[15]:

#get_ipython().system('pip install scikit-learn')

# In[16]:

from os import altsep, write
import altair
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
import streamlit as st
import plotly.express as px
from streamlit.legacy_caching.caching import _TTLCACHE_TIMER

# Data Collection and Processing

# In[17]:
header = st.container()
dataset = st.container()
features = st.container() 
modelTraining = st.container()
rad = st.sidebar.radio("",["Home","Abstract","Libraries","Workflow","Linear Regression","Lasso Regression"])

with header:
    if rad == "Home":
        st.title("Used Car Price Prediction Project!!!")
        img='https://miro.medium.com/max/1400/1*c_fiB-YgbnMl6nntYGBMHQ.jpeg'
        st.image(img, caption=None, width=250, use_column_width=True, clamp=False, channels='RGB', output_format='auto')
        st.header("Introduction to Machine Learning")

        st.write("The term Machine Learning was coined by Arthur Samuel in 1959, an American pioneer in the field of computer gaming and artificial intelligence, and stated that “it gives computers the ability to learn without being explicitly programmed") 
        st.write("** Machine Learning** is used anywhere from automating mundane tasks to offering intelligent insights, industries in every sector try to benefit from it.")
        st.write("It can be used for the following:")
        st.write("* **Prediction** ")
        st.write("* **Image Recognition**")
        st.write('* **Speech Recognition**')
        st.write('* **Medical Diagnoses**')
        st.write('* **Financial Industry and Trading**')

    if rad == "Abstract":
        st.title("Abstract")
        st.write("A Car price prediction has been a high interest research \n area, as it requires noticeable effort and knowledge of the field expert. Considerable number of distinct attributes are examined for the reliable and accurate prediction.")
        st.write(" To build a model for predicting the price of used cars the data used for the prediction was collected from the Web portal **Kaggle**")
        st.write('Dataset : [https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho](https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho)')
        st.write(
            "Github : [https://github.com/harini-spec/Machine-learning](https://github.com/harini-spec/Machine-learning)")

with dataset:

    if rad == "Libraries":
        st.title("Libraries ")
        st.subheader("Pandas")
        st.write("Pandas is a Python library used for working with data sets.")
        st.write("It has functions for analyzing, cleaning, exploring, and manipulating data.")
        st.subheader("Seaborn")
        st.write("Seaborn is a library that uses Matplotlib underneath to plot graphs.")
        st.write("It will be used to visualize random distributions.")
        st.subheader("Matplotlib")
        st.write("Matplotlib is a low level graph plotting library in python that serves as a visualization utility.")
        st.write("It allows us visual access to huge amounts of data in easily digestible visuals. Matplotlib consists of several plots like line, bar, scatter, histogram etc.")
        st.subheader("Scikit-learn")
        st.write("It provides a selection of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction via a consistence interface in Python. This library, which is largely written in Python, is built upon NumPy, SciPy and Matplotlib.")
        st.subheader("Streamlit")
        st.write("Streamlit is a **API** application programming interface.Streamlit is an open-source python library that is useful to create and share data web apps. It is slowly gaining a lot of momentum in the data science community. Because of the ease with which one can develop a data science web app, many developers use it in their daily workflow.")


# loading the data from csv file to pandas dataframe
if rad == "Workflow":

    st.title("WorkFlow")
    img1 ="https://i.postimg.cc/HngC7hCz/Whats-App-Image-2021-09-28-at-7-44-07-PM.jpg"
    st.image(img1, caption=None, width=250, use_column_width=True, clamp=False, channels='RGB', output_format='auto')
    st.write('')
    st.write('')
    st.header("DataSet")
    st.write("In the mind of a computer, a data set is any collection of data.")
    st.write("It can be anything from an array to a complete database.")
    car_dataset = pd.read_csv(r'C:\Users\Harini\Desktop\demo\car data.csv')


# In[18]:

# inspecting the first 5 rows of the dataframe


# In[19]:

    if st.checkbox("Show Dataset"):
        number = st.number_input("Number of Rows to View",0,10)
        st.dataframe(car_dataset.head(number))

    if st.checkbox("Select Columns To Show"):
        all_columns = car_dataset.columns.tolist()
        selected_columns = st.multiselect("Select",all_columns)
        new_df = car_dataset[selected_columns]
        st.dataframe(new_df)

    st.header("Shape of Data")
# checking the number of rows and columns
    st.write(car_dataset.shape)


# In[20]:


# checking the number of missing values
    #Data Cleaning
    car_dataset.isnull().sum()


# In[21]:


    #Checking the distribution of categorical data
    print(car_dataset.Fuel_Type.value_counts())
    print(car_dataset.Seller_Type.value_counts())
    print(car_dataset.Transmission.value_counts())


# In[22]:


    #encoding "Fuel_Type" Column 
    car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

    #encoding "Seller_Type" Column
    car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

    #encoding "Transmission" Column
    car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)


# In[23]:


    car_dataset.head()


# Splitting the data and Target

# In[24]:       

    X = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
    Y = car_dataset['Selling_Price']

# In[51]:

# Model Training

# 1. Linear Regression

# In[29]:

if rad == "Linear Regression":    

    car_dataset = pd.read_csv(r'C:\Users\Harini\Desktop\demo\car data.csv')

    car_dataset.isnull().sum()


# In[21]:


# checking the distribution of categorical data
    print(car_dataset.Fuel_Type.value_counts())
    print(car_dataset.Seller_Type.value_counts())
    print(car_dataset.Transmission.value_counts())


# In[22]:


# encoding "Fuel_Type" Column
    car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

# encoding "Seller_Type" Column
    car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

# encoding "Transmission" Column
    car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)
# In[50]:
    

    X = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
    Y = car_dataset['Selling_Price']

    st.title("Linear Regression")
    st.write("Linear regression uses the relationship between the data-points to draw a straight line through all them.")
    st.write("Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecasting.")
    st.subheader("")
    st.write(" ")

    
# In[25]:


    print(X)


# In[26]:


    print(Y)


# In[27]:


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)

    # loading the linear regression model
    lin_reg_model = LinearRegression()


# In[32]:


    lin_reg_model.fit(X_train,Y_train)


# Model Evaluation

# In[33]:


# prediction on Training data
    training_data_prediction = lin_reg_model.predict(X_train)


# In[34]:


# R squared Error
    error_score = metrics.r2_score(Y_train, training_data_prediction)
    print("R squared Error : ", error_score)


# Visualize the actual prices and Predicted prices

# In[35]:

    st.header("Linear Regression - Training Data")

    st.write("R squared Error : ", error_score)
    plt.scatter(Y_train, training_data_prediction)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(" Actual Prices vs Predicted Prices")
    plt.show()
    fig = px.scatter(car_dataset,
                    x=Y_train,
                    y=training_data_prediction)

    fig.update_layout(
        xaxis_title="Actual Price",
        yaxis_title="Predicted Price",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )

    st.plotly_chart(fig)


# In[ ]:

# In[37]:


# prediction on Training data
    test_data_prediction = lin_reg_model.predict(X_test)


# In[38]:


# R squared Error
    error_score = metrics.r2_score(Y_test, test_data_prediction)
    print("R squared Error : ", error_score)


# In[39]:

    st.header("Linear Regression - Testing Data")

    st.write("R squared Error : ", error_score)
    plt.scatter(Y_test, test_data_prediction)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(" Actual Prices vs Predicted Prices")
    plt.show()

    fig = px.scatter(car_dataset,
                    x=Y_test,
                    y=test_data_prediction)

    fig.update_layout(
        xaxis_title="Actual Price",
        yaxis_title="Predicted Price",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )

    st.plotly_chart(fig)

# 2.Lasso Regression

if rad == "Lasso Regression":
    car_dataset = pd.read_csv(r'C:\Users\Harini\Desktop\demo\car data.csv')
    
    car_dataset.isnull().sum()


# In[21]:


# checking the distribution of categorical data
    print(car_dataset.Fuel_Type.value_counts())
    print(car_dataset.Seller_Type.value_counts())
    print(car_dataset.Transmission.value_counts())


# In[22]:


# encoding "Fuel_Type" Column
    car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

# encoding "Seller_Type" Column
    car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

# encoding "Transmission" Column
    car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)
# In[50]:
    

    X = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
    Y = car_dataset['Selling_Price']

    st.title("Lasso Regression")
    st.write("Lasso regression is a regularization technique. It is used over regression methods for a more accurate prediction. This model uses shrinkage. Shrinkage is where data values are shrunk towards a central point as the mean. The lasso procedure encourages simple, sparse models (i.e. models with fewer parameters).")
    st.subheader("")
    st.write(" ")

    
# In[25]:


    print(X)


# In[26]:


    print(Y)


# In[27]:


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)
# In[40]:


# loading the linear regression model
    lass_reg_model = Lasso()


# In[41]:


    lass_reg_model.fit(X_train,Y_train)


# In[44]:


# prediction on Training data
    training_data_prediction = lass_reg_model.predict(X_train)


# In[45]:


# R squared Error
    error_score = metrics.r2_score(Y_train, training_data_prediction)
    print("R squared Error : ", error_score)


# Visualize the actual prices and Predicted prices

# In[46]:

    st.header("Lasso Regression - Training Data ")

    st.write("R squared Error : ", error_score)
    plt.scatter(Y_train, training_data_prediction)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(" Actual Prices vs Predicted Prices")
    plt.show()

    fig = px.scatter(car_dataset,
                    x=Y_train,
                    y=training_data_prediction)

    fig.update_layout(
        xaxis_title="Actual Price",
        yaxis_title="Predicted Price",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )

    st.plotly_chart(fig)
# In[47]:


# prediction on Training data
    test_data_prediction = lass_reg_model.predict(X_test)


# In[48]:


# R squared Error
    error_score = metrics.r2_score(Y_test, test_data_prediction)
    print("R squared Error : ", error_score)


# In[49]:

    st.header("Lasso Regression - Testing Data")

    st.write("R squared Error : ", error_score)
    plt.scatter(Y_test, test_data_prediction)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(" Actual Prices vs Predicted Prices")
    plt.show()

    fig = px.scatter(car_dataset,
                    x=Y_test,
                    y=test_data_prediction)

    fig.update_layout(
        xaxis_title="Actual Price",
        yaxis_title="Predicted Price",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )

    st.plotly_chart(fig)
