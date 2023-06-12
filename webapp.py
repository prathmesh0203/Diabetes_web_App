import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st


# Creating a title and sub-title
st.write("""
Diabetes Detection
Detect if someone has diabetes using machine learning and python!
""")

# Open and display an image
image = Image.open('download.jpeg')
st.image(image, caption='ML', use_column_width=True)

# Get the data
df = pd.read_csv('diabetes.csv')

# Set a subheader
st.subheader('Data Information:')

# Show the data as a table
st.dataframe(df)

# Show statistics on the data
st.write(df.describe())

# Show the data as a chart
chart = st.bar_chart(df)

# Split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

# Split the dataset into 75% Training and 25% Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Get the feature input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)

    # Store a dictionary into a variable
    user_data = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'BMI': BMI,
        'DPF': DPF,
        'age': age
    }

    # Transform the data into a data frame
    features = pd.DataFrame(user_data, index=[0])
    return features

# Store the user input into a variable
user_input = get_user_input()

# Set a subheader and display the user input
st.subheader('User Input:')
st.write(user_input)

# Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Show the model's metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')

# Store the model's predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

# Set a subheader and display the classification
st.subheader('Classification:')
st.write(prediction)

# Add educational content and resources
st.subheader('Educational Resources')

# What is Diabetes?
st.markdown('**What is Diabetes?**')
st.markdown(' - [About Diabetes](https://www.diabetes.org/diabetes)')

# Types of Diabetes
st.markdown('**Types of Diabetes:**')
st.markdown(' - [Types of Diabetes](https://www.diabetes.org/diabetes/type-1)')
st.markdown(' - [Diabetes Types](https://www.mayoclinic.org/diseases-conditions/diabetes/diagnosis-treatment/drc-20371451)')

# Diabetes Risk Factors
st.markdown('**Diabetes Risk Factors:**')
st.markdown(' - [Diabetes Risk Factors](https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/symptoms-causes/syc-20351193)')

# Prevention and Management of Diabetes
st.markdown('**Prevention and Management of Diabetes:**')
st.markdown(' - [Prevention](https://www.diabetes.org/diabetes/prevention) and [Living with Diabetes](https://www.diabetes.org/diabetes/living-with)')
st.markdown(' - [Prevent Diabetes](https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-type-2-diabetes)')
st.markdown(' - [Prevent Type 2 Diabetes](https://www.cdc.gov/diabetes/prevention/index.html)')

# Healthy Lifestyle Tips
st.markdown('**Healthy Lifestyle Tips:**')
st.markdown(' - [Healthy Eating](https://www.diabetes.org/nutrition)')
st.markdown(' - [Physical Activity](https://www.diabetes.org/fitness)')
st.markdown(' - [Stress Management](https://www.diabetes.org/mental-health)')

# Diabetes Monitoring and Treatment
st.markdown('**Diabetes Monitoring and Treatment:**')
st.markdown(' - [Diabetes Diagnosis and Treatment](https://www.mayoclinic.org/diseases-conditions/diabetes/diagnosis-treatment/drc-20371451)')

# Support and Communities
st.markdown('**Support and Communities:**')
st.markdown(' - [Community and Support](https://www.diabetes.org/community)')
st.markdown(' - [Support Forum](https://www.diabetes.org.uk/how_we_help/support_forum)')
