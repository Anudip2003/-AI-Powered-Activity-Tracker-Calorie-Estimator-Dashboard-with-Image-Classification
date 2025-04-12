import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from PIL import Image
import os
import json

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# ✅ Load trained model and class labels
model = load_model("activity_classifier.h5")

with open("class_labels.json", "r") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}


# ✅ Classify uploaded image
def classify_image(img_path):
    img = keras_image.load_img(img_path, target_size=(128, 128))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    return class_labels[class_idx]


# ✅ Set up Streamlit page
st.set_page_config(page_title="Activity Tracker", page_icon="🔥", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    file_path = "D:/TY/Sem6/DS/DataSet.xlsx"
    df = pd.read_excel(file_path, sheet_name="Activity Log")
    stress_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    df['Stress Level'] = df['Stress Level'].map(stress_mapping)
    return df

df = load_data()

# Train calorie prediction model
@st.cache_resource
def train_model(df):
    X = df[['Duration (hrs)', 'Intensity', 'Frequency']]
    y = df['Calories Burned']
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_train.columns

model_calorie, feature_names = train_model(df)

# Sidebar Inputs
st.sidebar.title("🔹 Enter Activity Details")
activity = st.sidebar.selectbox("🏃 Select Activity", df['Activity'].unique())
duration = st.sidebar.slider("⏳ Duration (hrs)", 0.5, 5.0, 1.0)
intensity = st.sidebar.selectbox("💪 Intensity", df['Intensity'].unique())
frequency = st.sidebar.slider("🔄 Frequency (times per week)", 1, 7, 3)

if st.sidebar.button("🔥 Predict Calories Burned"):
    input_data = pd.DataFrame([[duration, intensity, frequency]], columns=['Duration (hrs)', 'Intensity', 'Frequency'])
    input_data = pd.get_dummies(input_data, drop_first=True)
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[feature_names].astype(float)
    prediction = model_calorie.predict(input_data)[0]
    st.sidebar.success(f"🔥 Estimated Calories Burned: {prediction:.2f} kcal")

    if prediction < 200:
        st.sidebar.info("⚡ Try High-Intensity Activities for Better Results!")
    elif 200 <= prediction < 400:
        st.sidebar.info("✅ Good! Maintain Consistency.")
    else:
        st.sidebar.success("🏆 Great Job! Keep It Up.")

# BMI Calculator
st.sidebar.title("⚖ BMI Calculator")
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170)
bmi = round(weight / ((height / 100) ** 2), 2)

if bmi < 18.5:
    bmi_status = "Underweight"
elif 18.5 <= bmi < 24.9:
    bmi_status = "Normal"
elif 25 <= bmi < 29.9:
    bmi_status = "Overweight"
else:
    bmi_status = "Obese"

st.sidebar.success(f"📊 BMI: {bmi} ({bmi_status})")

if bmi >= 25:
    st.sidebar.subheader("🏋️ Recommended Activities for Weight Loss")
    weight_loss_activities = {
        "Running": "30-60 min",
        "Cycling": "45-60 min",
        "Swimming": "40-60 min",
        "Jump Rope": "15-30 min",
        "HIIT Workout": "20-30 min"
    }
    for activity, duration in weight_loss_activities.items():
        st.sidebar.write(f"✅ {activity}: {duration}")

# Main Dashboard
st.title("🏍️ Activity Tracker Dashboard")

# Calories per activity
st.subheader("📊 Calories Burned Per Activity")
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x='Activity', y='Calories Burned', data=df, ax=ax, palette="coolwarm")
plt.xticks(rotation=45)
st.pyplot(fig)

# Weekly Progress Chart
st.subheader("📉 Weekly Calories Burned Progress")
df['Date'] = pd.to_datetime(df['Date'])
weekly_data = df.groupby(pd.Grouper(key='Date', freq='W'))['Calories Burned'].sum()
fig, ax = plt.subplots(figsize=(8, 4))
weekly_data.plot(ax=ax, color="green", marker="o", linestyle="-")
plt.xlabel("Week")
plt.ylabel("Calories Burned")
st.pyplot(fig)

# Activity Distribution
st.subheader("📊 Activity Distribution")
fig, ax = plt.subplots(figsize=(6, 6))
df['Activity'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, startangle=90, cmap="Set3")
st.pyplot(fig)

# Top Performers
st.subheader("🏆 Top 5 Performers for Selected Activity")
selected_activity = st.selectbox("Select an Activity to View Top Performers", df['Activity'].unique())

if st.button("Show Result"):
    top_performers = df[df['Activity'] == selected_activity].sort_values(by="Calories Burned", ascending=False).head(5)
    if not top_performers.empty:
        st.write("### 💙 Top 5 Performers")
        st.dataframe(top_performers[['Name', 'Calories Burned', 'Duration (hrs)', 'Intensity', 'Frequency']])
    else:
        st.warning("⚠ No data available for this activity.")

# Download Processed Data
st.sidebar.header("📅 Download Processed Data")
csv = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("📅 Download CSV", csv, "Activity_Tracker.csv", "text/csv")

# Upload and Predict Activity from Image
st.subheader("📸 Upload an Image to Identify Activity")
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image_obj = Image.open(uploaded_image)
    st.image(image_obj, caption="Uploaded Image", use_column_width=True)

    # Save temporarily
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # Predict activity
    predicted_activity = classify_image(temp_path)
    st.success(f"🏷 Predicted Activity: {predicted_activity}")

    # Cleanup (optional)
    os.remove(temp_path)

st.write("🚀 *Use this dashboard to track and analyze your activity performance!*")
