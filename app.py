import streamlit as st
import pandas as pd
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

# Create pickle file
pickle.dump(model, open("decision_tree_classifier.pkl", "wb"))

# Load pickle
loaded_model = pickle.load(
    open("decision_tree_classifier.pkl", "rb")
)

st.title("Iris Flower Classification")
st.write("Enter flower details")

sepal_length = st.number_input(
    "Sepal Length (cm)", value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)", value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)", value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)", value=0.2
)

if st.button("Predict"):

    input_data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    input_scaled = scaler.transform(
        input_data
    )

    prediction = loaded_model.predict(
        input_scaled
    )

    classes = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    st.success(
        f"Predicted Flower: {classes[prediction[0]]}"
    )