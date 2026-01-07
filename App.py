import streamlit as st
import numpy as np
import pickle

# Load scaler and centroids
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("centroids.pkl", "rb") as f:
    centroids = pickle.load(f)

st.title("Customer Segmentation App 🛍️")

st.write("Enter customer details to predict their cluster:")

# Input fields
annual_income = st.number_input("Annual Income (k$):", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
spending_score = st.number_input("Spending Score (1-100):", min_value=1, max_value=100, value=50, step=1)

# Button to predict cluster
if st.button("Predict Cluster"):
    # Create array and scale
    customer_data = np.array([[annual_income, spending_score]])
    customer_scaled = scaler.transform(customer_data)

    # Compute distances to centroids
    distances = np.linalg.norm(centroids - customer_scaled, axis=1)
    cluster = np.argmin(distances)

    st.success(f"The customer belongs to Cluster #{cluster} 🎯")
    st.write("Distance to each cluster centroid:", distances)

# Optional: Show centroids for reference
if st.checkbox("Show cluster centroids (scaled)"):
    st.write(centroids)
