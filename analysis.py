import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_path = "C:\\Users\\Rose Psalms\\Documents\\Projects\\assignment  data science\\archive\\Road Accident Data.csv"
df = pd.read_csv(file_path)

# Display basic info
print(df.head())
print(df.info())

# Step 1: Select features (independent variables) and target (dependent variable)
features = ["Speed_limit", "Number_of_Casualties", "Number_of_Vehicles", 
            "Road_Type", "Weather_Conditions", "Light_Conditions", "Urban_or_Rural_Area"]
target = "Accident_Severity"

# Step 2: Handle missing values
df = df[features + [target]].dropna()

# Step 3: Encode categorical variables
label_encoders = {}
for col in ["Road_Type", "Weather_Conditions", "Light_Conditions", "Urban_or_Rural_Area", "Accident_Severity"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for later use

# Step 4: Split data into training and testing sets
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:\nMean Squared Error: {mse:.2f}\nR² Score: {r2:.2f}")

# Step 7: Save model
joblib.dump(model, "accident_severity_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Model saved successfully!")

# Step 8: Predict a new accident scenario
new_data = pd.DataFrame([[30, 2, 1, "Single carriageway", "Fine no high winds", "Daylight", "Urban"]],
                        columns=features)

# Convert new data to numerical values
for col in ["Road_Type", "Weather_Conditions", "Light_Conditions", "Urban_or_Rural_Area"]:
    new_data[col] = label_encoders[col].transform(new_data[col])

predicted_severity = model.predict(new_data)
print(f"Predicted Accident Severity: {predicted_severity[0]:.2f}")
