import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset (You can replace this with your own dataset)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
df = pd.read_csv(url)

# Rename columns for clarity
df.columns = ["Date", "Temperature"]

# Generate synthetic features (Replace with real-world data)
np.random.seed(42)
df["Humidity"] = np.random.randint(50, 100, df.shape[0])
df["WindSpeed"] = np.random.randint(0, 30, df.shape[0])
df["WeatherCondition"] = np.random.choice(["Sunny", "Cloudy", "Rainy", "Foggy"], df.shape[0])

# Convert WeatherCondition (NLP) into numeric labels
label_encoder = LabelEncoder()
df["WeatherConditionEncoded"] = label_encoder.fit_transform(df["WeatherCondition"])

# Select features and target variable
X = df[["Humidity", "WindSpeed", "WeatherConditionEncoded"]]
y = df["Temperature"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict temperature for test set
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}°C")

# Predict temperature for a new weather condition
def predict_temperature(humidity, wind_speed, weather_condition):
    weather_code = label_encoder.transform([weather_condition])[0]
    prediction = model.predict([[humidity, wind_speed, weather_code]])
    return prediction[0]

# Example prediction
humidity_input = 80
wind_speed_input = 10
weather_input = "Rainy"

predicted_temp = predict_temperature(humidity_input, wind_speed_input, weather_input)
print(f"Predicted Temperature for {weather_input} (Humidity: {humidity_input}%, Wind: {wind_speed_input} km/h): {predicted_temp:.2f}°C")
