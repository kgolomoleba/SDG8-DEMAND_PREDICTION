Project Plan: Demand Prediction for Small Food Businesses (SDG 8)
1. Problem Statement
Predict daily demand (number of customers or quantity of food sold) for small food outlets in South Africa to reduce food waste and improve profitability.

2. Data Collection
Features:

Day of the week

Date (to capture trends/seasonality)

Weather data (temperature, rainfall, etc.) — South African weather APIs like SA Weather Service API or openweathermap.org

Local events or holidays (can create synthetic event flags)

Past sales data (number of customers or sales volume) — you can create synthetic data if real data is unavailable

3. Data Preprocessing
Handle missing data

Encode categorical variables (e.g., day of week)

Normalize/scale numerical features

Train/test split

4. Modeling Approach
Use Supervised Regression models to predict numeric demand:

Start simple with Linear Regression

Try Decision Trees or Random Forests for better performance

(Optional) Explore neural networks with TensorFlow or PyTorch

5. Evaluation Metrics
Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Visualize predicted vs actual demand

6. Ethical Considerations
Ensure dataset represents diverse regions (urban and rural)

Avoid bias that may neglect small rural businesses

Promote sustainability by reducing food waste and supporting fair economic growth

7. Deliverables
Python notebook with data processing, model training, evaluation

1-page report summarizing problem, ML approach, results, and ethics

Presentation/demo showing impact (e.g., potential food waste reduction)

Starter Code Outline (Python, Scikit-learn)
python
Copy
Edit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Example: Create synthetic dataset
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=200)
day_of_week = dates.day_name()
weather_temp = np.random.normal(25, 5, size=len(dates))  # temp in °C
event_flag = np.random.choice([0,1], size=len(dates), p=[0.8,0.2])  # 20% days have events
demand = (50 + 10 * (day_of_week.isin(['Friday', 'Saturday'])) + 
          5 * event_flag + 
          np.random.normal(0, 5, len(dates)))

data = pd.DataFrame({
    'date': dates,
    'day_of_week': day_of_week,
    'temperature': weather_temp,
    'event': event_flag,
    'demand': demand
})

# Preprocessing
X = data[['day_of_week', 'temperature', 'event']]
y = data['demand']

# One-hot encode day_of_week
X = pd.get_dummies(X, columns=['day_of_week'], drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))

# Plot predicted vs actual
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Actual vs Predicted Demand")
plt.show()
