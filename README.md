AI Demand Prediction for Small Food Businesses
Supporting UN SDG 8: Decent Work & Economic Growth

🌍 Project Overview
Small food businesses are a key part of South Africa’s local economy, but many struggle to predict daily customer demand accurately.

When they overestimate demand, they produce too much food → food waste, higher costs, and environmental harm.

When they underestimate demand, they run out of stock → lost sales, unhappy customers, and reduced profits.

This project uses Machine Learning (ML) to build a demand forecasting model that helps small businesses prepare the right amount of food each day.

✅ Reduced food waste → more sustainable operations
✅ Improved profits → better economic growth
✅ Data-driven decisions → empowers entrepreneurs

🎯 Why SDG 8?
UN Sustainable Development Goal 8: Decent Work and Economic Growth focuses on promoting sustained, inclusive, and sustainable economic growth, full and productive employment, and decent work for all.

This project contributes to SDG 8 by:

Helping small entrepreneurs stabilize income and grow their businesses

Improving operational efficiency and sustainability

Supporting local economic resilience

It also indirectly aligns with SDG 12: Responsible Consumption and Production by reducing food waste.

🤖 Machine Learning Approach
Type of Learning: Supervised Learning → Regression Problem

Algorithm Used: Random Forest Regressor (ensemble model for better accuracy)

Key Features Used:

Day of the Week (e.g., weekends have higher demand)

Weather Data (hot days → more cold drinks, etc.)

Event Flags (public holidays, local events)

Target Variable: Predicted daily demand (number of customers or sales volume)

The model is trained and tested on synthetic data (simulated customer demand), but can be adapted to real sales data.

📊 Sample Workflow
1️⃣ Data Collection

Historical sales data (or simulated dataset)

Weather & event indicators

2️⃣ Data Preprocessing

Handle missing values

Encode categorical data (day names → one-hot encoding)

Normalize numerical features

3️⃣ Model Training

Train a Random Forest model

Evaluate using Mean Absolute Error (MAE) & Root Mean Square Error (RMSE)

4️⃣ Results & Visualization

Compare Actual vs Predicted demand

Plot demand trends

🛠 Tools & Libraries
Google Colab / Jupyter Notebook (for running the model)

Python 3.x

Libraries:

Pandas & NumPy → Data handling

Scikit-learn → Machine Learning

Matplotlib → Visualization

✅ How to Run
Clone this repo or download the files.

Open demand_prediction.ipynb in Google Colab.

Run all cells → The model will generate predictions and show visualizations.

Modify features (add real sales data, live weather APIs) to improve accuracy.

📂 Project Structure
bash
Copy
Edit
📦 SDG8-Demand-Prediction
 ┣ 📜 demand_prediction.ipynb   # Main ML code (Google Colab Notebook)
 ┣ 📜 report.pdf                # 1-page project report
 ┣ 📜 slides.pdf                # Pitch deck presentation
 ┗ 📜 README.md                 # Project documentation
📈 Expected Results
Accurate daily demand predictions (low MAE → better forecasting)

Visualization showing Actual vs Predicted demand

Insights into how day-of-week, weather, and events affect demand

These predictions can help businesses reduce waste & increase profits.

🤝 Ethical & Social Impact
Ensure data includes diverse communities (urban & rural businesses).

Avoid algorithmic bias that could harm underrepresented groups.

Use data ethically & securely, especially if dealing with real sales records.

Promote sustainability by reducing food waste.

🚀 Future Improvements
Integrate real historical sales data

Connect live weather API for real-time forecasting

Compare multiple ML models (e.g., XGBoost, Neural Networks)

Deploy as a simple web app (Flask/Streamlit) for easy use by small businesses

🏆 Why This Project Matters
AI can empower small businesses to make better decisions, reduce waste, and grow sustainably. By focusing on SDG 8, this project contributes to economic development while promoting responsible business practices.

“AI isn’t just about code—it’s a tool to solve humanity’s greatest challenges.”
