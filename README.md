# AI Demand Prediction for Small Food Businesses  
*Supporting UN SDG 8: Decent Work & Economic Growth*  

---

## 🌍 Project Overview  

Small food businesses are a **key part of South Africa’s local economy**, but many struggle to **predict daily customer demand** accurately.

- **Overestimating demand** leads to food waste, higher costs, and environmental harm.  
- **Underestimating demand** results in lost sales, unhappy customers, and reduced profits.

This project uses **Machine Learning (ML)** to build a **demand forecasting model** that helps small businesses prepare the **right amount of food** each day.

✅ **Reduce food waste** → more sustainable operations  
✅ **Improve profits** → better economic growth  
✅ **Empower entrepreneurs** with data-driven decisions  

---

## 🎯 Why SDG 8?  

**UN Sustainable Development Goal 8: Decent Work and Economic Growth** promotes sustained, inclusive, and sustainable economic growth, full and productive employment, and decent work for all.

This project contributes by:  
- Helping small entrepreneurs **stabilize income and grow their businesses**  
- Improving **operational efficiency and sustainability**  
- Supporting **local economic resilience**  

It also aligns with **SDG 12: Responsible Consumption and Production** by reducing food waste.

---

## 🤖 Machine Learning Approach  

- **Type of Learning:** Supervised Learning → **Regression Problem**  
- **Algorithm Used:** Random Forest Regressor  
- **Key Features:**  
  - Day of the Week (e.g., weekends have higher demand)  
  - Weather Data (temperature)  
  - Event Flags (public holidays, local events)  
- **Target:** Predicted daily demand (customer count or sales volume)  

The model is trained and tested on **synthetic data** but can be adapted for real-world sales data.

---

## 📊 Sample Workflow  

1. **Data Collection:** Historical/simulated sales, weather, event data  
2. **Preprocessing:** Handle missing values, encode categorical variables, normalize features  
3. **Model Training:** Train Random Forest, evaluate with MAE and RMSE  
4. **Results:** Visualize Actual vs Predicted demand, analyze trends  

---

## 🛠 Tools & Libraries  

- Python 3.x  
- Google Colab / Jupyter Notebook  
- Pandas & NumPy (data manipulation)  
- Scikit-learn (machine learning)  
- Matplotlib (visualization)  

---

## ✅ How to Run  

1. Clone this repo or download files  
2. Open `demand_prediction.ipynb` in Google Colab  
3. Run all cells to generate predictions and plots  
4. Enhance with real data or APIs as needed  

---

## 📂 Project Structure  

📦 SDG8-Demand-Prediction
┣ 📜 demand_prediction.ipynb # ML model code (Google Colab notebook)
┣ 📜 report.pdf # 1-page project report
┣ 📜 slides.pdf # Pitch deck presentation
┗ 📜 README.md # This documentation file

yaml
Copy
Edit

---

## 📈 Expected Results  

- Accurate daily demand forecasts (low MAE)  
- Clear visualization of Actual vs Predicted demand  
- Insights into effects of day, weather, and events on demand  

---

## 🤝 Ethical & Social Impact  

- Include **diverse regions** (urban and rural) in data  
- Avoid **algorithmic bias** against underrepresented groups  
- Protect privacy and handle data ethically  
- Promote sustainability by reducing food waste  

---

## 🚀 Future Work  

- Integrate **real historical sales data**  
- Connect **live weather APIs** for real-time forecasting  
- Experiment with advanced ML models (e.g., XGBoost, Neural Nets)  
- Deploy as a **web app** for small business use  

---

## 🏆 Why This Project Matters  

**AI empowers small businesses** to make better decisions, reduce waste, and grow sustainably. This project supports economic development while encouraging responsible practices.

> *“AI isn’t just about code—it’s a tool to solve humanity’s greatest challenges.”*  

---

## 📌 Author  
*Kgololosego Moleba* 
📧 [kgolomoleba@gmail.com](mailto:your.kgolomoleba@gmail.com)
