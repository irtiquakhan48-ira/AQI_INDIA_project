🌍 Air Quality Index (AQI) Analysis & Prediction
📌 Project Overview

This project focuses on analyzing air pollution data across different states in India and building a predictive model to understand pollution trends. Using data analysis and visualization techniques, the project identifies key pollutants, highly polluted regions, and relationships between pollution metrics.

The project also includes a machine learning model to predict pollution levels based on available features.

🎯 Objectives
Analyze the distribution of air pollution levels
Compare pollution across different pollutant types
Identify the most polluted states in India
Study pollutant variability using statistical plots
Understand relationships between pollution factors
Analyze monthly pollution trends
Perform hypothesis testing on pollution levels
Build and evaluate a prediction model using Linear Regression
📊 Dataset
Dataset used: AQI India Dataset
Contains 3113 rows and 12 columns
Key features:
Latitude & Longitude
Pollutant ID (PM10, PM2.5, NO2, CO, etc.)
Minimum, Maximum, and Average pollutant values
Timestamp (last_update)
🛠️ Technologies Used
Python
Pandas – Data manipulation
NumPy – Numerical operations
Matplotlib & Seaborn – Data visualization
Scikit-learn – Machine Learning
SciPy – Statistical testing
📈 Visualizations Included
Distribution of Average Pollutant Levels
Mean Pollution by Pollutant Type
Top 10 Most Polluted States
Pollutant Level Distribution (Boxplot)
Correlation Heatmap
Monthly Pollution Trend
Linear Regression Prediction Plot
🤖 Machine Learning Model
Model Used: Linear Regression
Target Variable: pollutant_avg
📊 Performance
R² Score: 0.933
RMSE: 10.51

This indicates a strong predictive performance with low error.

🧪 Hypothesis Testing
Conducted a one-sample t-test
Tested whether average pollution level differs from a benchmark value (100)
📌 Key Insights
PM10 and PM2.5 are the most dominant pollutants
Northern states like Delhi, Haryana, and Uttar Pradesh show higher pollution levels
Strong correlation exists between minimum, maximum, and average pollutant values
Pollution distribution is right-skewed, indicating extreme high values in some regions
