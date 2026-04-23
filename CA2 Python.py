#CA-2 

# 1. LOAD LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from scipy import stats

# Style
sns.set_style("whitegrid")
sns.set_context("talk")
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['figure.facecolor'] = '#8FB6C1'   # outer blue
plt.rcParams['axes.facecolor'] = '#F5F5DC'     # inner beige


# 2. LOAD DATA
df = pd.read_csv("C:/Users/Irtiqua/Downloads/AQI-INDIA.csv")

# 3. DATA CLEANING
df = df.drop_duplicates()
df = df.dropna()

if 'last_update' in df.columns:
    df['last_update'] = pd.to_datetime(df['last_update'])
    df['month'] = df['last_update'].dt.month_name()

numeric_df = df.select_dtypes(include=[np.number])

# 4. EDA SUMMARY
print("Dataset Shape:", df.shape)
print("\nStatistical Summary:\n", numeric_df.describe())

# 5. VISUALIZATION (6 GRAPHS)
# 1. Distribution
plt.figure()
sns.histplot(df['pollutant_avg'], kde=True, color='#8E5A9E', edgecolor='white')

mean_val = df['pollutant_avg'].mean()
plt.axvline(mean_val, linestyle='--', color='red', label=f"Mean={mean_val:.1f}")

plt.title("1.Distribution of Avg Pollutant Level", weight='bold', color='navy')
plt.xlabel("Pollutant Average", color='navy')
plt.ylabel("Frequency", color='navy')
plt.legend()
plt.show()

# 2. Mean by Pollutant
plt.figure()
pollutant_mean = df.groupby('pollutant_id')['pollutant_avg'].mean().sort_values(ascending=False)

sns.barplot(
    x=pollutant_mean.index,
    y=pollutant_mean.values,
    palette=['#4B3F72','#51608A','#3E6C7A','#2F837F','#3E9C7F','#6DB06D','#9AC43C']
)

plt.title("2.Mean Pollution Level by Pollutant Type", weight='bold', color='navy')
plt.xticks(rotation=45, color='darkred')
plt.xlabel("Pollutant Type", color='navy')
plt.ylabel("Mean Pollution Level", color='navy')
plt.show()

# 3. Top States
plt.figure()
top_states = df.groupby('state')['pollutant_avg'].mean().sort_values(ascending=False).head(10)

sns.barplot(
    x=top_states.values,
    y=top_states.index,
    palette=sns.color_palette("magma", 10)
)

plt.title("3.Top 10 Most Polluted States", weight='bold', color='navy')
plt.xlabel("Average Pollution Level", color='navy')
plt.ylabel("State", color='navy')
plt.yticks(color='darkred')
plt.show()

# 4. Boxplot
plt.figure()
sns.boxplot(
    x='pollutant_id',
    y='pollutant_avg',
    data=df,
    palette=['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494']
)

plt.title("4.Pollutant Level Distribution", weight='bold', color='navy')
plt.xlabel("Pollutant Type", color='navy')
plt.ylabel("Pollutant Average", color='navy')
plt.xticks(rotation=45, color='darkred')
plt.show()

# 5. Heatmap
plt.figure()
corr = numeric_df.corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    linewidths=1,
    linecolor='black'
)

plt.title("5.Correlation Heatmap", weight='bold', color='navy')
plt.xticks(color='darkred')
plt.yticks(color='darkred')
plt.show()

# 6. Monthly Trend
if 'month' in df.columns:
    plt.figure()
    monthly = df.groupby('month')['pollutant_avg'].mean()

    sns.lineplot(x=monthly.index, y=monthly.values, marker='o', color='green')

    plt.title("6.Monthly Trend", weight='bold', color='navy')
    plt.xlabel("Month", color='navy')
    plt.ylabel("Average Pollution Level", color='navy')
    plt.xticks(rotation=45, color='darkred')
    plt.show()

# 6. HYPOTHESIS TEST
sample = numeric_df['pollutant_avg']
t_stat, p_value = stats.ttest_1samp(sample, 100)

print("\nT-test Results")
print("T-statistic:", t_stat)
print("P-value:", p_value)

# 7. MODEL (LINEAR REGRESSION)
X = numeric_df.drop('pollutant_avg', axis=1)
y = numeric_df['pollutant_avg']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 8. PREDICTION
y_pred = model.predict(X_test)

# 9. EVALUATION
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("\nModel Performance")
print("R2 Score:", r2)
print("RMSE:", rmse)

# 10. PREDICTION PLOT
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred, color='purple')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val],
         [min_val, max_val],
         linestyle='--',
         color='red',
         label='Perfect Fit')
plt.title(f"7.Linear Regression Prediction\n(R²={r2:.3f}, RMSE={rmse:.2f})",
          weight='bold', color='navy')
plt.xlabel("Actual", color='navy')
plt.ylabel("Predicted", color='navy')
plt.legend()
plt.show()