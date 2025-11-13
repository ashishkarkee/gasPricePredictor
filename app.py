import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from datetime import timedelta

st.title("California Weekly Gas Price Predictor")
st.write("Predicts weekly average gas prices in California using historical data and regression.")

# --- Load CSV ---
df = pd.read_csv("gas_prices.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Price'] = df['Price'].astype(float)
df = df.sort_values('Date').reset_index(drop=True)

# Feature engineering
df['WeekNum'] = range(len(df))
X = df[['WeekNum']]
y = df['Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
st.write(f"Mean Absolute Error: ${mae:.3f}")

# Slider to predict n weeks ahead
weeks_ahead = st.slider("Weeks into future", 1, 8)
future_week_num = len(df) + weeks_ahead - 1
predicted_price = model.predict([[future_week_num]])[0]
st.write(f"Predicted average gas price in {weeks_ahead} week(s): ${predicted_price:.3f}")

# Most recent actual
latest_price = df['Price'].iloc[-1]
st.write(f"Most recent actual weekly average (California): ${latest_price:.3f}")

# --- Bar chart ---
comparison_weeks = X_test.index
actual_prices = y_test.values
predicted_prices = y_pred

# Prepare bars including future prediction
bar_positions = list(comparison_weeks) + [future_week_num]
bar_actual = list(actual_prices) + [None]  # no actual yet for future
bar_pred = list(predicted_prices) + [predicted_price]

# Colors
colors_actual = ['skyblue'] * len(comparison_weeks) + [None]
colors_pred = ['orange'] * len(comparison_weeks) + ['green']

fig, ax = plt.subplots(figsize=(12,6))
ax.bar([i-0.15 for i in bar_positions], bar_actual, width=0.3, label='Actual', color=colors_actual)
ax.bar([i+0.15 for i in bar_positions], bar_pred, width=0.3, label='Predicted', color=colors_pred)

# X-axis labels: include future week date
last_date = df['Date'].iloc[-1]
dates_for_xticks = list(df['Date'].iloc[comparison_weeks])
future_date = last_date + timedelta(weeks=weeks_ahead)
dates_for_xticks.append(future_date)
dates_for_xticks_str = [d.strftime('%Y-%m-%d') for d in dates_for_xticks]

ax.set_xticks(bar_positions)
ax.set_xticklabels(dates_for_xticks_str, rotation=45)
ax.set_ylabel("Price ($)")
ax.set_title("Actual vs Predicted Weekly Gas Prices")
ax.legend()

# Add value labels on bars
for bar in ax.patches:
    if bar.get_height() is not None:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height():.2f}", ha='center', fontsize=9)

st.pyplot(fig)
