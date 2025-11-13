import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import timedelta

st.title("California Weekly Gas Price Predictor")
st.write("Predicts weekly average gas prices in California using historical data and linear regression.")

# --- Load CSV ---
@st.cache_data
def load_data():
    df = pd.read_csv("gas_prices.csv")  # make sure the filename matches exactly
    df['Date'] = pd.to_datetime(df['Date'])
    df['Price'] = df['Price'].astype(float)
    df = df.sort_values('Date').reset_index(drop=True)
    return df

df = load_data()

# Feature engineering: simple time index
df['WeekNum'] = range(len(df))
X = df[['WeekNum']]
y = df['Price']

# Train/test split (no shuffling to respect time order)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"**Mean Absolute Error:** ${mae:.3f}")
st.write(f"**R² (test set):** {r2:.3f}")

# Slider to predict n weeks ahead
weeks_ahead = st.slider("Weeks into the future to predict", 1, 8, value=4)
future_week_num = len(df) + weeks_ahead - 1
predicted_price = model.predict([[future_week_num]])[0]
st.write(f"**Predicted average gas price in {weeks_ahead} week(s):** ${predicted_price:.3f}")

# Most recent actual
latest_date = df['Date'].iloc[-1]
latest_price = df['Price'].iloc[-1]
st.write(f"Most recent actual weekly average in California ({latest_date.strftime('%Y-%m-%d')}): **${latest_price:.3f}**")

# --- Bar chart: Actual vs Predicted (test set + future prediction) ---

# Use indices of test set as x positions for historical bars
comparison_weeks = X_test.index.to_list()
actual_prices = y_test.values.tolist()
predicted_prices = y_pred.tolist()

# Positions for bars
actual_x = [i - 0.15 for i in comparison_weeks]
pred_x = [i + 0.15 for i in comparison_weeks]

fig, ax = plt.subplots(figsize=(12, 6))

# Bars for historical test data
bars_actual = ax.bar(actual_x, actual_prices, width=0.3, label='Actual (test)', color='skyblue')
bars_pred = ax.bar(pred_x, predicted_prices, width=0.3, label='Predicted (test)', color='orange')

# Single bar for future prediction
future_x = future_week_num + 0.15
bars_future = ax.bar(future_x, predicted_price, width=0.3,
                     label=f'Predicted (+{weeks_ahead} weeks)', color='green')

# X-axis labels: dates for test set + future date
last_date = df['Date'].iloc[-1]
dates_for_xticks = list(df['Date'].iloc[comparison_weeks])
future_date = last_date + timedelta(weeks=weeks_ahead)
dates_for_xticks.append(future_date)

xtick_positions = comparison_weeks + [future_week_num]
xtick_labels = [d.strftime('%Y-%m-%d') for d in dates_for_xticks]

ax.set_xticks(xtick_positions)
ax.set_xticklabels(xtick_labels, rotation=45, ha='right')

ax.set_ylabel("Price ($)")
ax.set_title("Actual vs Predicted Weekly Gas Prices (California)")
ax.legend()

# Add value labels on top of bars
for container in [bars_actual, bars_pred, bars_future]:
    for bar in container:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.2f}",
            ha='center',
            va='bottom',
            fontsize=8
        )

st.pyplot(fig)

# --- Optional: Show raw data table ---
with st.expander("Show raw historical data"):
    st.dataframe(df[['Date', 'Price']].rename(columns={'Date': 'Week', 'Price': 'Average Price ($)'}))
