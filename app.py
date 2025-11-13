import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import timedelta

st.set_page_config(
    page_title="CA Gas Price Predictor",
    layout="wide"
)

st.title("California Weekly Gas Price Predictor")
st.write(
    "This app uses historical weekly gas prices in California and regression models "
    "to predict future prices. Built with Python, Pandas, scikit-learn, and Streamlit."
)

# =========================
# Data loading
# =========================
@st.cache_data
def load_data():
    # Make sure this filename matches the CSV in your repo
    df = pd.read_csv("gas_prices.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Price"] = df["Price"].astype(float)
    df = df.sort_values("Date").reset_index(drop=True)
    df["WeekNum"] = range(len(df))  # simple time index
    return df

df = load_data()

st.subheader("Historical Data Snapshot")
col_info, col_stats = st.columns(2)

with col_info:
    st.write(f"Number of weeks in dataset: **{len(df)}**")
    st.write(f"Date range: **{df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}**")

with col_stats:
    st.write(f"Min price: **${df['Price'].min():.3f}**")
    st.write(f"Max price: **${df['Price'].max():.3f}**")
    st.write(f"Average price: **${df['Price'].mean():.3f}**")

# =========================
# Model selection
# =========================
st.subheader("Model Configuration")

model_type = st.selectbox(
    "Choose regression model",
    ["Linear Regression", "Polynomial Regression"]
)

degree = 1
if model_type == "Polynomial Regression":
    degree = st.slider("Polynomial degree", min_value=2, max_value=4, value=2)

weeks_ahead = st.slider("Weeks into the future to predict", 1, 8, value=4)

# =========================
# Build features and train model
# =========================
X = df[["WeekNum"]]
y = df["Price"]

# Time-based train/test split (no shuffling)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

if model_type == "Linear Regression":
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Future prediction
    future_week_num = int(df["WeekNum"].iloc[-1] + weeks_ahead)
    future_pred = model.predict([[future_week_num]])[0]

    # Slope (rate of change per week)
    slope = model.coef_[0]
else:
    # Polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)

    future_week_num = int(df["WeekNum"].iloc[-1] + weeks_ahead)
    future_X_poly = poly.transform([[future_week_num]])
    future_pred = model.predict(future_X_poly)[0]

    # Effective slope near the end (finite difference)
    if len(df) >= 2:
        last_week = df["WeekNum"].iloc[-1]
        prev_week = df["WeekNum"].iloc[-2]
        last_X_poly = poly.transform([[last_week]])
        prev_X_poly = poly.transform([[prev_week]])
        slope = (model.predict(last_X_poly)[0] - model.predict(prev_X_poly)[0]) / (
            last_week - prev_week
        )
    else:
        slope = np.nan

# =========================
# Metrics
# =========================
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance (on most recent test portion)")
st.write(f"**Mean Absolute Error (MAE):** ${mae:.3f}")
st.write(f"**R² score:** {r2:.3f}")

if not np.isnan(slope):
    st.write(f"**Estimated weekly change in price:** {slope:+.4f} $/week")

# Residual-based uncertainty estimate
residuals = y_test - y_pred
if len(residuals) > 1:
    sigma = residuals.std(ddof=1)
    st.write(f"**Estimated prediction uncertainty (test residual std):** ±${sigma:.3f}")

# =========================
# Latest actual + future prediction
# =========================
latest_date = df["Date"].iloc[-1]
latest_price = df["Price"].iloc[-1]
future_date = latest_date + timedelta(weeks=weeks_ahead)

st.subheader("Prediction")
st.write(
    f"Most recent actual weekly average in California "
    f"({latest_date.strftime('%Y-%m-%d')}): **${latest_price:.3f}**"
)
st.write(
    f"Predicted average gas price in **{weeks_ahead} week(s)** "
    f"({future_date.strftime('%Y-%m-%d')}): **${future_pred:.3f}**"
)

# =========================
# Visualization
# =========================
st.subheader("Historical and Predicted Prices")

fig, ax = plt.subplots(figsize=(12, 6))

# All historical points (line)
ax.plot(
    df["WeekNum"],
    df["Price"],
    marker="o",
    linestyle="-",
    label="Historical actual",
)

# Test period predictions (orange)
ax.scatter(
    X_test["WeekNum"],
    y_pred,
    color="orange",
    s=80,
    label="Model prediction (test segment)",
)

# Future prediction (green)
ax.scatter(
    future_week_num,
    future_pred,
    color="green",
    s=120,
    label=f"Future prediction (+{weeks_ahead} weeks)",
)

# X-axis labels as dates
all_weeks = df["WeekNum"].tolist() + [future_week_num]
all_dates = df["Date"].tolist() + [future_date]
xtick_labels = [d.strftime("%Y-%m-%d") for d in all_dates]

ax.set_xticks(all_weeks)
ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

ax.set_xlabel("Week")
ax.set_ylabel("Price ($)")
ax.set_title("California Weekly Gas Prices: Historical & Predicted")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

# =========================
# Extra info for your DS / applied math flex
# =========================
with st.expander("How this model works"):
    st.markdown(
        """
        - Each week is converted into a numeric index (`WeekNum`), so the model can learn how price changes over time.
        - We split the data into an **earlier training part** and a **more recent test part** without shuffling, 
          which is important for time-series-like data.
        - Depending on your choice:
          - **Linear Regression** fits a straight line to prices over time.
          - **Polynomial Regression** lets the model fit a curved trend of degree 2–4.
        - We evaluate the model using:
          - **Mean Absolute Error (MAE)** — average absolute difference between predictions and actual prices.
          - **R²** — how much of the variation in price is explained by the model.
        """
    )

with st.expander("Model limitations"):
    st.markdown(
        """
        - The dataset currently has only a few weeks of prices, so the model is more of a demo than a production forecast.
        - Real gas prices depend on many factors (oil markets, taxes, refinery issues, seasons), not just week number.
        - A stronger model would add more historical data and more features, and it might use specialized time-series methods.
        """
    )

with st.expander("View raw data"):
    st.dataframe(
        df[["Date", "Price"]].rename(
            columns={"Date": "Week", "Price": "Average Price ($)"}
        )
    )
