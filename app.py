import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

st.title("Fremont Gas Price Predictor")
st.write("""
Predicts average gas prices in Fremont, CA using historical AAA data.
""")

# --- Step 1: Load AAA data ---
url = "https://gasprices.aaa.com/state-prices/"  # CA state page
tables = pd.read_html(url)
df = tables[0]

# Adjust column names if needed
df_fremont = df[df['City'] == 'Fremont']
df_fremont = df_fremont[['Date', 'Price']]
df_fremont['Date'] = pd.to_datetime(df_fremont['Date'])
df_fremont['Price'] = df_fremont['Price'].replace('[\$,]', '', regex=True).astype(float)
df_fremont = df_fremont.sort_values('Date')

# Feature engineering
df_fremont['Day'] = range(len(df_fremont))
X = df_fremont[['Day']]
y = df_fremont['Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
st.write(f"Mean Absolute Error: ${mae:.2f}")

# Predict next week
next_day = [[len(df_fremont)]]
predicted_price = model.predict(next_day)[0]
st.write(f"Predicted next average gas price: ${predicted_price:.2f}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(df_fremont['Date'], y, label='Actual', color='blue')
ax.plot(df_fremont['Date'].iloc[-len(y_pred):], y_pred, label='Predicted', color='orange')
ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.set_title("Fremont Gas Prices - AAA Data")
ax.legend()
st.pyplot(fig)
