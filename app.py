
import streamlit as st
import pandas as pd
from prophet import Prophet

st.title("📈 Monthly SKU Sales Forecasting App")
st.write("Upload your historical monthly sales file (from your internal Excel export).")

uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]

    if not {'sku', 'month', 'quantity'}.issubset(df.columns):
        st.error("Excel file must contain columns: 'sku', 'month', 'quantity'")
    else:
        df['month'] = pd.to_datetime(df['month'])

        st.write("Number of unique SKUs detected:", df['sku'].nunique())
        forecast_period = st.slider("Forecast how many months?", 1, 12, 3)

        all_forecasts = []
        for sku in df['sku'].unique():
            sku_data = df[df['sku'] == sku][['month', 'quantity']].rename(columns={'month': 'ds', 'quantity': 'y'})
            if len(sku_data) < 6:
                continue
            model = Prophet()
            model.fit(sku_data)
            future = model.make_future_dataframe(periods=forecast_period, freq='M')
            forecast = model.predict(future)
            forecast['sku'] = sku
            all_forecasts.append(forecast[['ds', 'yhat', 'sku']].tail(forecast_period))

        if all_forecasts:
            result = pd.concat(all_forecasts)
            result.columns = ['Month', 'Predicted Quantity', 'SKU']
            st.success("Forecast completed!")
            st.dataframe(result)

            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button("Download Forecast CSV", data=csv, file_name='sku_forecast.csv')
        else:
            st.warning("Not enough data to forecast any SKU.")
