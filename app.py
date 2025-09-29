# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import standard_preprocess, save_model, load_model, ensure_models_dir
from models import arima_train_predict, prophet_train_predict, xgboost_train_predict, lstm_train_predict, HAS_PMDARIMA, HAS_PROPHET, HAS_XGBOOST, HAS_TF
from aqi import calculate_aqi
import os

st.set_page_config(page_title="AirAware — Smart Air Quality Forecast", layout="wide")

st.title("AirAware — Smart Air Quality Forecasting (Streamlit)")
st.markdown("Upload a CSV containing historical AQ / pollutant readings (timestamp column required).")

st.sidebar.header("Upload dataset / Admin")
uploaded = st.sidebar.file_uploader("Upload CSV file (timestamp column required)", type=["csv"])
resample_rule = st.sidebar.selectbox("Resample rule", options=["D", "H"], index=0)
fill_method = st.sidebar.selectbox("Missing value fill method", options=["ffill", "interpolate"], index=0)
selected_pollutant = st.sidebar.text_input("Forecast pollutant column (e.g., PM2.5, PM10)", value="PM2.5")
forecast_periods = st.sidebar.number_input("Forecast horizon (periods)", min_value=1, max_value=365, value=14)

if uploaded:
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = None

tabs = st.tabs(["Data & EDA", "Train & Evaluate Models", "Forecast & Alerts", "Admin / Retrain"])

with tabs[0]:
    st.header("Data & EDA (Milestone 1)")
    if df_raw is None:
        st.info("Please upload your CSV in the left sidebar to start.")
    else:
        st.subheader("Raw Data Preview")
        st.dataframe(df_raw.head(200))
        try:
            df = standard_preprocess(df_raw, resample_rule=resample_rule, fill_method=fill_method)
        except Exception as e:
            st.error(f"Could not preprocess dataset automatically: {e}")
            st.stop()
        st.subheader("Resampled & Preprocessed Data")
        st.dataframe(df.head(200))

        pollutant_cols = [c for c in df.columns if c.lower().startswith("pm") or c.lower() in ["no2", "so2", "o3", "co", "pm2.5", "pm10"]]
        if not pollutant_cols:
            pollutant_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64] and c not in ["dayofweek","month","day","year"]][:4]
        st.write("Detected pollutant / numeric columns:", pollutant_cols)
        col = st.selectbox("Choose a column to inspect", pollutant_cols, index=0)
        st.line_chart(df[col], use_container_width=True)

        st.subheader("Monthly Trends")
        monthly = df[col].resample("M").mean()
        fig = px.line(monthly, title=f"Monthly mean of {col}")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation heatmap (numeric features)")
        numeric = df.select_dtypes(include=[np.number])
        corr = numeric.corr()
        fig2 = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig2, use_container_width=True)

with tabs[1]:
    st.header("Model Training & Evaluation (Milestone 2)")
    if df_raw is None:
        st.info("Upload CSV to train models.")
    else:
        s_col = selected_pollutant
        col_match = None
        df = standard_preprocess(df_raw, resample_rule=resample_rule, fill_method=fill_method)
        for c in df.columns:
            if c.lower().replace(" ", "") == s_col.lower().replace(" ", ""):
                col_match = c
        if col_match is None:
            pm_candidates = [c for c in df.columns if "pm" in c.lower()]
            if pm_candidates:
                col_match = pm_candidates[0]
            else:
                st.error(f"Could not find column named {s_col}. Detected numeric columns: {list(df.select_dtypes(include=[np.number]).columns)}")
                st.stop()

        series = df[col_match].dropna()
        st.write(f"Using series column: **{col_match}** with {len(series)} records.")
        train_size = st.slider("Train ratio", 0.5, 0.95, 0.8)
        split_idx = int(len(series) * train_size)
        train = series.iloc[:split_idx]
        test = series.iloc[split_idx:]
        st.write(f"Train points: {len(train)} | Test points: {len(test)}")

        st.subheader("Select models to train")
        do_arima = st.checkbox("ARIMA (pmdarima auto_arima)", value=True)
        do_prophet = st.checkbox("Prophet", value=True)
        do_xgb = st.checkbox("XGBoost", value=True)
        do_lstm = st.checkbox("LSTM (TensorFlow)", value=False)
        results = {}

        if st.button("Train selected models"):
            with st.spinner("Training..."):
                if do_arima:
                    try:
                        model_arima, fc_arima = arima_train_predict(train, periods=len(test))
                        rmse = np.sqrt(((fc_arima[:len(test)].values - test.values) ** 2).mean())
                        st.success(f"ARIMA trained. RMSE on test: {rmse:.3f}")
                        results["ARIMA"] = {"model": model_arima, "forecast": fc_arima, "rmse": rmse}
                        save_model(model_arima, "arima_model.joblib")
                    except Exception as e:
                        st.error(f"ARIMA failed: {e}")

                if do_prophet:
                    try:
                        model_prophet, fc_prophet = prophet_train_predict(train, periods=len(test))
                        rmse = np.sqrt(((fc_prophet.values - test.values) ** 2).mean())
                        st.success(f"Prophet trained. RMSE on test: {rmse:.3f}")
                        results["Prophet"] = {"model": model_prophet, "forecast": fc_prophet, "rmse": rmse}
                        save_model(model_prophet, "prophet_model.joblib")
                    except Exception as e:
                        st.error(f"Prophet failed: {e}")

                if do_xgb:
                    try:
                        model_xgb, fc_xgb = xgboost_train_predict(train, exog=None, periods=len(test))
                        rmse = np.sqrt(((fc_xgb.values - test.values) ** 2).mean())
                        st.success(f"XGBoost trained. RMSE on test: {rmse:.3f}")
                        results["XGBoost"] = {"model": model_xgb, "forecast": fc_xgb, "rmse": rmse}
                        save_model(model_xgb, "xgb_model.joblib")
                    except Exception as e:
                        st.error(f"XGBoost failed: {e}")

                if do_lstm:
                    try:
                        model_lstm, fc_lstm = lstm_train_predict(train, periods=len(test))
                        rmse = np.sqrt(((fc_lstm.values - test.values) ** 2).mean())
                        st.success(f"LSTM trained. RMSE on test: {rmse:.3f}")
                        results["LSTM"] = {"model": model_lstm, "forecast": fc_lstm, "rmse": rmse}
                        model_lstm.save("models/lstm_model")
                    except Exception as e:
                        st.error(f"LSTM failed: {e}")

            if results:
                st.subheader("Model forecasts vs Actual (test range)")
                fig = px.line(title="Actual vs Forecast")
                fig.add_scatter(x=test.index, y=test.values, mode="lines+markers", name="Actual")
                for name, res in results.items():
                    f = res["forecast"]
                    f_aligned = f[:len(test)]
                    fig.add_scatter(x=test.index, y=f_aligned.values, mode="lines", name=f"{name} Forecast")
                st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.header("Forecast, Alerting & Trend Visualization (Milestone 3)")
    if df_raw is None:
        st.info("Upload CSV to use forecasting & alerts.")
    else:
        model_choice = st.selectbox("Model", options=["ARIMA", "Prophet", "XGBoost", "LSTM", "Auto-best"], index=1)
        forecast_horizon = st.number_input("Forecast horizon (days)", min_value=1, max_value=365, value=forecast_periods)

        if st.button("Run forecast"):
            df = standard_preprocess(df_raw, resample_rule=resample_rule, fill_method=fill_method)
            s_col = selected_pollutant
            col_match = None
            for c in df.columns:
                if c.lower().replace(" ", "") == s_col.lower().replace(" ", ""):
                    col_match = c
            if col_match is None:
                pm_candidates = [c for c in df.columns if "pm" in c.lower()]
                col_match = pm_candidates[0] if pm_candidates else df.columns[0]
            series = df[col_match].dropna()

            try:
                if model_choice == "ARIMA":
                    model_loaded, forecast_series = arima_train_predict(series, periods=forecast_horizon)
                elif model_choice == "Prophet":
                    model_loaded, forecast_series = prophet_train_predict(series, periods=forecast_horizon)
                elif model_choice == "XGBoost":
                    model_loaded, forecast_series = xgboost_train_predict(series, exog=None, periods=forecast_horizon)
                elif model_choice == "LSTM":
                    model_loaded, forecast_series = lstm_train_predict(series, periods=forecast_horizon)
                else:
                    try:
                        model_loaded, forecast_series = prophet_train_predict(series, periods=forecast_horizon)
                    except Exception:
                        model_loaded, forecast_series = xgboost_train_predict(series, exog=None, periods=forecast_horizon)

                st.subheader("Forecast visualization")
                fig = px.line(title=f"Forecast by {model_choice}")
                fig.add_scatter(x=series.iloc[-365:].index, y=series.iloc[-365:].values, mode="lines", name="Recent actual")
                fig.add_scatter(x=forecast_series.index, y=forecast_series.values, mode="lines+markers", name="Forecast")
                st.plotly_chart(fig, use_container_width=True)

                fdf = pd.DataFrame({col_match: forecast_series})
                fdf["AQI"], fdf["Category"] = zip(*fdf.apply(lambda r: calculate_aqi(r), axis=1))
                st.dataframe(fdf.head(50))
                high_risk = fdf[fdf["AQI"]>100]
                if not high_risk.empty:
                    st.warning(f"Predicted {len(high_risk)} high-risk periods with AQI > 100.")
                    st.table(high_risk[["AQI","Category"]])
                else:
                    st.success("No high-risk periods predicted in forecast horizon.")

            except Exception as e:
                st.error(f"Forecast failed: {e}")

        st.subheader("Long-term trends")
        df = standard_preprocess(df_raw, resample_rule=resample_rule, fill_method=fill_method)
        if selected_pollutant in df.columns:
            monthly_mean = df[selected_pollutant].resample("M").mean()
            fig2 = px.line(monthly_mean, title=f"Long term monthly avg of {selected_pollutant}")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Selected pollutant not found in preprocessed data.")

with tabs[3]:
    st.header("Admin / Upload & Retrain (Milestone 4)")
    st.markdown("Upload new dataset here to retrain or overwrite model files. Models are saved to `/models` folder.")
    if uploaded is None:
        st.info("Upload your CSV via sidebar to enable admin actions.")
    else:
        st.write("Current dataset:", uploaded.name)
        if st.button("Retrain all available models with current dataset"):
            st.info("Retraining will run and overwrite saved models. It may take time.")
            try:
                s_col = selected_pollutant
                df = standard_preprocess(df_raw, resample_rule=resample_rule, fill_method=fill_method)
                col_match = None
                for c in df.columns:
                    if c.lower().replace(" ", "") == s_col.lower().replace(" ", ""):
                        col_match = c
                if col_match is None:
                    pm_candidates = [c for c in df.columns if "pm" in c.lower()]
                    col_match = pm_candidates[0] if pm_candidates else df.columns[0]
                series = df[col_match].dropna()
                if HAS_PMDARIMA:
                    m_arima, _ = arima_train_predict(series, periods=14)
                    save_model(m_arima, "arima_model.joblib")
                if HAS_PROPHET:
                    m_prophet, _ = prophet_train_predict(series, periods=14)
                    save_model(m_prophet, "prophet_model.joblib")
                if HAS_XGBOOST:
                    m_xgb, _ = xgboost_train_predict(series, exog=None, periods=14)
                    save_model(m_xgb, "xgb_model.joblib")
                if HAS_TF:
                    m_lstm, _ = lstm_train_predict(series, periods=14, epochs=10)
                    m_lstm.save("models/lstm_model")
                st.success("Retraining completed (where dependencies were available). Models saved in /models.")
            except Exception as e:
                st.error(f"Retrain error: {e}")

st.markdown("---")
st.caption("Built from provided specification.")
