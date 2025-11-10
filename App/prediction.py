from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
import numpy as np
import streamlit as st
import pandas as pd

def create_features(df, lag=21):

    df_agg = df.groupby('date')['total_sales'].sum().reset_index()
    df_agg.columns = ['ds', 'y']
    
    df_agg['dayofweek'] = df_agg['ds'].dt.dayofweek 
    df_agg['month'] = df_agg['ds'].dt.month
    df_agg['dayofyear'] = df_agg['ds'].dt.dayofyear
    df_agg['is_weekend'] = df_agg['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    for i in range(1, lag + 1):
        df_agg[f'revenue_lag_{i}'] = df_agg['y'].shift(i)

    df_agg.dropna(inplace=True)
    
    return df_agg

@st.cache_data
def run_sklearn_prediction(df, forecast_periods=30, backtest_periods=90):
    LAG_DAYS = 21 # Días de lag para las features
    
    # Usar la función original create_features con todos los 21 lags
    df_features = create_features(df, lag=LAG_DAYS)
    
    TARGET = 'y'
    FEATURES = [col for col in df_features.columns if col not in ['ds', TARGET]]
    
    # 2. División de Datos para Entrenamiento y Backtesting
    train_size = len(df_features) - backtest_periods
    
    df_train = df_features.iloc[:train_size].copy()
    df_backtest_actuals = df_features.iloc[train_size:].copy()
    
    X_train = df_train[FEATURES]
    y_train = df_train[TARGET]
    
    # 3. Entrenamiento del Modelo (Optimizaciones de Regularización)
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10,        # Mantenemos la regularización
        min_samples_leaf=5,  # Mantenemos la regularización
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred_history = model.predict(X_train)
    historical_mae = mean_absolute_error(y_train, y_pred_history) # MAE Histórico
    
    # ----------------------------------------------------
    # 4. BACKTESTING WALK-FORWARD (RECURSIVO REAL)
    # ----------------------------------------------------
    
    backtest_results = pd.DataFrame(
        index=df_backtest_actuals['ds'], 
        columns=['actual', 'prediction']
    )
    backtest_results['actual'] = df_backtest_actuals['y'].values
    
    last_train_row = df_train.iloc[-1]
    current_input = last_train_row[FEATURES].to_dict()
    
    # Bucle de predicción Walk-Forward
    for i, date in enumerate(df_backtest_actuals['ds']):
        
        # 4a. Actualizar Features de Fecha 
        current_input['dayofweek'] = date.dayofweek
        current_input['month'] = date.month
        current_input['dayofyear'] = date.dayofyear
        current_input['is_weekend'] = 1 if date.dayofweek >= 5 else 0
        
        # 4b. Predecir el siguiente valor
        X_pred = pd.DataFrame([current_input], columns=FEATURES)
        next_revenue = model.predict(X_pred)[0]
        
        backtest_results.loc[date, 'prediction'] = next_revenue
        
        # 4c. ACUMULACIÓN (Usando la PREDICCIÓN para propagar el error)
        # ESTO ES EL BACKTESTING REAL (donde el error se propaga)
        for j in range(LAG_DAYS, 1, -1):
            current_input[f'revenue_lag_{j}'] = current_input[f'revenue_lag_{j-1}']
            
        # El Lag_1 para la siguiente iteración es la predicción del día actual
        current_input['revenue_lag_1'] = next_revenue # <--- ¡CLAVE!
        
    # 4d. Cálculo de MAE del Backtesting (Recurrente)
    backtest_mae = mean_absolute_error(
        backtest_results['actual'].values, 
        backtest_results['prediction'].values
    )
    
    # ----------------------------------------------------
    # 5. PRONÓSTICO FUTURO (Sigue igual)
    # ----------------------------------------------------
    
    last_date = df_features['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods)
    forecast_results = pd.DataFrame(index=future_dates, columns=['prediction'])
    
    last_row_full = df_features.iloc[-1]
    current_forecast_input = last_row_full[FEATURES].to_dict()
    
    for date in future_dates:
        # ... (Actualizar features de fecha, predicción y propagación del error) ...
        current_forecast_input['dayofweek'] = date.dayofweek
        current_forecast_input['month'] = date.month
        current_forecast_input['dayofyear'] = date.dayofyear
        current_forecast_input['is_weekend'] = 1 if date.dayofweek >= 5 else 0
        
        X_pred_forecast = pd.DataFrame([current_forecast_input], columns=FEATURES)
        next_revenue_forecast = model.predict(X_pred_forecast)[0]
        
        forecast_results.loc[date, 'prediction'] = next_revenue_forecast
        
        # PROPAGACIÓN REAL DEL ERROR para el Pronóstico Futuro
        for i in range(LAG_DAYS, 1, -1):
            current_forecast_input[f'revenue_lag_{i}'] = current_forecast_input[f'revenue_lag_{i-1}']
        current_forecast_input['revenue_lag_1'] = next_revenue_forecast 
        
    # 6. RETORNO DE RESULTADOS
    return df_features, forecast_results, future_dates, historical_mae, backtest_results, backtest_mae

@st.cache_data
def run_sarima_prediction(df, forecast_periods=30, order=(1, 0, 1), seasonal_order=(1, 1, 1, 7)):
    
    # 1. Preparar la serie de tiempo
    df_series = df.groupby('date')['total_sales'].sum().reset_index()
    df_series.columns = ['ds', 'y']
    df_series = df_series.set_index('ds')['y']
    
    # Datos de entrenamiento
    train_data = df_series.asfreq('D')
    train_data = train_data.ffill()

    # 2. Ajustar el modelo SARIMA
    try:
        model = sm.tsa.statespace.SARIMAX(
            train_data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            low_memory=True
        )
        # Usamos 'low_memory=True' para evitar errores de RAM en Streamlit Cloud
        model_fit = model.fit(disp=False, method='lbfgs', maxiter=200, tol=1e-5) 
        
        # 3. Pronóstico
        last_date = train_data.index.max()
        future_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods)
        
        forecast_obj = model_fit.get_forecast(steps=forecast_periods)
        forecast_mean = forecast_obj.predicted_mean
        
        # Crear DataFrame de resultados
        forecast_results = pd.DataFrame(forecast_mean.values, index=future_index, columns=['prediction'])
        
        # 4. Cálculo de Error (MAE histórico - solo para comparación)
        
        # ⚠️ CORRECCIÓN: Definimos la ventana de calentamiento. 
        # Esto elimina los primeros puntos que el modelo usa para iniciar la diferenciación.
        WARMUP_PERIOD = 28 
        
        # Haremos predicciones dentro de la muestra (in-sample)
        # Empezamos la predicción después del periodo de calentamiento.
        pred_in_sample = model_fit.predict(start=train_data.index[WARMUP_PERIOD], end=train_data.index[-1])
        
        # Calculamos el MAE comparando la predicción con el valor real
        mae_sarima = mean_absolute_error(train_data.iloc[WARMUP_PERIOD:], pred_in_sample)
        
        return forecast_results, mae_sarima, model_fit
        
    except Exception as e:
        # st.error(f"Error al ajustar el modelo SARIMA: {e}") # Mejor comentarlo para Streamlit Cloud
        st.error("Error al ajustar el modelo SARIMA. Revisa los logs.")
        return None, None, None

def sarima_backtest(series, order, seasonal_order, backtest_periods=90):
    """
    Rolling backtest for SARIMA using the same last N days as test set.
    """
    series = series.asfreq('D').ffill()
    n = len(series)
    train_size = n - backtest_periods

    preds = []
    test_index = []

    for i in range(train_size, n):
        train = series.iloc[:i]
        test = series.iloc[i:i + 1]

        try:
            model = sm.tsa.statespace.SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=1)
            preds.append(forecast.values[0])
            test_index.append(test.index[0])
        except Exception as e:
            preds.append(np.nan)
            test_index.append(test.index[0])

    preds = pd.Series(preds, index=test_index)
    aligned = pd.concat([series.loc[preds.index], preds], axis=1)
    aligned.columns = ['actual', 'predicted']

    mae = mean_absolute_error(aligned['actual'].dropna(), aligned['predicted'].dropna())

    return aligned, mae