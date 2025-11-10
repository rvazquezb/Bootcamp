import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import plotly.express as px
import plotly.graph_objects as go
import bcrypt
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error

# Conexion a Neon
NEON_DATABASE_URL = "postgresql://neondb_owner:npg_lz1fJwWeEr7n@ep-aged-flower-aba6hlv1-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require" 
TABLE_NAME = "ventas_cine_final" 

# Inicializar el estado de sesión si no existe
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
    st.session_state['user_role'] = None
    st.session_state['username'] = None
    st.session_state['active_view'] = 'dashboard'
    st.session_state['confirm_insert'] = False

# Función para verificar las credenciales 
def authenticate_user(username, password):
    try:
        engine = create_engine(NEON_DATABASE_URL)
        
        query = text(
            "SELECT u.password_hash, r.role_name "
            "FROM users u JOIN group_roles gr ON u.group_id = gr.id_group JOIN roles r ON gr.id_role = r.id "
            "WHERE u.username = :username "
            "ORDER BY gr.id_role"
        )
        
        with engine.connect() as connection:
            result = connection.execute(query, {"username": username}).fetchall()
            
            if result:
                password_hash = result[0][0].encode('utf-8')
                roles_list = []

                for row in result:
                    role_name = row[1] 
                    roles_list.append(role_name)
                
                if bcrypt.checkpw(password.encode('utf-8'), password_hash):
                    return True, roles_list
                else:
                    return False, None
            else:
                return False, None 

    except Exception as e:
        st.error(f"Error de conexión o base de datos: {e}")
        return False, None

# Función para cargar los datos
@st.cache_data
def load_data():
    try:
        engine = create_engine(NEON_DATABASE_URL)
        
        df = pd.read_sql_table(TABLE_NAME, con=engine)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        return df
    except Exception as e:
        st.error(f"❌ Error al conectar o cargar datos desde Neon (Cloud).")
        st.error("Verifica que la cadena de conexión es correcta y que la base de datos Neon está activa.")
        st.error(f"Detalle del error: {e}")
        return pd.DataFrame() 

def show_login_form():
    
    st.title("🔐 Acceso a la aplicación")
    
    with st.form("login_form"):
        st.markdown("Introduce tus credenciales:")
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Iniciar Sesión")
        
        if submitted:
            success, role = authenticate_user(username, password)
            
            if success:
                st.session_state['authenticated'] = True
                st.session_state['user_role'] = role
                st.session_state['username'] = username
                st.session_state['active_view'] = 'dashboard'
                st.session_state['confirm_insert'] = False
                
                st.success(f"¡Bienvenido/a {username}!")
                
                st.rerun() 
                return 
            else:
                st.error("Usuario o contraseña incorrectos.")

def kpis(df):
    st.markdown("### 🎯 KPIs")
                    
    # Métrica de Revenue Total
    total_revenue = df['total_sales'].sum()
    
    # Métrica de Tickets Vendidos
    total_tickets_sold = df['tickets_sold'].sum()

    # Métrica de Ocupación Promedio
    avg_occupation = df['occu_perc'].mean()
    
    # Métrica de Tasa de Cancelación
    total_tickets_out = df['tickets_out'].sum()
    cancellation_rate_kpi = (total_tickets_out / total_tickets_sold) if total_tickets_sold > 0 else 0
    
    revenue_delta = total_revenue - 200000000
    occupation_delta = avg_occupation - 25.0

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    # KPI 1: REVENUE TOTAL
    kpi_col1.metric(
        label="Revenue Total Anual",
        value=f"€ {total_revenue:,.0f}", 
        delta=f"{revenue_delta:,.0f}€ vs Target" 
    )
    
    # KPI 2: TICKETS VENDIDOS TOTALES
    kpi_col2.metric(
        label="Tickets Vendidos Totales",
        value=f"{total_tickets_sold:,.0f}", 
    )
    
    # KPI 3: OCUPACIÓN PROMEDIO
    kpi_col3.metric(
        label="Ocupación Promedio",
        value=f"{avg_occupation:.2f}%", 
        delta=f"{occupation_delta:.2f}% vs Target" 
    )

    # KPI 4: TASA DE CANCELACIÓN
    
    kpi_col4.metric(
        label="Tasa de Cancelación",
        value=f"{cancellation_rate_kpi:.2%}", 
    )

    st.markdown("---")
    
    col1, col2 = st.columns(2) 

    with col1:
        st.markdown("##### 🎯 YTD Revenue Goal")

        TARGET_REVENUE = 200000000.00
        REVENUE_ACTUAL = df['total_sales'].sum()
        
        progreso_porcentaje = (REVENUE_ACTUAL / TARGET_REVENUE) * 100
        progreso_porcentaje = round(progreso_porcentaje, 1) 
        
        valor_indicador = min(REVENUE_ACTUAL, TARGET_REVENUE)
    
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = REVENUE_ACTUAL,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Revenue Total Actual (€)", 'font': {'size': 14}},
            delta = {'reference': TARGET_REVENUE, 'relative': False, 'valueformat': '$,.0f'},
            number = {'valueformat': '$,.0f'},
            gauge = {
                'shape': "angular",
                'axis': {'range': [None, TARGET_REVENUE * 1.5], 'tickwidth': 1, 'tickcolor': "darkblue", 'tickformat': '$,.0f'},
                'bar': {'color': "#006400"}, 
                'steps': [
                    {'range': [0, TARGET_REVENUE * 0.50], 'color': "lightcoral"}, 
                    {'range': [TARGET_REVENUE * 0.50, TARGET_REVENUE * 0.90], 'color': "lightgray"}, 
                    {'range': [TARGET_REVENUE * 0.90, TARGET_REVENUE * 1.5], 'color': "lightgreen"} 
                ],
                'threshold': { 
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': TARGET_REVENUE
                }
            }
        ))
        fig_gauge.update_layout(height=300)

        st.plotly_chart(fig_gauge, width='stretch') 
        st.metric(label="Progreso %", value=f"{progreso_porcentaje}%", delta=f"Target: {TARGET_REVENUE:,.0f}€")

    with col2:
        st.markdown("##### 🎯 YTD Occupancy Goal")

        TARGET_OCCUPATION = 25.0

        OCCUPATION_ACTUAL = df['occu_perc'].mean()
        OCCUPATION_ACTUAL_ROUNDED = round(OCCUPATION_ACTUAL, 2)
        
        progreso_porcentaje_occ = (OCCUPATION_ACTUAL / TARGET_OCCUPATION) * 100
        progreso_porcentaje_occ_rounded = round(progreso_porcentaje_occ, 1)

        fig_gauge_occ = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = OCCUPATION_ACTUAL_ROUNDED,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Ocupación Promedio (%)", 'font': {'size': 14}},
            delta = {'reference': TARGET_OCCUPATION, 'relative': False, 'valueformat': '.2f'},
            number = {'suffix': "%", 'valueformat': '.2f'}, 
            gauge = {
                'shape': "angular",
                'axis': {'range': [None, TARGET_OCCUPATION * 1.5], 'tickwidth': 1, 'tickcolor': "darkblue", 'tickformat': '.0f'}, # Rango hasta 1.5 veces el target
                'bar': {'color': "#006400"}, 
                'steps': [
                    {'range': [0, TARGET_OCCUPATION * 0.5], 'color': "lightcoral"}, 
                    {'range': [TARGET_OCCUPATION * 0.5, TARGET_OCCUPATION * 0.95], 'color': "lightgray"}, 
                    {'range': [TARGET_OCCUPATION * 0.95, TARGET_OCCUPATION * 1.5], 'color': "lightgreen"} 
                ],
                'threshold': { 
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': TARGET_OCCUPATION
                }
            }
        ))
        fig_gauge_occ.update_layout(height=300)

        st.plotly_chart(fig_gauge_occ, width='stretch')
        st.metric(label="Progreso %", value=f"{progreso_porcentaje_occ_rounded}%", delta=f"Target: {TARGET_OCCUPATION:.0f}%")

def graficos(df):
    st.subheader("Análisis Detallado")
    user_role = st.session_state['user_role']
    tab_titles = ["Matriz de Cines", "Análisis de Precios", "Tendencia Semanal"]
    if 'Analista' in user_role:
        tab_titles.append("Predicción de Revenue")
        tab_titles.append("Ahorro de Costes")
    all_tabs = st.tabs(tab_titles)
    with all_tabs[0]:
        df_agg = df.groupby('cinema_code').agg(
            total_tickets_sold=('tickets_sold', 'sum'),
            total_revenue=('total_sales', 'sum'),
            average_occupation=('occu_perc', 'mean'),
            total_tickets_out=('tickets_out', 'sum')
        ).reset_index()

        df_agg['cancellation_rate'] = np.where(
            df_agg['total_tickets_sold'] > 0,
            (df_agg['total_tickets_out'] / df_agg['total_tickets_sold']),
            0  
        )
        
        df_metrics = df_agg[[
            'cinema_code', 
            'total_tickets_sold', 
            'total_revenue', 
            'average_occupation',
            'cancellation_rate'
        ]].copy()
        
        df_metrics.columns = [
            'Cine', 
            'Tickets Vendidos', 
            'Recaudación Total (€)', 
            'Ocupación Promedio (%)',
            'Tasa de Cancelación (%)'
        ]
        
        st.dataframe(
            df_metrics.style.format({
                'Recaudación Total (€)': "€{:,.2f}",
                'Ocupación Promedio (%)': "{:.2f}%",
                'Tasa de Cancelación': "{:.2%}"
            }), 
            width='stretch',
            hide_index=True
        )

    with all_tabs[1]:
        bins = [0, 5, 10, np.inf]
        labels = ["<5€", "5€ a 10€", ">10€"]

        df['price_range'] = pd.cut(
            df['ticket_price'], 
            bins=bins, 
            labels=labels, 
            right=True, 
            include_lowest=True 
        )

        df_price_agg = df.groupby('price_range', observed=True)['tickets_sold'].sum().reset_index()
        df_price_agg.columns = ['Rango de Precio', 'Tickets Vendidos']

        fig_price = px.bar(
            df_price_agg,
            x='Rango de Precio',
            y='Tickets Vendidos',
            color='Rango de Precio'
        )
        
        st.plotly_chart(fig_price, width='stretch')

    with all_tabs[2]:
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

        df_weekly_revenue = df.groupby('week_of_year', observed=True)['total_sales'].sum().reset_index()
        df_weekly_revenue.columns = ['Semana del Año', 'Ingresos Totales (€)']

        fig_weekly = px.line(
            df_weekly_revenue,
            x='Semana del Año',
            y='Ingresos Totales (€)',
            markers=True
        )

        fig_weekly.update_xaxes(tick0=1, dtick=4) 
        
        st.plotly_chart(fig_weekly, width='stretch')
    if 'Analista' in user_role:
        df_analysis = prepare_cost_analysis_df(df)
        available_cinemas = sorted(df_analysis['cinema_code'].unique())
        with all_tabs[3]:
            forecast_days = st.slider("Días a predecir", 7, 90, 30)
            df_features, forecast_results, future_dates, historical_mae, backtest_results, backtest_mae = run_sklearn_prediction(df, forecast_periods=forecast_days)

            df_historic = df_features.rename(columns={'ds': 'Fecha', 'y': 'Revenue'}).set_index('Fecha')['Revenue']
            df_forecast = forecast_results.rename(columns={'prediction': 'Revenue'})['Revenue']

            df_plot = pd.concat([df_historic, df_forecast])
            
            fig = px.line(
                df_plot.reset_index().rename(columns={'index': 'Fecha'}),
                x='Fecha',
                y='Revenue',
                title=f'Revenue Histórico y Predicción ({forecast_days} días)'
            )
            start_date_str = future_dates.min().strftime('%Y-%m-%d')
            fig.add_shape(
                type='line',
                xref='x', 
                yref='paper', 
                x0=start_date_str, 
                y0=0, 
                x1=start_date_str, 
                y1=1, 
                line=dict(
                    color='red',
                    width=2,
                    dash='dash', 
                )
            )
        
            fig.add_vrect(
                x0=start_date_str, 
                x1=df_plot.index.max().strftime('%Y-%m-%d'), 
                fillcolor="rgba(255, 0, 0, 0.1)",
                layer="below",
                line_width=0,
                name="Zona de Predicción"
            )
            
            st.plotly_chart(fig, width='stretch')

            st.metric(
                label="Error Absoluto Medio (MAE)",
                value=f"€ {historical_mae:,.0f}"
            )

            st.metric(
                label="MAE de Backtesting (últimos 90 días)",
                value=f"€ {backtest_mae:,.0f}"
            )

            if backtest_mae > historical_mae * 1.5:
                st.warning("El error de Backtesting es significativamente mayor que el error de entrenamiento. ¡Cuidado con el sobreajuste!")
        
        with all_tabs[4]:
            selected_cinema = st.selectbox(
                "Seleccione el Cine para analizar:",
                options=available_cinemas,
                index=0
            )

            if selected_cinema:
                
                # Ejecutar el análisis
                df_impact, total_revenue = analyze_costs(df_analysis, selected_cinema)
                
                if df_impact.empty:
                    st.warning("No hay datos de Revenue para el cine seleccionado.")
                else:
                    st.markdown(f"**Revenue Total Histórico del Cine {selected_cinema}:** € {total_revenue:,.0f}")
                    st.markdown("---")
                    
                    st.subheader("Revenue Total por Franja Horaria y Día de la Semana")
                    
                    fig_heatmap = px.imshow(
                        df_impact.pivot(index='time_slot', columns='day_of_week_es', values='revenue_segment').fillna(0),
                        color_continuous_scale='YlOrRd', 
                        labels=dict(x="Día de la Semana", y="Franja Horaria", color="Revenue (€)"),
                        text_auto=True,
                        aspect="auto"
                    )
                    fig_heatmap.update_layout(
                        xaxis_title=None, 
                        yaxis_title=None,
                        height=400
                    )
                    
                    st.plotly_chart(fig_heatmap, width='stretch')
                    
                    st.markdown("---")

                    st.subheader("Tabla de Oportunidades de Ahorro")
                    st.markdown("Estas franjas tienen el **menor impacto en el Revenue**. Son las candidatas principales al cierre.")

                    threshold_percentage = st.slider(
                        "Seleccione el Umbral de Contribución al Revenue",
                        min_value=0.5,
                        max_value=5.0,
                        value=2.0, 
                        step=0.5,
                        format="%.1f %%"
                    )

                    # 1. Ordenar por el porcentaje de Revenue (ascendente)
                    df_savings = df_impact.sort_values('revenue_percentage', ascending=True)
                    
                    # 2. Seleccionar las N peores franjas (ej. las 5 peores, para la tabla)
                    top_savings_candidates = df_savings.head(5).copy()
                    
                    # 3. Identificar Candidatas Clave para el Cierre (Ahorro Máximo)
                    df_critical_candidates = df_savings[df_savings['revenue_percentage'] <= threshold_percentage].copy()

                    if not df_critical_candidates.empty:
                        total_lost_percentage = df_critical_candidates['revenue_percentage'].sum()
                        
                        st.metric(
                            label=f"Pérdida de Revenue si se cierran las franjas con contribución <= {threshold_percentage:.1f}%",
                            value=f"{total_lost_percentage:.2f} %",
                            delta="Esta es la pérdida de ingresos que se sacrificaría para maximizar el ahorro en costes fijos."
                        )
                        
                        st.warning(f"🚨 **CANDIDATAS CLAVE AL CIERRE (Contribución < {threshold_percentage:.1f}%):**")
                        
                        # Mostrar la tabla con las candidatas que cumplen el umbral
                        st.dataframe(
                            df_critical_candidates[['day_of_week_es', 'time_slot', 'revenue_segment', 'revenue_percentage', 'sessions_count']]
                            .rename(columns={
                                'day_of_week_es': 'Día',
                                'time_slot': 'Franja Horaria',
                                'revenue_segment': 'Revenue Absoluto',
                                'revenue_percentage': '% Revenue Total',
                                'sessions_count': '# Sesiones'
                            })
                            .style.format({
                                'Revenue Absoluto': "€ {:,.0f}",
                                '% Revenue Total': "{:.2f} %"
                            }),
                            width='stretch'
                        )
                        
                    else:
                        st.info(f"Ninguna franja horaria tiene una contribución de Revenue inferior o igual al {threshold_percentage:.1f}% para el Cine {selected_cinema}.")


                    st.markdown("---")
                    st.markdown("**Las 5 franjas con menor Revenue:**")
                    
                    # Mostrar la tabla de las 5 peores franjas
                    st.dataframe(
                        top_savings_candidates[['day_of_week_es', 'time_slot', 'revenue_segment', 'revenue_percentage', 'sessions_count']]
                        .rename(columns={
                            'day_of_week_es': 'Día',
                            'time_slot': 'Franja Horaria',
                            'revenue_segment': 'Revenue Absoluto',
                            'revenue_percentage': '% Revenue Total',
                            'sessions_count': '# Sesiones'
                        })
                        .style.format({
                            'Revenue Absoluto': "€ {:,.0f}",
                            '% Revenue Total': "{:.2f} %"
                        }),
                        width='stretch'
                    )

def insert_data(date, film_code, cinema_code, ticket_price, tickets_sold, ticket_use, show_time, tickets_out, capacity):
    try:
        engine = create_engine(NEON_DATABASE_URL)
        
        total_sales = ticket_price * tickets_sold
        occu_perc = (ticket_use / capacity) * 100 
        month = date.month
        quarter = (date.month - 1) // 3 + 1 
        day = date.day
        day_name = date.strftime("%A")

        insert_query = text(
            "INSERT INTO ventas_cine_final (film_code, cinema_code, total_sales, tickets_sold, tickets_out, show_time, occu_perc, ticket_price, ticket_use, capacity, date, month, quarter, day, day_name) "
            "VALUES (:film_code, :cinema_code, :total_sales, :tickets_sold, :tickets_out, :show_time, :occu_perc, :ticket_price, :ticket_use, :capacity, :date, :month, :quarter, :day, :day_name)"
        )
        
        with engine.connect() as connection:
            connection.execute(insert_query, {
                "film_code": film_code,
                "cinema_code": cinema_code,
                "total_sales": total_sales,
                "tickets_sold": tickets_sold,
                "tickets_out": tickets_out,
                "show_time": show_time,
                "occu_perc": occu_perc,
                "ticket_price": ticket_price,
                "ticket_use": ticket_use,
                "capacity": capacity,
                "date": date,
                "month": month,
                "quarter": quarter,
                "day": day,
                "day_name": day_name
            })
            connection.commit() 

        st.cache_data.clear() 
        return True
        
    except Exception as e:
        st.error(f"❌ Error al insertar datos: {e}")
        return False

def change_view(view_name):
    st.session_state['active_view'] = view_name

def create_features(df, lag=14):

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
    LAG_DAYS = 14
    
    df_features = create_features(df, lag=LAG_DAYS)
    
    TARGET = 'y'
    FEATURES = [col for col in df_features.columns if col not in ['ds', TARGET]]

    # Tamaño de los datos de entrenamiento (restantes después del backtesting)
    train_size = len(df_features) - backtest_periods
    
    # Datos de entrenamiento para el modelo inicial
    df_train = df_features.iloc[:train_size].copy()
    
    # Datos de validación/backtesting
    df_backtest_actuals = df_features.iloc[train_size:].copy()
    
    X_train = df_train[FEATURES]
    y_train = df_train[TARGET]
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)

    backtest_results = pd.DataFrame(
        index=df_backtest_actuals['ds'], 
        columns=['actual', 'prediction']
    )
    backtest_results['actual'] = df_backtest_actuals['y'].values
    
    # Tomar la última fila de entrenamiento como punto de partida para el bucle
    last_train_row = df_train.iloc[-1]
    current_input = last_train_row[FEATURES].to_dict()

    # Bucle de predicción acumulativa (Walk-Forward)
    for i, date in enumerate(df_backtest_actuals['ds']):
        
        # 3a. Actualizar Features de Fecha (día de la semana, etc.)
        current_input['dayofweek'] = date.dayofweek
        current_input['month'] = date.month
        current_input['dayofyear'] = date.dayofyear
        current_input['is_weekend'] = 1 if date.dayofweek >= 5 else 0
        
        # 3b. Predecir el siguiente valor
        X_pred = pd.DataFrame([current_input], columns=FEATURES)
        next_revenue = model.predict(X_pred)[0]
        
        backtest_results.loc[date, 'prediction'] = next_revenue
        
        # 3c. Acumular el Error (Actualizar los Lags)
        # Aquí propagamos el error: usamos la PREDICCIÓN (next_revenue) como
        # el lag_1 para la siguiente iteración.
        for j in range(LAG_DAYS, 1, -1):
            current_input[f'revenue_lag_{j}'] = current_input[f'revenue_lag_{j-1}']
        current_input['revenue_lag_1'] = next_revenue # <--- Clave de la acumulación
        
    # ----------------------------------------------------
    # 4. CALCULAR MÉTRICAS DE BACKTESTING
    # ----------------------------------------------------
    backtest_mae = mean_absolute_error(
        backtest_results['actual'].values, 
        backtest_results['prediction'].values
    )

    X = df_features[FEATURES]
    y = df_features[TARGET]
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
    model.fit(X, y)
    
    y_pred_history = model.predict(X)
    mae = mean_absolute_error(y, y_pred_history)
    
    last_date = df_features['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods)
    
    forecast_results = pd.DataFrame(index=future_dates, columns=['prediction'])
    
    last_row = df_features.iloc[-1]
    current_input = last_row[FEATURES].to_dict()
    
    for date in future_dates:
        current_input['dayofweek'] = date.dayofweek
        current_input['month'] = date.month
        current_input['dayofyear'] = date.dayofyear
        current_input['is_weekend'] = 1 if date.dayofweek >= 5 else 0
        
        X_pred = pd.DataFrame([current_input], columns=FEATURES)
        next_revenue = model.predict(X_pred)[0]
        
        forecast_results.loc[date, 'prediction'] = next_revenue
        
        for i in range(LAG_DAYS, 1, -1):
            current_input[f'revenue_lag_{i}'] = current_input[f'revenue_lag_{i-1}']
        current_input['revenue_lag_1'] = next_revenue 
        
    return df_features, forecast_results, future_dates, mae, backtest_results, backtest_mae

def map_show_time_to_slot(show_time):
    if 1 <= show_time <= 15:
        return '01 - 15 (Mañana)'
    elif 16 <= show_time <= 40:
        return '16 - 40 (Tarde)'
    elif 41 <= show_time <= 60:
        return '41 - 60 (Noche)'
    else:
        return 'Otro'

def prepare_cost_analysis_df(df):
    df_analysis = df.copy()
    df_analysis['time_slot'] = df_analysis['show_time'].apply(map_show_time_to_slot)
    df_analysis['day_of_week'] = df_analysis['date'].dt.day_name()
    
    day_map_es = {
        'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles', 
        'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    df_analysis['day_of_week_es'] = df_analysis['day_of_week'].map(day_map_es)
    
    return df_analysis

def analyze_costs(df, cinema_id):
    
    # Filtrar por el cine seleccionado
    df_cinema = df[df['cinema_code'] == cinema_id].copy()

    # Calcular el Revenue total de ese cine
    total_revenue_cinema = df_cinema['total_sales'].sum()
    
    if total_revenue_cinema == 0:
        return pd.DataFrame(), 0 
        
    # Agrupar por Día de la Semana y Franja Horaria
    df_grouped = df_cinema.groupby(['day_of_week_es', 'time_slot']).agg(
        revenue_segment=('total_sales', 'sum'),
        sessions_count=('date', 'count')
    ).reset_index()
    
    # Calcular %
    df_grouped['revenue_percentage'] = (df_grouped['revenue_segment'] / total_revenue_cinema) * 100
    
    # Ordenar los días para visualización
    day_order_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    df_grouped['day_of_week_es'] = pd.Categorical(df_grouped['day_of_week_es'], categories=day_order_es, ordered=True)
    df_grouped = df_grouped.sort_values(by='day_of_week_es')
    
    return df_grouped, total_revenue_cinema

def main():
    if not st.session_state['authenticated']:
        show_login_form()
    else:
        def logout():
            st.session_state['authenticated'] = False
            st.session_state['user_role'] = None
            st.session_state['username'] = None
            st.session_state['confirm_insert'] = False
            
        st.sidebar.button("Cerrar Sesión", on_click=logout)
        st.sidebar.markdown(f"**Rol:** {st.session_state['user_role'][0]}")
        df = load_data()
        
        # Control de vista basado en el rol del usuario
        st.set_page_config(page_title="Aussie Cines Dashboard", layout="wide")
        st.title("🎬 Aussie Cines") 
        st.subheader(f"👋 Bienvenid@ {st.session_state['username']}")
        
        user_role = st.session_state['user_role']

        st.markdown("---")

        if 'Exec' in user_role:  
            if 'Administrador' in user_role: 
                st.warning("⚠️ VISTA COMPLETA: Administrador")
                st.sidebar.button("📝 Insertar Datos", 
                        on_click=change_view, 
                        args=('insert_form',), 
                        type="primary") 
    
                st.sidebar.button("📊 Ver Dashboard", 
                        on_click=change_view, 
                        args=('dashboard',))
                
                active_view = st.session_state['active_view']
                if active_view == 'insert_form':
                    st.header("Herramientas de Administración: Inserción de Datos")

                    with st.form(key='data_entry_form', clear_on_submit=True):
                        st.subheader("📝 Insertar Nueva Fila de Datos")
                        
                        col_a, col_b, col_c, col_d = st.columns(4)
                        
                        # Primera fila
                        date_input = col_a.date_input("Fecha", value="today")
                        cinema_code_input = col_b.number_input("Código de Cine", min_value=0, step=1)
                        film_code_input = col_c.number_input("Código de Película", min_value=0, step=1)
                        show_time_input = col_d.number_input("Sesión", min_value=0, step=1)
                        
                        # Segunda fila
                        col_d, col_e, col_f, col_g, col_h = st.columns(5)
                        capacity_input = col_d.number_input("Capacidad", min_value=1, step=1)
                        tickets_sold_input = col_e.number_input("Tickets Vendidos", min_value=0, step=1)
                        tickets_out_input = col_f.number_input("Tickets Anulados", min_value=0, step=1)
                        ticket_use_input = col_g.number_input("Tickets Usados", min_value=0, step=1)
                        ticket_price_input = col_h.number_input("Precio del Ticket (€)", min_value=0.0, step=0.01, format="%.2f")

                        submit_button = st.form_submit_button(label='💾 Insertar Fila')

                        if submit_button:
                            if insert_data(date_input, film_code_input, cinema_code_input, ticket_price_input, tickets_sold_input, ticket_use_input, show_time_input, tickets_out_input, capacity_input):
                                st.success(f"Fila insertada correctamente para el cine {cinema_code_input} en la fecha {date_input}.")
                            else:
                                st.error("No se pudo insertar la fila. Revise los datos o la consola de errores.")
                else:
                    if not df.empty:
                        kpis(df)
                        st.markdown("---")
                        graficos(df)
                    else:
                        st.error("No fue posible cargar los datos desde Neon.")
            else:
                if not df.empty:
                    kpis(df)
                    st.markdown("---")
                    graficos(df)
                else:
                    st.error("No fue posible cargar los datos desde Neon.")
        elif 'Analista' in user_role:
            if not df.empty:
                graficos(df)
            else:
                st.error("No fue posible cargar los datos desde Neon.")
main()