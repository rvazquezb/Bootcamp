import datetime
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prediction import predict_next_day_anomaly_sarima, prepare_ts_data, run_sklearn_prediction, run_sarima_prediction, sarima_backtest
from analisis_costes import prepare_cost_analysis_df, analyze_costs
from alerts import send_anomaly_alert

#Función para mostrar los KPIs principales
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
    
    #KPI 1: REVENUE TOTAL
    kpi_col1.metric(
        label="Revenue Total Anual",
        value=f"€ {total_revenue:,.0f}", 
        delta=f"{revenue_delta:,.0f}€ vs Target" 
    )
    
    #KPI 2: TICKETS VENDIDOS TOTALES
    kpi_col2.metric(
        label="Tickets Vendidos Totales",
        value=f"{total_tickets_sold:,.0f}", 
    )
    
    #KPI 3: OCUPACIÓN PROMEDIO
    kpi_col3.metric(
        label="Ocupación Promedio",
        value=f"{avg_occupation:.2f}%", 
        delta=f"{occupation_delta:.2f}% vs Target" 
    )

    #KPI 4: TASA DE CANCELACIÓN
    kpi_col4.metric(
        label="Tasa de Cancelación",
        value=f"{cancellation_rate_kpi:.2%}", 
    )

    st.markdown("---")
    
    col1, col2 = st.columns(2) 

    with col1:
        st.markdown("##### 🎯 YTD Revenue Goal")

        TARGET_REVENUE = 200000000.00
        REVENUE_ACTUAL = total_revenue
        
        progreso_porcentaje = (REVENUE_ACTUAL / TARGET_REVENUE) * 100
        progreso_porcentaje = round(progreso_porcentaje, 1) 
    
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

#Función para mostrar gráficos y análisis en distintas ventanas
def graficos(df):
    st.subheader("Análisis Detallado")
    user_role = st.session_state['user_role']
    available_cinemas = []
    tab_titles = ["Matriz de Cines", "Análisis de Precios", "Tendencia Semanal"]
    if 'Analista' in user_role:
        tab_titles.append("Predicción de Revenue")
        tab_titles.append("Ahorro de Costes")
    all_tabs = st.tabs(tab_titles)
    #Primera ventana: matriz de cines
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

    #Segunda ventana: análisis de precios
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

    #Tercera ventana: tendencia semanal
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
    #Si el usuario es analista (o admin), mostramos las ventanas adicionales
    if 'Analista' in user_role:
        df_analysis = prepare_cost_analysis_df(df)
        available_cinemas = sorted(df_analysis['cinema_code'].unique())

        #Cuarta ventana: predicción de revenue
        with all_tabs[3]:
            forecast_days = st.slider("Días a predecir", 7, 90, 30)
            df_features, forecast_results_rf, future_dates, historical_mae_rf, backtest_results, backtest_mae_rf = run_sklearn_prediction(df, forecast_periods=forecast_days)
            forecast_results_sarima, mae_sarima, model_fit_sarima, backtest_mae_sarima = run_sarima_prediction(df, forecast_periods = forecast_days)
            df_series = df.groupby('date')['total_sales'].sum().asfreq('D').ffill()

            sarima_succeeded = forecast_results_sarima is not None and mae_sarima is not None
            st.header("Análisis de Rendimiento del Modelo")

            col_rf, col_sarima = st.columns(2)

            with col_rf:
                st.subheader("Random Forest (RF)")
                st.metric(label="MAE", value=f"{historical_mae_rf:,.0f}")
                st.metric(label="MAE Backtesting (Propagación)", value=f"{backtest_mae_rf:,.0f}") 

            with col_sarima:
                st.subheader("SARIMA (Estacionalidad 7)")
                if sarima_succeeded:
                    st.metric(label="MAE", value=f"{mae_sarima:,.0f}")
                    st.metric(label="MAE Backtesting (Propagación)", value=f"{backtest_mae_sarima:,.0f}")
                else:
                    st.error("No se pudo ajustar el modelo SARIMA. Revise la estacionalidad y los órdenes (p,d,q).")
                    
            if forecast_results_sarima is not None:
                st.header("Pronóstico de Ventas (Comparación de Modelos)")
                forecast_results_sarima = np.maximum(0, forecast_results_sarima)
                comparison_df_forecast = forecast_results_rf.rename(columns={'prediction': 'Random Forest'})
                comparison_df_forecast['SARIMA'] = forecast_results_sarima['prediction']

                df_lookback = df_series.iloc[-30:]
                df_historical = df_lookback.rename('Datos Reales').to_frame()
                
                combined_df = pd.concat([df_historical, comparison_df_forecast], axis=1)
                
                combined_df.index.name = 'Fecha'

                st.line_chart(combined_df)

            st.markdown("---")
            st.header("🕵️ Detección de Anomalías (Análisis SARIMA)")
            st.markdown("Utiliza el modelo SARIMA para predecir las ventas del día siguiente basándose en el historial.")
            
            analysis_mode = st.radio(
                "Seleccione el alcance del análisis:",
                options=["Ventas Totales (Global)", "Por Cine (Específico)"],
                index=0, 
                horizontal=True
            )

            if analysis_mode == "Por Cine (Específico)":
                selected_cinema_anomaly = st.selectbox(
                    "Seleccione el Cine para el análisis de Anomalías:",
                    options=available_cinemas, 
                    index=0
                )
                
                series_to_analyze = prepare_ts_data(df, selected_cinema_anomaly)
                st.subheader(f"Análisis Específico: {selected_cinema_anomaly}")

            else: 
                series_to_analyze = df_series
                selected_cinema_anomaly = "TOTALES" 
                st.subheader("Análisis Global: Ventas Totales")

            if series_to_analyze.empty or len(series_to_analyze) < 7:
                st.warning("Datos insuficientes para realizar un análisis SARIMA.")
            else:
                max_date_available = series_to_analyze.index.max().date()
                min_date_available = series_to_analyze.index.min().date()

                if (max_date_available - min_date_available).days < 7:
                    st.warning(f"Datos insuficientes para el lookback de {7} días.")
                else:
                    selected_cutoff_date = st.date_input(
                        "Seleccione el último día de datos a USAR:",
                        value=max_date_available,
                        min_value=min_date_available + datetime.timedelta(days=7),
                        max_value=max_date_available
                    )
                    
                    selected_cutoff_date_dt = pd.to_datetime(selected_cutoff_date).date()

                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button(f"Evaluar Anomalía para el día: {selected_cutoff_date_dt + datetime.timedelta(days=1)}", width='stretch'):
                        with st.spinner(f"Ajustando modelo SARIMA con datos hasta {selected_cutoff_date_dt}..."):
                            
                            report, _, next_day_prediction, last_10_days_predictions = predict_next_day_anomaly_sarima( 
                                series=series_to_analyze, 
                                cut_off_date=selected_cutoff_date_dt
                            )
                        
                        st.markdown("---")
                        
                        if isinstance(report, dict):
                            
                            st.metric(
                                label=f"Ventas Predichas para {report['prediction_date']}",
                                value=f"€ {report['predicted_sales']:,.0f}",
                                delta=f"Umbral de Anomalía: € {report['anomaly_threshold']:,.0f}"
                            )
                            if report['is_anomaly']:
                                st.error(f"🚨 ¡ANOMALÍA DETECTADA! La desviación real ({report['deviation_from_prediction']:,.0f}€) superó el umbral.", icon="⚠️")
                                st.metric(
                                    label="Ventas Reales (Día de Predicción)",
                                    value=f"€ {report['actual_sales']:,.0f}"
                                )
                                prediction_date = pd.to_datetime(report['prediction_date'])
                                send_anomaly_alert(
                                    fecha=prediction_date, 
                                    valor_real=report['actual_sales'], 
                                    valor_predicho=report['predicted_sales'], 
                                    umbral=report['anomaly_threshold']
                                )
                            elif report['actual_sales'] == "N/A (Dato futuro)":
                                st.info("Predicción generada. Necesitará el dato real de mañana para la evaluación de anomalía.", icon="⏳")
                            else:
                                st.success(f"✅ Venta Real ({report['actual_sales']:,.0f}€) dentro del rango esperado.", icon="👍")
 
                            # 2. Obtener las 10 predicciones históricas y combinar con la predicción del día siguiente
                            last_day_of_plot = last_10_days_predictions.index.max()
                            next_day_date = last_day_of_plot + pd.Timedelta(days=1)
                            last_10_days_predictions = np.maximum(0, last_10_days_predictions)
                            future_dates = pd.date_range(
                                start=last_day_of_plot + pd.Timedelta(days=1), 
                                periods=1, 
                                freq='D'
                            )

                            dates_to_plot = last_10_days_predictions.index.tolist() + future_dates.tolist()
                            last_10_actual = series_to_analyze.loc[dates_to_plot].rename('Valor Real').to_frame().reset_index()
                            last_10_actual.columns = ['Fecha', 'Valor Real'] 
                            num_historical_predictions = len(last_10_days_predictions)

                            forecast_plot_data = pd.DataFrame({
                                'Fecha': last_10_days_predictions.index.tolist() + [next_day_date], 
                                'Valor': last_10_days_predictions.tolist() + [report['predicted_sales']], 
                                'Tipo': ['Predicción'] * num_historical_predictions + ['Predicción Día Siguiente']
                            })

                            forecast_plot_data['Fecha'] = pd.to_datetime(forecast_plot_data['Fecha'])

                            col_model, col_actual = st.columns(2)

                            with col_model:
                                fig_model = go.Figure()

                                fig_model.add_trace(go.Bar(
                                    x=forecast_plot_data[forecast_plot_data['Tipo'] == 'Predicción']['Fecha'],
                                    y=forecast_plot_data[forecast_plot_data['Tipo'] == 'Predicción']['Valor'],
                                    name='Predicción Histórica',
                                    marker_color='skyblue'
                                ))
                                
                                fig_model.add_trace(go.Bar(
                                    x=forecast_plot_data[forecast_plot_data['Tipo'] == 'Predicción Día Siguiente']['Fecha'],
                                    y=forecast_plot_data[forecast_plot_data['Tipo'] == 'Predicción Día Siguiente']['Valor'],
                                    name='Predicción Día Siguiente',
                                    marker_color='orange'
                                ))

                                fig_model.update_layout(
                                    title="Predicciones (10 Días Previos + Próximo Día)",
                                    height=400,
                                    showlegend=False
                                )

                                st.plotly_chart(fig_model, width='stretch')

                            with col_actual:
                                fig_actual = go.Figure()
                                
                                fig_actual.add_trace(go.Bar(
                                    x=last_10_actual['Fecha'],
                                    y=last_10_actual['Valor Real'],
                                    name='Ventas Reales',
                                    marker_color='darkgreen'
                                ))

                                fig_actual.update_layout(
                                    title="Ventas Reales (10 Días Previos + Próximo Día)",
                                    height=400,
                                    showlegend=False
                                )

                                st.plotly_chart(fig_actual, width='stretch') 
                            df_real = last_10_actual[['Fecha', 'Valor Real']] 
                            df_predicho = pd.DataFrame({
                                'Fecha': last_10_days_predictions.index.tolist() + [next_day_date], 
                                'Valor Predicho': last_10_days_predictions.tolist() + [report['predicted_sales']], 
                            })
                            df_predicho['Fecha'] = pd.to_datetime(df_predicho['Fecha'])

                            df_comparacion = pd.merge(
                                df_real, 
                                df_predicho, 
                                on='Fecha', 
                                how='inner' 
                            )

                            fig_comparison = px.line(
                                df_comparacion,
                                x='Fecha', 
                                y=['Valor Real', 'Valor Predicho'],
                                title="Valores Reales vs. Predicción SARIMA (Histórico)",
                                height=400
                            )

                            upper_bound = df_comparacion['Valor Predicho'] + report['anomaly_threshold']
                            lower_bound = df_comparacion['Valor Predicho'] - report['anomaly_threshold']

                            fig_comparison.add_trace(go.Scatter(
                                x=df_comparacion['Fecha'],
                                y=lower_bound,
                                line=dict(width=0,color='rgba(0, 150, 0, 0.2)'), 
                                name='Límite Inferior',
                                showlegend=False
                            ))

                            fig_comparison.add_trace(go.Scatter(
                                x=df_comparacion['Fecha'],
                                y=upper_bound,
                                line=dict(width=0, color = 'rgba(0, 128, 0, 0.2)'), 
                                fill='tonexty', 
                                fillcolor='rgba(0, 128, 0, 0.05)',
                                name='Banda de Umbral',
                                showlegend=False
                            ))

                            for trace in fig_comparison.data:
                                if trace.name in ('Valor Real', 'Valor Predicho'):
                                    trace.update(line=dict(width=3), zorder=10) 

                            fig_comparison.update_layout(
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )

                            fig_comparison.update_layout(
                                yaxis_title="", 
                                xaxis_title="",
                                legend_title="Tipo de Valor",
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            st.plotly_chart(fig_comparison, width='stretch')
                        else:
                            st.error(report)

        #Quinta ventana: análisis de ahorro de costes               
        with all_tabs[4]:
            selected_cinema = st.selectbox(
                "Seleccione el Cine para analizar:",
                options=available_cinemas,
                index=0 if available_cinemas else None
            )

            if selected_cinema:
                
                df_impact, total_revenue = analyze_costs(df_analysis, selected_cinema)
                
                if df_impact.empty:
                    st.warning("No hay datos de Revenue para el cine seleccionado.")
                else:
                    st.markdown(f"**Revenue Total Histórico del Cine {selected_cinema}:** € {total_revenue:,.0f}")
                    st.markdown("---")
                    
                    st.subheader("Revenue Total por Franja Horaria y Día de la Semana")
                    
                    fig_heatmap_revenue = px.imshow(
                        df_impact.pivot(index='time_slot', columns='day_of_week_es', values='revenue_segment').fillna(0),
                        color_continuous_scale='YlOrRd', 
                        labels=dict(x="Día de la Semana", y="Franja Horaria", color="Revenue (€)"),
                        text_auto=True,
                        aspect="auto",
                        title="Ingresos Históricos por Franja"
                    )
                    df_pivot_gasto = df_impact.pivot(index='time_slot', columns='day_of_week_es', values='gasto_medio_empleados').fillna(0)
                    fig_heatmap_revenue.update_traces(
                        customdata=df_pivot_gasto.round(0).values,
                        hovertemplate="<b>%{y} - %{x}</b><br>Revenue: €%{z:,.0f}<br>**Gasto Total Acumulado:** €%{customdata:,.0f}<extra></extra>"
                    )
                    fig_heatmap_revenue.update_layout(
                        xaxis_title=None, 
                        yaxis_title=None,
                        height=400
                    )
                    
                    st.plotly_chart(fig_heatmap_revenue, width='stretch')
                    
                    st.markdown("---")

                    st.subheader("Gasto Fijo por Turno (8h) en Empleados por Franja")
                    fig_heatmap_gasto = px.imshow(
                        df_pivot_gasto, 
                        color_continuous_scale='Blues', 
                        labels=dict(x="Día de la Semana", y="Franja Horaria", color="Gasto Fijo por Turno (€)"),
                        text_auto=True,
                        aspect="auto",
                        title="Gasto Salarial Total por Franja y Día"
                    )
                    fig_heatmap_gasto.update_layout(xaxis_title=None, yaxis_title=None, height=400)
                    st.plotly_chart(fig_heatmap_gasto, width='stretch')

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

                    df_savings = df_impact.sort_values('revenue_percentage', ascending=True)
                    
                    top_savings_candidates = df_savings.head(5).copy()
                    
                    df_critical_candidates = df_savings[(df_savings['revenue_percentage'] <= threshold_percentage) | (df_savings['revenue_segment'] <= df_savings['gasto_medio_empleados'])].copy()

                    if not df_critical_candidates.empty:
                        total_lost_percentage = df_critical_candidates['revenue_percentage'].sum()
                        total_savings = df_critical_candidates['gasto_medio_empleados'].sum() - df_critical_candidates['revenue_segment'].sum()
                        hay_franjas_perdedoras = (df_critical_candidates['revenue_segment'] < df_critical_candidates['gasto_medio_empleados']).any()
                        if hay_franjas_perdedoras:
                            st.metric(
                                label=f"Ahorro de coste si se cierran las siguientes franjas",
                                value=f"{total_savings:,.0f} €"
                            )
                            st.warning("🚨 Algunas franjas tienen un gasto en empleados mayor que su revenue.")
                        debajo_threshold = (df_savings['revenue_percentage'] <= threshold_percentage).any()
                        if debajo_threshold:
                            st.metric(
                                label=f"Pérdida de Revenue si se cierran las franjas con contribución <= {threshold_percentage:.1f}%",
                                value=f"{total_lost_percentage:.2f} %"
                            )
                            st.warning(f"🚨 **CANDIDATAS CLAVE AL CIERRE (Contribución < {threshold_percentage:.1f}%):**")
                        
                        st.dataframe(
                            df_critical_candidates[['day_of_week_es', 'time_slot', 'revenue_segment', 'revenue_percentage', 'gasto_medio_empleados', 'sessions_count']]
                            .rename(columns={
                                'day_of_week_es': 'Día',
                                'time_slot': 'Franja Horaria',
                                'revenue_segment': 'Revenue Absoluto',
                                'revenue_percentage': '% Revenue Total',
                                'gasto_medio_empleados': 'Gasto Total Empleados',
                                'sessions_count': '# Sesiones'
                            })
                            .style.format({
                                'Revenue Absoluto': "€ {:,.0f}",
                                '% Revenue Total': "{:.2f} %",
                                'Gasto Total Empleados': "€ {:,.0f}"
                            }),
                            width='stretch'
                        )
                        
                    else:
                        st.info(f"Ninguna franja horaria tiene una contribución de Revenue inferior o igual al {threshold_percentage:.1f}% para el Cine {selected_cinema}.")


                    st.markdown("---")
                    st.markdown("**Las 5 franjas con menor Revenue:**")
                    
                    st.dataframe(
                        top_savings_candidates[['day_of_week_es', 'time_slot', 'revenue_segment', 'revenue_percentage', 'gasto_medio_empleados', 'sessions_count']]
                        .rename(columns={
                            'day_of_week_es': 'Día',
                            'time_slot': 'Franja Horaria',
                            'revenue_segment': 'Revenue Absoluto',
                            'revenue_percentage': '% Revenue Total',
                            'gasto_medio_empleados': 'Gasto Total Empleados',
                            'sessions_count': '# Sesiones'
                        })
                        .style.format({
                            'Revenue Absoluto': "€ {:,.0f}",
                            '% Revenue Total': "{:.2f} %",
                            'Gasto Total Empleados': "€ {:,.0f}"
                        }),
                        width='stretch'
                    )