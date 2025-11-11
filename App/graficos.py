import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prediction import run_sklearn_prediction, run_sarima_prediction, sarima_backtest
from analisis_costes import prepare_cost_analysis_df, analyze_costs

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
            df_features, forecast_results_rf, future_dates, historical_mae_rf, backtest_results, backtest_mae_rf = run_sklearn_prediction(df, forecast_periods=forecast_days)
            forecast_results_sarima, mae_sarima, model_fit_sarima = run_sarima_prediction(df, forecast_periods = forecast_days)
            df_series = df.groupby('date')['total_sales'].sum().asfreq('D').ffill()

            # Run SARIMA backtest on the last 90 days
            aligned_sarima, backtest_mae_sarima = sarima_backtest(
                df_series,
                order=(1, 0, 1),
                seasonal_order=(1, 1, 1, 7),
                backtest_periods=90
            )
            sarima_succeeded = forecast_results_sarima is not None and mae_sarima is not None
            st.header("Análisis de Rendimiento del Modelo")

            col_rf, col_sarima = st.columns(2)

            with col_rf:
                st.subheader("Random Forest (RF)")
                st.metric(label="MAE", value=f"{historical_mae_rf:,.0f}")
                st.metric(label="MAE Backtesting (Propagación)", value=f"{backtest_mae_rf:,.0f}", 
                        delta=f"{(backtest_mae_rf - 227000):,.0f} más que el error base") 

            with col_sarima:
                st.subheader("SARIMA (Estacionalidad 7)")
                if sarima_succeeded:
                    st.metric(label="MAE", value=f"{mae_sarima:,.0f}")
                    st.metric(label="MAE Backtesting (Propagación)", value=f"{backtest_mae_sarima:,.0f}")
                else:
                    st.error("No se pudo ajustar el modelo SARIMA. Revise la estacionalidad y los órdenes (p,d,q).")
            if forecast_results_sarima is not None:
                st.header("Pronóstico de Ventas (Comparación de Modelos)")

                # 1. Unir resultados en un solo DataFrame
                comparison_df = forecast_results_rf.rename(columns={'prediction': 'Random Forest'})
                comparison_df['SARIMA'] = forecast_results_sarima['prediction']
                comparison_df.index.name = 'Fecha'

                # 2. Crear gráfico
                st.line_chart(comparison_df)
        
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

                    st.subheader("💸 Gasto Fijo por Turno (8h) en Empleados por Franja")
                    fig_heatmap_gasto = px.imshow(
                        df_pivot_gasto, # Usamos el pivot del gasto que calculamos antes
                        color_continuous_scale='Blues', # Un color distinto para el gasto
                        labels=dict(x="Día de la Semana", y="Franja Horaria", color="Gasto Fijo por Turno (€)"),
                        text_auto=True,
                        aspect="auto",
                        title="Gasto Salarial Promedio por Franja y Día"
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
                            df_critical_candidates[['day_of_week_es', 'time_slot', 'revenue_segment', 'revenue_percentage', 'gasto_medio_empleados', 'sessions_count']]
                            .rename(columns={
                                'day_of_week_es': 'Día',
                                'time_slot': 'Franja Horaria',
                                'revenue_segment': 'Revenue Absoluto',
                                'revenue_percentage': '% Revenue Total',
                                'gasto_medio_empleados': 'Gasto Medio Empleados',
                                'sessions_count': '# Sesiones'
                            })
                            .style.format({
                                'Revenue Absoluto': "€ {:,.0f}",
                                '% Revenue Total': "{:.2f} %",
                                'Gasto Medio Empleados': "€ {:,.0f}"
                            }),
                            width='stretch'
                        )
                        
                    else:
                        st.info(f"Ninguna franja horaria tiene una contribución de Revenue inferior o igual al {threshold_percentage:.1f}% para el Cine {selected_cinema}.")


                    st.markdown("---")
                    st.markdown("**Las 5 franjas con menor Revenue:**")
                    
                    # Mostrar la tabla de las 5 peores franjas
                    st.dataframe(
                        top_savings_candidates[['day_of_week_es', 'time_slot', 'revenue_segment', 'revenue_percentage', 'gasto_medio_empleados', 'sessions_count']]
                        .rename(columns={
                            'day_of_week_es': 'Día',
                            'time_slot': 'Franja Horaria',
                            'revenue_segment': 'Revenue Absoluto',
                            'revenue_percentage': '% Revenue Total',
                            'gasto_medio_empleados': 'Gasto Medio Empleados',
                            'sessions_count': '# Sesiones'
                        })
                        .style.format({
                            'Revenue Absoluto': "€ {:,.0f}",
                            '% Revenue Total': "{:.2f} %",
                            'Gasto Medio Empleados': "€ {:,.0f}"
                        }),
                        width='stretch'
                    )