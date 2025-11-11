import pandas as pd

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
    HOURS_PER_SHIFT = 8 
    
    # 1. Filtrar por el cine seleccionado
    df_cinema = df[df['cinema_code'] == cinema_id].copy()

    df_cinema['gasto_turno_fijo'] = (
        df_cinema['n_empleados'] * df_cinema['salario_hora'] * HOURS_PER_SHIFT
    )
    
    total_revenue_cinema = df_cinema['total_sales'].sum()
    
    if total_revenue_cinema == 0:
        return pd.DataFrame(), 0 
        
    df_grouped = df_cinema.groupby(['day_of_week_es', 'time_slot']).agg(
        revenue_segment=('total_sales', 'sum'),
        sessions_count=('date', 'count'),
        dias_con_sesiones=('date', 'nunique') 
    ).reset_index()

    # Gasto de un solo turno (fijo por cine y franja - el nocturno tiene plus)
    df_cinema['gasto_turno_unico'] = df_cinema['n_empleados'] * df_cinema['salario_hora'] * HOURS_PER_SHIFT
    
    gasto_por_franja_unica = df_cinema.groupby('time_slot')['gasto_turno_unico'].first().reset_index()
    gasto_por_franja_unica = gasto_por_franja_unica.rename(columns={'gasto_turno_unico': 'gasto_fijo_diario'})
    
    # b. Unimos el gasto fijo diario al DF agrupado para poder multiplicarlo
    df_grouped = pd.merge(df_grouped, gasto_por_franja_unica, on='time_slot', how='left')

    # c. Calculamos el GASTO TOTAL ACUMULADO: Gasto Fijo Diario * Días de Operación
    df_grouped['gasto_total_empleados'] = (
        df_grouped['gasto_fijo_diario'] * df_grouped['dias_con_sesiones']
    )
    
    # d. Renombramos la columna para que coincida con el uso en el frontend
    df_grouped['gasto_medio_empleados'] = df_grouped['gasto_total_empleados'] 
    
    # 5. Calcular % de Revenue
    df_grouped['revenue_percentage'] = (df_grouped['revenue_segment'] / total_revenue_cinema) * 100
    
    # 6. Ordenar los días y limpiar
    day_order_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    df_grouped['day_of_week_es'] = pd.Categorical(df_grouped['day_of_week_es'], categories=day_order_es, ordered=True)
    df_grouped = df_grouped.sort_values(by='day_of_week_es')
    
    df_grouped = df_grouped.drop(columns=['gasto_fijo_diario', 'gasto_total_empleados'])
    return df_grouped, total_revenue_cinema