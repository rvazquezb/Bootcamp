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