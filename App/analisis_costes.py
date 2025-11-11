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
    
    # 📌 PARÁMETRO DE COSTE FIJO POR TURNO
    # El gasto se calcula por el turno completo de 8 horas, independientemente
    # del número de sesiones que caigan dentro de esa franja.
    HOURS_PER_SHIFT = 8 
    
    # 1. Filtrar por el cine seleccionado
    df_cinema = df[df['cinema_code'] == cinema_id].copy()

    # 2. CÁLCULO DEL GASTO SALARIAL POR TURNO (NUEVA LÓGICA)
    # Asumimos que la asignación de empleados (n_empleados) cubre el turno completo.
    # Coste Fijo del Turno = Empleados * Salario_Hora * 8 Horas
    df_cinema['gasto_turno_fijo'] = (
        df_cinema['n_empleados'] * df_cinema['salario_hora'] * HOURS_PER_SHIFT
    )
    
    total_revenue_cinema = df_cinema['total_sales'].sum()
    
    if total_revenue_cinema == 0:
        return pd.DataFrame(), 0 
        
    # 3. Agrupar para obtener los valores agregados de Revenue, Coste y Conteo
    # La clave es agrupar por Día y Franja, pero necesitamos obtener el 'gasto_turno_fijo'
    # de manera única, ya que se repite para cada sesión dentro de la misma franja.
    
    df_grouped = df_cinema.groupby(['day_of_week_es', 'time_slot']).agg(
        revenue_segment=('total_sales', 'sum'),
        # La suma total de 'gasto_turno_fijo' está inflada, ya que cuenta el coste por CADA SESIÓN.
        # En su lugar, agruparemos de manera más granular para contar días únicos.
        sessions_count=('date', 'count'),
        dias_con_sesiones=('date', 'nunique') # Cuenta cuántos días hubo actividad en esta franja
    ).reset_index()

    # 4. CALCULAR EL COSTE PROMEDIO (Este paso se vuelve más simple)
    # Ahora que tenemos 'dias_con_sesiones', necesitamos el Gasto Diario Real.
    
    # Paso Intermedio: Calcular el Gasto por Franja en un solo día (Valor único)
    # Tomamos el gasto de la primera fila de esa franja en el DF filtrado, 
    # ya que debería ser el mismo para todas las sesiones de ese día y franja.
    
    # Primero, calculamos el gasto que corresponde a una sola instancia de esa franja
    # (asumiendo que n_empleados y salario_hora son constantes por cine, que es su lógica)
    
    # Gasto de un solo turno (fijo por cine y franja - el nocturno tiene plus)
    df_cinema['gasto_turno_unico'] = df_cinema['n_empleados'] * df_cinema['salario_hora'] * HOURS_PER_SHIFT
    
    # Extraemos el valor único del Gasto Fijo por Turno para el cine seleccionado.
    # Dado que el 'n_empleados' y 'salario_hora' varían solo por 'time_slot' (nocturno)
    # y 'cinema_code', el coste fijo por turno es constante para esa combinación.

    gasto_por_franja_unica = df_cinema.groupby('time_slot')['gasto_turno_unico'].first().reset_index()
    gasto_por_franja_unica = gasto_por_franja_unica.rename(columns={'gasto_turno_unico': 'gasto_fijo_diario'})
    
    # b. Unimos el gasto fijo diario al DF agrupado para poder multiplicarlo
    df_grouped = pd.merge(df_grouped, gasto_por_franja_unica, on='time_slot', how='left')

    # c. Calculamos el GASTO TOTAL ACUMULADO: Gasto Fijo Diario * Días de Operación
    df_grouped['gasto_total_empleados'] = (
        df_grouped['gasto_fijo_diario'] * df_grouped['dias_con_sesiones']
    )
    
    # d. Renombramos la columna para que coincida con el uso en el frontend
    df_grouped['gasto_medio_empleados'] = df_grouped['gasto_total_empleados'] # 👈 SE MANTIENE EL NOMBRE DE COLUMNA DEL FRONTEND
    
    # 5. Calcular % de Revenue
    df_grouped['revenue_percentage'] = (df_grouped['revenue_segment'] / total_revenue_cinema) * 100
    
    # 6. Ordenar los días y limpiar
    day_order_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    df_grouped['day_of_week_es'] = pd.Categorical(df_grouped['day_of_week_es'], categories=day_order_es, ordered=True)
    df_grouped = df_grouped.sort_values(by='day_of_week_es')
    
    df_grouped = df_grouped.drop(columns=['gasto_fijo_diario', 'gasto_total_empleados'])
    return df_grouped, total_revenue_cinema