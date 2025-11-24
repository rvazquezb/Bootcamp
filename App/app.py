import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import bcrypt
from graficos import graficos
from graficos import kpis
from snow import sync_data_to_snowflake

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
        db_url = st.secrets["neon_db"]["connection_string"]
        engine = create_engine(db_url)
        
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
        db_url = st.secrets["neon_db"]["connection_string"]
        engine = create_engine(db_url)
        
        table_name = st.secrets["table_name"]["table_string"]
        df = pd.read_sql_table(table_name, con=engine)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        return df
    except Exception as e:
        st.error(f"❌ Error al conectar o cargar datos desde Neon (Cloud).")
        st.error("Verifica que la cadena de conexión es correcta y que la base de datos Neon está activa.")
        st.error(f"Detalle del error: {e}")
        return pd.DataFrame() 

#Función para mostrar el formulario de login
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

#Función para insertar datos en la base de datos
def insert_data(date, film_code, cinema_code, ticket_price, tickets_sold, ticket_use, show_time, tickets_out, capacity):
    try:
        db_url = st.secrets["neon_db"]["connection_string"]
        engine = create_engine(db_url)
        
        total_sales = ticket_price * tickets_sold
        occu_perc = (ticket_use / capacity) * 100 
        month = date.month
        quarter = (date.month - 1) // 3 + 1 
        day = date.day
        day_name = date.strftime("%A")
        table_name = st.secrets["table_name"]["table_string"]
        insert_query = text(
            f"INSERT INTO {table_name} (film_code, cinema_code, total_sales, tickets_sold, tickets_out, show_time, occu_perc, ticket_price, ticket_use, capacity, date, month, quarter, day, day_name) "
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
        st.cache_resource.clear()
        return True
        
    except Exception as e:
        st.error(f"❌ Error al insertar datos: {e}")
        return False

#Función para cambiar la vista activa
def change_view(view_name):
    st.session_state['active_view'] = view_name

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
        
        #Control de vista basado en el rol del usuario
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
                st.sidebar.button('Sync Snow', on_click=sync_data_to_snowflake)
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