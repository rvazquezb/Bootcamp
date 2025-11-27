import pandas as pd
import psycopg2 
import snowflake.connector 
from snowflake.connector.pandas_tools import write_pandas
import streamlit as st

def get_neon_connection():
    try:
        secrets = st.secrets["neon_db"]
        conn_string = secrets["connection_string"]
        conn = psycopg2.connect(conn_string)
        return conn
    except KeyError as e:
        st.error(f"Falta la clave de configuración de Neon: {e}. Revisa secrets.toml.")
        raise
    except Exception as e:
        st.error(f"Error al conectar con Neon: {e}")
        raise

def get_snowflake_connection():
    try:
        secrets = st.secrets["snow"]
        conn = snowflake.connector.connect(
            user=secrets["user"],
            password=secrets["password"],
            account=secrets["account"],
            warehouse=secrets["warehouse"],
            database=secrets["database"],
            schema=secrets["schema"],
            role=secrets["role"]
        )
        return conn
    except KeyError as e:
        st.error(f"Falta la clave de configuración de Snowflake: {e}. Revisa secrets.toml.")
        raise
    except Exception as e:
        st.error(f"Error al conectar con Snowflake: {e}")
        raise

def read_changes_from_neon(conn):
    sql_read = """
    SELECT *
    FROM ventas_cine_final_staging
    ORDER BY ts_transaccion ASC;
    """
    
    df = pd.read_sql(sql_read, conn)
    
    staging_ids = df['staging_id'].tolist() 
    
    return df, staging_ids

def format_sql_value(value, is_string=False):
    if pd.isna(value) or value is None:
        return 'NULL'
    
    if isinstance(value, pd.Timestamp):
        return f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'"
        
    if is_string or isinstance(value, str):
        safe_value = str(value).replace("'", "''") 
        return f"'{safe_value}'"
        
    return str(value)

def apply_changes_to_snowflake(conn, neon_conn, df_changes):
    
    cursor = conn.cursor()
    cursor_neon = neon_conn.cursor()
    df_changes = df_changes.sort_values(by='ts_transaccion') 
    processed_staging_ids = []
    for index, row in df_changes.iterrows():
        if not row['copied']:
            op = row['operacion']
            
            where_clause = f"""
                WHERE FILM_CODE = {row['film_code']} 
                AND CINEMA_CODE = {row['cinema_code']} 
                AND SHOW_TIME = {row['show_time']}
                AND DATE = '{row['date'].strftime('%Y-%m-%d %H:%M:%S')}'
            """
            try:
                if op == 'I':
                    insert_sql = f"""
                    INSERT INTO VENTAS_CINE_FINAL (
                        FILM_CODE, CINEMA_CODE, TOTAL_SALES, TICKETS_SOLD, TICKETS_OUT, SHOW_TIME, 
                        OCCU_PERC, TICKET_PRICE, TICKET_USE, CAPACITY, DATE, MONTH, QUARTER, 
                        DAY, DAY_NAME, N_SALAS, N_EMPLEADOS, SALARIO_HORA
                    )
                    VALUES (
                        {row['film_code']}, {row['cinema_code']}, {row['total_sales']}, {row['tickets_sold']}, {row['tickets_out']}, {row['show_time']}, 
                        {row['occu_perc']}, {row['ticket_price']}, {row['ticket_use']}, {row['capacity']}, '{row['date'].strftime('%Y-%m-%d %H:%M:%S')}', {row['month']}, {row['quarter']}, 
                        {row['day']}, '{row['day_name']}', {row['n_salas']}, {row['n_empleados']}, {row['salario_hora']}
                    );
                    """
                    cursor.execute(insert_sql)
                    
                elif op == 'U':
                    update_sql = f"""
                    UPDATE VENTAS_CINE_FINAL SET
                        TOTAL_SALES = {row['total_sales']},
                        TICKETS_SOLD = {row['tickets_sold']},
                        TICKETS_OUT = {row['tickets_out']},
                        SHOW_TIME = {row['show_time']},
                        OCCU_PERC = {row['occu_perc']},
                        TICKET_PRICE = {row['ticket_price']},
                        TICKET_USE = {row['ticket_use']},
                        CAPACITY = {row['capacity']},
                        DATE = {format_sql_value(row['date'].strftime('%Y-%m-%d %H:%M:%S'))},
                        MONTH = {row['month']},
                        QUARTER = {row['quarter']},
                        DAY = {row['day']},
                        DAY_NAME = '{row['day_name']}',
                        N_SALAS = {row['n_salas']},
                        N_EMPLEADOS = {row['n_empleados']},
                        SALARIO_HORA = {row['salario_hora']}
                    {where_clause};
                    """
                    cursor.execute(update_sql)
                    
                elif op == 'D':
                    delete_sql = f"DELETE FROM VENTAS_CINE_FINAL {where_clause};"
                    cursor.execute(delete_sql)
                processed_staging_ids.append(row['staging_id'])

                conn.commit()
        
            except Exception as e:
                # Si algo falla en Snowflake, hacemos rollback y relanzamos
                conn.rollback() 
                print(f"Error crítico en Snowflake. Se deshacen los cambios. Error: {e}")
                raise 
                
        else:
            print(f"Saltando fila ID {row['staging_id']} (Op: {op}) - Ya marcada como copiada.")

    if processed_staging_ids:
        ids_placeholder = ','.join([str(id) for id in processed_staging_ids])
        
        neon_update_sql = f"""
            UPDATE ventas_cine_final_staging 
            SET copied = TRUE
            WHERE staging_id IN ({ids_placeholder});
        """
        try:
            cursor_neon.execute(neon_update_sql)
            neon_conn.commit()
            print(f"Checkpoint exitoso: {len(processed_staging_ids)} filas marcadas como copiadas en Neon.")
        except Exception as e:
            neon_conn.rollback()
            print(f"ALERTA: Fallo al guardar el checkpoint en Neon. Las filas se reprocesarán. Error: {e}")
            raise
    
    print(f"Aplicación completada. {len(processed_staging_ids)} cambios aplicados en Snowflake.")
    cursor.close()
    cursor_neon.close()

def clean_neon_staging_table(conn, staging_ids):
    
    if not staging_ids:
        return
        
    ids_tuple = tuple(staging_ids)
    
    ids_placeholder = ','.join(['%s'] * len(ids_tuple)) 
    
    sql_delete = f"""
    DELETE FROM ventas_cine_final_staging
    WHERE staging_id IN ({ids_placeholder});
    """
    
    cursor = conn.cursor()
    cursor.execute(sql_delete, ids_tuple) 
    conn.commit()
    cursor.close()

def sync_data_to_snowflake():
    neon_conn = get_neon_connection()
    sf_conn = get_snowflake_connection()
    
    try:
        df_changes, staging_ids = read_changes_from_neon(neon_conn)
        
        if df_changes.empty:
            print("No hay cambios nuevos para sincronizar.")
            return

        apply_changes_to_snowflake(sf_conn, neon_conn, df_changes)

        clean_neon_staging_table(neon_conn, staging_ids)
        
        print(f"Sincronización completada. {len(df_changes)} registros procesados.")

    except Exception as e:
        print(f"Error en la sincronización: {e}")
        if sf_conn: sf_conn.rollback() 
        if neon_conn: neon_conn.rollback()
        
    finally:
        if sf_conn: sf_conn.close()
        if neon_conn: neon_conn.close()

if __name__ == "__main__":
    sync_data_to_snowflake()