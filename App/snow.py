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
        secrets = st.secrets["snowflake"]
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

def apply_changes_to_snowflake(conn, df_changes):
    SNOWFLAKE_DB = conn.database
    SNOWFLAKE_SCHEMA = conn.schema
    TEMPORARY_TABLE = "CDC_CHANGES_STAGING" 
    
    try:
        success, n_chunks, n_rows, output = write_pandas(
            conn=conn,
            df=df_changes,
            table_name=TEMPORARY_TABLE,
            database=SNOWFLAKE_DB,
            schema=SNOWFLAKE_SCHEMA,
            auto_create_table=True, 
            overwrite=True,         
        )
        
        if not success:
            raise Exception(f"Fallo al cargar datos a la tabla temporal: {output}")

    except Exception as e:
        print(f"Error en la carga de datos a Snowflake: {e}")
        conn.rollback()
        raise
    
    cursor = conn.cursor()
    
    merge_sql = f"""
    MERGE INTO {SNOWFLAKE_DB}.{SNOWFLAKE_SCHEMA}.VENTAS_CINE_FINAL AS TARGET
    USING {TEMPORARY_TABLE} AS SOURCE
    ON (TARGET.FILM_CODE = SOURCE.FILM_CODE AND TARGET.CINEMA_CODE = SOURCE.CINEMA_CODE AND TARGET.DATE = SOURCE.DATE AND TARGET.SHOW_TIME = SOURCE.SHOW_TIME)

    WHEN MATCHED THEN
        CASE
            WHEN SOURCE.OPERACION = 'U' THEN UPDATE SET
                TARGET.TOTAL_SALES = SOURCE.TOTAL_SALES,
                TARGET.TICKETS_SOLD = SOURCE.TICKETS_SOLD,
                TARGET.TICKETS_OUT = SOURCE.TICKETS_OUT,
                TARGET.SHOW_TIME = SOURCE.SHOW_TIME,
                TARGET.OCCU_PERC = SOURCE.OCCU_PERC,
                TARGET.TICKET_PRICE = SOURCE.TICKET_PRICE,
                TARGET.TICKET_USE = SOURCE.TICKET_USE,
                TARGET.CAPACITY = SOURCE.CAPACITY,
                TARGET.MONTH = SOURCE.MONTH,
                TARGET.QUARTER = SOURCE.QUARTER,
                TARGET.DAY = SOURCE.DAY,
                TARGET.DAY_NAME = SOURCE.DAY_NAME,
                TARGET.N_SALAS = SOURCE.N_SALAS,
                TARGET.N_EMPLEADOS = SOURCE.N_EMPLEADOS,
                TARGET.SALARIO_HORA = SOURCE.SALARIO_HORA
            WHEN SOURCE.OPERACION = 'D' THEN DELETE
            ELSE NOP 
        END

    WHEN NOT MATCHED AND SOURCE.OPERACION = 'I' THEN
        INSERT (
            FILM_CODE, CINEMA_CODE, TOTAL_SALES, TICKETS_SOLD, TICKETS_OUT, SHOW_TIME, OCCU_PERC, 
            TICKET_PRICE, TICKET_USE, CAPACITY, DATE, MONTH, QUARTER, DAY, DAY_NAME, 
            N_SALAS, N_EMPLEADOS, SALARIO_HORA
        )
        VALUES (
            SOURCE.FILM_CODE, SOURCE.CINEMA_CODE, SOURCE.TOTAL_SALES, SOURCE.TICKETS_SOLD, SOURCE.TICKETS_OUT, 
            SOURCE.SHOW_TIME, SOURCE.OCCU_PERC, SOURCE.TICKET_PRICE, SOURCE.TICKET_USE, SOURCE.CAPACITY, 
            SOURCE.DATE, SOURCE.MONTH, SOURCE.QUARTER, SOURCE.DAY, SOURCE.DAY_NAME, 
            SOURCE.N_SALAS, SOURCE.N_EMPLEADOS, SOURCE.SALARIO_HORA
        );
    """
    
    try:
        cursor.execute(merge_sql)
        conn.commit()
        print("MERGE INTO ejecutado con éxito.")
        
    except Exception as e:
        print(f"Error al ejecutar MERGE INTO: {e}")
        conn.rollback()
        raise
        
    finally:
        cursor.execute(f"DROP TABLE IF EXISTS {TEMPORARY_TABLE}")
        cursor.close()

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

        apply_changes_to_snowflake(sf_conn, df_changes)

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