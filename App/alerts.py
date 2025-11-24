import streamlit as st
import pandas as pd
import sib_api_v3_sdk 
from sib_api_v3_sdk.rest import ApiException

def send_anomaly_alert(fecha, valor_real, valor_predicho, umbral):
    try:
        api_key = st.secrets["brevo"]["api_key"]
        sender_email = st.secrets["brevo"]["sender_email"]
        receiver_email = "raulvazquez97@hotmail.com" 
        configuration = sib_api_v3_sdk.Configuration()
        configuration.api_key['api-key'] = api_key
        configuration.verify_ssl = False
        api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))
        
    except KeyError as e:
        st.warning(f"Falta una clave en st.secrets['brevo']: {e}")
        return

    if isinstance(fecha, str):
        fecha = pd.to_datetime(fecha)
        
    fecha_str = fecha.strftime('%Y-%m-%d')
    
    subject = f"🚨 ALERTA DE ANOMALÍA - {fecha_str}"
    
    html_content = f"""
    <html>
        <body style="font-family: Arial, sans-serif; color: #333;">
            <h2 style="color: #D9534F;">⚠️ Anomalía Detectada: {fecha_str}</h2>
            <p>Se ha detectado que el valor real ha superado el umbral de desviación del modelo SARIMA.</p>
            <table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%;">
                <tr>
                    <td style="background-color: #f2f2f2;"><strong>Valor Real (Ventas)</strong></td>
                    <td>{valor_real:,.2f} €</td>
                </tr>
                <tr>
                    <td style="background-color: #f2f2f2;"><strong>Valor Predicho</strong></td>
                    <td>{valor_predicho:,.2f} €</td>
                </tr>
                <tr>
                    <td style="background-color: #f2f2f2;"><strong>Umbral de Desviación</strong></td>
                    <td>{umbral:,.2f} €</td>
                </tr>
            </table>
            <p style="margin-top: 20px;">Acción: Por favor, revise el gráfico y los datos de ventas para este día.</p>
        </body>
    </html>
    """

    send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
        to=[{"email": receiver_email}],
        html_content=html_content,
        sender={"name": "Alerta SARIMA", "email": sender_email},
        subject=subject
    )
    
    try:
        api_response = api_instance.send_transac_email(send_smtp_email)
        # st.success(f"Alerta enviada a {receiver_email} vía API de Brevo.")
    except ApiException as e:
        st.error(f"Fallo al enviar el correo (API de Brevo): {e}")