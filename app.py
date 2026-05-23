import os
import gradio as ui
import pandas as pd
from typing import Tuple

# Importamos las funciones de tu pipeline (Persona B)
from persona_b.control_logic import cargar_eventos_mock, filtrar_eventos_cooldown
from persona_b.speech_engine import generar_audio_segmentos

def procesar_sistema_completo(video_path: str) -> Tuple[pd.DataFrame, str]:
    """
    Función puente que conecta el video cargado con el ciclo de vida del proyecto.
    Por ahora usa el archivo mock de eventos simulados.
    """
    if not video_path:
        return pd.DataFrame(), None
        
    # 📌 NOTA PARA EL FUTURO: Aquí es donde la Persona A tomará el 'video_path',
    # lo procesará con OpenCV + BLIP y generará el archivo JSON real.
    ruta_mock = "fixtures/mock_events.json"
    
    # --- CICLO DE VIDA DEL PIPELINE (PERSONA B) ---
    # 1. Carga de eventos
    eventos_totales = cargar_eventos_mock(ruta_mock)
    
    # 2. Filtrado por Cooldown (Regla de negocio)
    eventos_filtrados = filtrar_eventos_cooldown(eventos_totales, cooldown=5.0)
    
    # 3. Síntesis de voz con Bark
    segmentos_audio = generar_audio_segmentos(eventos_filtrados)
    
    # --- PREPARACIÓN DE DATOS PARA LA INTERFAZ ---
    # Creamos un historial visual para la tabla de la tarea
    historial_datos = []
    ids_filtrados = [e.timestamp for e in eventos_filtrados]
    
    for ev in eventos_totales:
        estado = "✅ Aprobado (Audio Generado)" if ev.timestamp in ids_filtrados else "🚫 Bloqueado por Cooldown"
        historial_datos.append({
            "Segundo": f"{ev.timestamp}s",
            "Evento Detectado": ev.commentary_text,
            "Acción del Sistema": estado
        })
    
    df_historial = pd.DataFrame(historial_datos)
    
    # Tomamos el primer audio generado para el reproductor de la interfaz
    audio_final = segmentos_audio[0].audio_path if segmentos_audio else None
    
    return df_historial, audio_final

# --- DISEÑO DE LA INTERFAZ GRÁFICA (GRADIO) ---
with ui.Blocks(title="Ecosystem de Narración IA - NuncaHuboPan") as interfaz:
    ui.Markdown("# 🎮 IA Shoutcaster Ecosystem - Control Panel")
    ui.Markdown("Prueba del pipeline completo de visión, filtrado de eventos y síntesis de voz.")
    
    with ui.Row():
        # --- COLUMNA IZQUIERDA: ENTRADA DE VIDEO ---
        with ui.Column(scale=1):
            ui.Markdown("### 📂 Entrada de Video")
            video_input = ui.Video(label="Cargar partida (.mp4)", format="mp4")
            btn_procesar = ui.Button("🚀 Iniciar Pipeline de Narración", variant="primary")
            
        # --- COLUMNA DERECHA: MONITOREO Y SALIDA DE AUDIO ---
        with ui.Column(scale=2):
            ui.Markdown("### 🎙️ Resultados del Sistema")
            
            # Reproductor de audio nativo de Gradio
            audio_output = ui.Audio(label="Narrador IA (Voz Bark Realista)", type="filepath")
            
            ui.Markdown("### 📊 Log de Decisiones (Lógica de Control)")
            # Tabla interactiva para mostrar el comportamiento del Cooldown
            tabla_log = ui.Dataframe(
                headers=["Segundo", "Evento Detectado", "Acción del Sistema"],
                datatype=["str", "str", "str"],
                interactive=False
            )

    # Conexión de componentes con la función del ciclo de vida
    btn_procesar.click(
        fn=procesar_sistema_completo,
        inputs=[video_input],
        outputs=[tabla_log, audio_output]
    )

# Lanzar la aplicación local
if __name__ == "__main__":
    interfaz.launch()