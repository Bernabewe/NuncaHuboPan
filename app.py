################################################################################
# Archivo principal de la aplicacion que integra los 
# componentes de vision y voz. Implementa la interfaz grafica de usuario usando 
# Gradio y coordina el procesamiento del video.
################################################################################

import sys
import os
import gradio as ui
import pandas as pd
import subprocess
from scipy.io import wavfile  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "outputs"))
os.makedirs(OUTPUTS_DIR, exist_ok=True) 

sys.path.append(BASE_DIR)

# Importa los modulos de la aplicacion para manejar los datos y procesar medios
from contracts import GameEvent
from vision_engine import extraer_frames, describir_frame, _get_client
from speech_engine import generar_audio_segmentos

# Convierte una descripcion simple a un comentario epico utilizando el modelo de lenguaje
def convertir_a_caster(desc_raw: str) -> str:
    try:
        client = _get_client()
        prompt = f"""Eres un caster experto en eSports. Tu objetivo es narrar clips de Warzone con emocion y precision.
        
        Descripcion tecnica: "{desc_raw}"
        
        INSTRUCCIONES DE ORO:
        - Maximo 12 palabras.
        - NO uses muletillas.
        - Usa frases directas.
        - Responde SOLO con el dialogo del caster."""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=40,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7 
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return desc_raw

# Ejecuta el flujo completo de analisis de frames y sintesis de voz en tiempo real
# Ejecuta el flujo completo de analisis de frames y sintesis de voz en tiempo real
def procesar_sistema_completo(video_path: str):
    if not video_path:
        yield pd.DataFrame(columns=["Segundo", "Evento Detectado", "Accion del Sistema"]), None, None
        return
        
    frames, duracion = extraer_frames(video_path)
    
    eventos_filtrados = []
    historial_datos = [] # Lista donde acumularemos las filas ordenadamente
    
    # Render inicial con la tabla limpia
    df_historial = pd.DataFrame(columns=["Segundo", "Evento Detectado", "Accion del Sistema"])
    yield df_historial, None, None
    
    ultimo_timestamp_hablado = -float('inf')
    cooldown = 5.0

    # Analiza cada fotograma y determina si se aprueba para narracion
    for f in frames:
        timestamp = f["timestamp"]
        path_imagen = f["path"]
        
        desc_raw = describir_frame(path_imagen)
        
        if timestamp - ultimo_timestamp_hablado >= cooldown:
            estado = "Aprobado (Caster Activo)"
            ultimo_timestamp_hablado = timestamp
            
            texto_caster = convertir_a_caster(desc_raw)
            
            evento = GameEvent(timestamp=timestamp, raw_caption=desc_raw, commentary_text=texto_caster)
            eventos_filtrados.append(evento)
            desc_para_tabla = texto_caster
        else:
            estado = "Repetitivo (Cooldown)"
            desc_para_tabla = desc_raw
            
        # CORRECCIÓN AQUÍ: Guardamos explícitamente los datos mapeados en formato de lista para el DataFrame
        historial_datos.append([
            f"{timestamp}s",
            desc_para_tabla,
            estado
        ])
        
        # Reconstruimos el DataFrame con todo el histórico acumulado hasta este segundo
        df_historial = pd.DataFrame(historial_datos, columns=["Segundo", "Evento Detectado", "Accion del Sistema"])
        yield df_historial, None, None

    # Sintetiza las voces y mezcla los audios resultantes con el video original usando FFmpeg
    segmentos_audio = generar_audio_segmentos(eventos_filtrados)
    audio_final = segmentos_audio[0].audio_path if segmentos_audio else None
    
    video_final_path = None
    if segmentos_audio:
        video_final_path = os.path.join(OUTPUTS_DIR, "video_final_narrado.mp4")
        
        comando_ffmpeg = ["ffmpeg", "-y", "-i", video_path]
        
        for seg in segmentos_audio:
            comando_ffmpeg.extend(["-i", seg.audio_path])
            
        filtros_delay = []
        etiquetas_salida = []
        tiempo_libre_ms = 0  
        
        for idx, seg in enumerate(segmentos_audio):
            input_idx = idx + 1 
            
            rate, data = wavfile.read(seg.audio_path)
            duracion_audio_ms = int((len(data) / rate) * 1000)
            
            retraso_ideal_ms = int(seg.timestamp * 1000)
            retraso_real_ms = max(retraso_ideal_ms, tiempo_libre_ms)
            
            etiqueta = f"[a{input_idx}]"
            filtros_delay.append(f"[{input_idx}:a]adelay={retraso_real_ms}|{retraso_real_ms}{etiqueta}")
            etiquetas_salida.append(etiqueta)
            
            tiempo_libre_ms = retraso_real_ms + duracion_audio_ms + 300
            
        if len(segmentos_audio) > 1:
            todas_las_etiquetas = "".join(etiquetas_salida)
            filtro_complejo = f"{';'.join(filtros_delay)};{todas_las_etiquetas}amix=inputs={len(segmentos_audio)}[aout]"
            map_audio = "[aout]"
        else:
            filtro_complejo = filtros_delay[0]
            map_audio = etiquetas_salida[0]
            
        comando_ffmpeg.extend([
            "-filter_complex", filtro_complejo,
            "-map", "0:v:0",
            "-map", map_audio,
            "-c:v", "copy",
            "-c:a", "aac",
            video_final_path
        ])
        
        try:
            subprocess.run(comando_ffmpeg, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            video_final_path = None

    # Retorno final manteniendo el dataframe del historial completamente lleno
    yield df_historial, audio_final, video_final_path

# Define y construye la interfaz grafica interactiva de usuario
with ui.Blocks(title="Ecosystem de Narracion IA - NuncaHuboPan", theme="dark") as interfaz:
    
    with ui.Row():
        with ui.Column():
            ui.Markdown("# Proyecto Final (Narrador IA)")
            ui.Markdown("### INTEGRANTES: <br> Bernardo Bejarano | Francisco Meza | Gabriel Roman | Lanna Meza")
    ui.Markdown("---")
    
    with ui.Row():
        with ui.Column(scale=1):
            ui.Markdown("### Entrada de Video")
            video_input = ui.Video(label="Cargar partida (.mp4)", format="mp4", sources=["upload"])
            btn_procesar = ui.Button("Iniciar Pipeline de Narracion", variant="primary")
            
        with ui.Column(scale=2):
            ui.Markdown("### Resultado Final")
            
            video_output = ui.Video(label="Video Narrado con IA", format="mp4")
            audio_output = ui.Audio(label="Narrador IA", type="filepath", visible=False)
            
            ui.Markdown("### Log de Decisiones (En tiempo real)")
            tabla_log = ui.Dataframe(
                headers=["Segundo", "Evento Detectado", "Accion del Sistema"],
                datatype=["str", "str", "str"],
                interactive=False
            )

    btn_procesar.click(
        fn=procesar_sistema_completo,
        inputs=[video_input],
        outputs=[tabla_log, audio_output, video_output]
    )

if __name__ == "__main__":
    interfaz.queue().launch(
        server_name="0.0.0.0", 
        server_port=7860,
        allowed_paths=[OUTPUTS_DIR]
    )