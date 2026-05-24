import sys
import os
import gradio as ui
import pandas as pd
import subprocess
from scipy.io import wavfile  # ¡NUEVO! Para medir la duración exacta de los audios

# --- 🚀 ARREGLO MAESTRO DE RUTAS Y CARPETAS 🚀 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "outputs"))
os.makedirs(OUTPUTS_DIR, exist_ok=True) 

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "persona_a"))
sys.path.append(os.path.join(BASE_DIR, "persona_b"))

# --- IMPORTAMOS LOS CONTRATOS ---
from contracts import GameEvent

# --- IMPORTAMOS LA PERSONA A Y B ---
from persona_a.vision_engine import extraer_frames, describir_frame, _get_client
from persona_b.speech_engine import generar_audio_segmentos

def convertir_a_caster(desc_raw: str) -> str:
    """Narración épica con tono de comentarista profesional."""
    try:
        client = _get_client()
        # Le damos un contexto de narrador profesional que busca fluidez
        prompt = f"""Eres un caster experto en eSports. Tu objetivo es narrar clips de Warzone con emoción y precisión.
        
        Descripción técnica: "{desc_raw}"
        
        INSTRUCCIONES DE ORO:
        - Máximo 12 palabras.
        - NO uses muletillas (y ahora, luego, entonces, acto seguido).
        - Usa frases directas que suenen a retransmisión oficial.
        - Si la acción es tensa, usa verbos de acción. Si es loot, usa tono calmado.
        - Responde SOLO con el diálogo del caster."""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=40,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7 # Un balance perfecto entre creatividad y control
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return desc_raw


def procesar_sistema_completo(video_path: str):
    if not video_path:
        yield pd.DataFrame(), None, None
        return
        
    print("🚀 Iniciando extracción de frames...")
    frames, duracion = extraer_frames(video_path)
    
    eventos_filtrados = []
    historial_datos = []
    
    # Creamos un DataFrame vacío para mostrarlo de inmediato
    df_historial = pd.DataFrame(columns=["Segundo", "Evento Detectado", "Acción del Sistema"])
    yield df_historial, None, None
    
    # Variables para calcular el Cooldown en tiempo real
    ultimo_timestamp_hablado = -float('inf')
    cooldown = 5.0

    print("🔍 Analizando frames con IA en tiempo real...")
    
    # 2. PROCESAMIENTO EN VIVO (FRAME POR FRAME)
    for f in frames:
        timestamp = f["timestamp"]
        path_imagen = f["path"]
        
        # Groq describe analíticamente el frame
        desc_raw = describir_frame(path_imagen)
        
        # Calculamos la Lógica de Control (Cooldown) en vivo
        if timestamp - ultimo_timestamp_hablado >= cooldown:
            estado = "✅ Aprobado (Caster Activo)"
            ultimo_timestamp_hablado = timestamp
            
            # ¡MAGIA EN VIVO! Convertimos la descripción a narración épica
            texto_caster = convertir_a_caster(desc_raw)
            
            # Guardamos el evento con el texto ya narrado
            evento = GameEvent(timestamp=timestamp, raw_caption=desc_raw, commentary_text=texto_caster)
            eventos_filtrados.append(evento)
            desc_para_tabla = texto_caster
        else:
            estado = "🚫 Repetitivo (Cooldown)"
            desc_para_tabla = desc_raw
            
        # Agregamos la nueva fila al historial
        historial_datos.append({
            "Segundo": f"{timestamp}s",
            "Evento Detectado": desc_para_tabla,
            "Acción del Sistema": estado
        })
        
        df_historial = pd.DataFrame(historial_datos)
        yield df_historial, None, None

    # 3. SÍNTESIS DE VOZ 
    print("🎙️ Sintetizando voces individuales...")
    segmentos_audio = generar_audio_segmentos(eventos_filtrados)
    audio_final = segmentos_audio[0].audio_path if segmentos_audio else None
    
    # 🎬 4. FUSIÓN AVANZADA CON FFMPEG (ANTI-COLISIÓN)
    video_final_path = None
    if segmentos_audio:
        print("🎬 Aplicando algoritmo Anti-Colisión y mezclando...")
        video_final_path = os.path.join(OUTPUTS_DIR, "video_final_narrado.mp4")
        
        comando_ffmpeg = ["ffmpeg", "-y", "-i", video_path]
        
        for seg in segmentos_audio:
            comando_ffmpeg.extend(["-i", seg.audio_path])
            
        filtros_delay = []
        etiquetas_salida = []
        tiempo_libre_ms = 0  # <--- NUESTRO CONTROLADOR DE MICRÓFONO
        
        for idx, seg in enumerate(segmentos_audio):
            input_idx = idx + 1 
            
            # 4.1 Medimos exactamente cuánto dura el audio generado
            rate, data = wavfile.read(seg.audio_path)
            duracion_audio_ms = int((len(data) / rate) * 1000)
            
            # 4.2 El tiempo en el que DEBERÍA empezar según el video
            retraso_ideal_ms = int(seg.timestamp * 1000)
            
            # 4.3 ANTI-COLISIÓN: Si el audio anterior no ha terminado, empujamos este
            retraso_real_ms = max(retraso_ideal_ms, tiempo_libre_ms)
            
            etiqueta = f"[a{input_idx}]"
            filtros_delay.append(f"[{input_idx}:a]adelay={retraso_real_ms}|{retraso_real_ms}{etiqueta}")
            etiquetas_salida.append(etiqueta)
            
            # 4.4 Calculamos cuándo estará libre el micrófono para el siguiente clip (+300ms de respiro)
            tiempo_libre_ms = retraso_real_ms + duracion_audio_ms + 300
            
        # Mezclamos todos los audios en una pista maestra
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
            print("✅ ¡Video sincronizado con éxito sin solapamientos!")
        except Exception as e:
            print(f"❌ Error al usar FFmpeg: {e}")
            video_final_path = None

    # 🚀 YIELD FINAL: Entregamos todo a la interfaz
    yield df_historial, audio_final, video_final_path


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
            
        # --- COLUMNA DERECHA: MONITOREO Y SALIDA ---
        with ui.Column(scale=2):
            ui.Markdown("### 🎬 Resultado Final")
            
            video_output = ui.Video(label="Video Narrado con IA", format="mp4")
            audio_output = ui.Audio(label="Narrador IA", type="filepath", visible=False)
            
            ui.Markdown("### 📊 Log de Decisiones (En tiempo real)")
            tabla_log = ui.Dataframe(
                headers=["Segundo", "Evento Detectado", "Acción del Sistema"],
                datatype=["str", "str", "str"],
                interactive=False
            )

    # Conectamos las 3 salidas
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