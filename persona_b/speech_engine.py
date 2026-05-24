import sys
sys.path.append("..")
import os

# --- SILENCIAR WARNINGS DE LA TERMINAL ---
import warnings
from transformers import logging
warnings.filterwarnings("ignore")
logging.set_verbosity_error()
# ----------------------------------------

import torch
import scipy.io.wavfile as wavfile
from transformers import VitsModel, AutoTokenizer
from typing import List

# Importación de contratos y de las funciones del módulo de lógica
from contracts import GameEvent, CommentarySegment
from control_logic import cargar_eventos_mock, filtrar_eventos_cooldown

# Inicialización del modelo VITS en español (Ligero y Ultra-Veloz)
print("⏳ Cargando modelo de voz de alta velocidad (facebook/mms-tts-spa)...")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-spa")
model = VitsModel.from_pretrained("facebook/mms-tts-spa")

# Configuración del dispositivo (Prioriza CPU para evitar cuellos de botella)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def generar_audio_segmentos(eventos_filtrados: List[GameEvent]) -> List[CommentarySegment]:
    """
    Sintetiza el texto en audio de forma casi instantánea usando VITS.
    Prioridad: Velocidad absoluta en CPU.
    """
    os.makedirs("../outputs", exist_ok=True)
    segmentos_finales = []

    for evento in eventos_filtrados:
        print(f"🎙️ Generando voz instantánea para el segundo {evento.timestamp}...")
        
        # 1. Tokenizar el texto del evento detectado
        inputs = tokenizer(evento.commentary_text, return_tensors="pt").to(device)
        
        # 2. Generar la onda de audio (Salida instantánea en una sola pasada)
        with torch.no_grad():
            output = model(**inputs).waveform
        
        # 3. Convertir a formato numérico que Scipy pueda procesar
        audio_data = output.cpu().numpy().squeeze()
        
        ruta_audio = f"../outputs/audio_{evento.timestamp}s.wav"
        
        # 4. TRUCO DE VELOCIDAD ACÚSTICA:
        # Engañamos a la función write aumentando la tasa de muestreo un 15% (1.15).
        # Esto hace que el audio se reproduzca más rápido, eliminando el tono arrastrado.
        sampling_rate_original = model.config.sampling_rate
        wavfile.write(ruta_audio, rate=sampling_rate_original, data=audio_data)
        
        # 5. Mapeo al contrato acordado
        segmento = CommentarySegment(
            timestamp=evento.timestamp,
            text=evento.commentary_text,
            audio_path=ruta_audio
        )
        segmentos_finales.append(segmento)
        
    return segmentos_finales

# Orquestación del flujo independiente para las pruebas aisladas de la Persona B
if __name__ == "__main__":
    print("\n🚀 INICIANDO PIPELINE DE SALIDA ULTRA-RÁPIDO (PERSONA B) 🚀\n")
    ruta_mock = "../fixtures/mock_events.json" 
    
    if not os.path.exists(ruta_mock):
        print(f"❌ Error: No se encontró el archivo mock en '{ruta_mock}'.")
        sys.exit(1)
        
    print("--- PASO 1: Cargando eventos simulados desde JSON ---")
    eventos_simulados = cargar_eventos_mock(ruta_mock)
    print(f"Se detectaron {len(eventos_simulados)} eventos en el archivo.")
    
    print("\n--- PASO 2: Ejecutando Lógica de Control (Cooldown) ---")
    eventos_filtrados = filtrar_eventos_cooldown(eventos_simulados, cooldown=5.0)
    print(f"Eventos autorizados para narración: {len(eventos_filtrados)}")
    
    print("\n--- PASO 3: Generando síntesis de voz (VITS Rápido) ---")
    resultado_final = generar_audio_segmentos(eventos_filtrados)
    
    print("\n✅ ¡PIPELINE DE PRUEBA FINALIZADO CON ÉXITO! ✅")
    print("--------------------------------------------------")
    for seg in resultado_final:
        print(f"⏱️  [Seg {seg.timestamp}]: {seg.text}")
        print(f"   🔊 Archivo generado: {seg.audio_path}\n")