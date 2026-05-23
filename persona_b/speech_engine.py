import sys
sys.path.append("C:\\libs")  
sys.path.append("..")
import os
# Redirige las descargas a servidores globales optimizados en caso de conexiones móviles
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
import torch
import scipy.io.wavfile as wavfile
from transformers import VitsModel, AutoTokenizer
from typing import List

# Importación de contratos y de las funciones del módulo de lógica
from contracts import GameEvent, CommentarySegment
from control_logic import cargar_eventos_mock, filtrar_eventos_cooldown

# Inicialización y descarga del modelo Text-to-Speech en español
print("⏳ Cargando modelo de voz en español (facebook/mms-tts-spa)...")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-spa")
model = VitsModel.from_pretrained("facebook/mms-tts-spa")

def generar_audio_segmentos(eventos_filtrados: List[GameEvent]) -> List[CommentarySegment]:
    """
    Sintetiza el texto en español de los eventos aprobados y almacena 
    los archivos de audio resultantes en el disco local.
    """
    os.makedirs("../outputs", exist_ok=True)
    segmentos_finales = []

    for evento in eventos_filtrados:
        print(f"🎙️ Generando voz para el segundo {evento.timestamp}...")
        
        # El tokenizador prepara el texto y el modelo genera la onda de audio
        inputs = tokenizer(evento.commentary_text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform
        
        # Escritura del archivo físico en formato WAV
        ruta_audio = f"../outputs/audio_{evento.timestamp}s.wav"
        sampling_rate = model.config.sampling_rate
        wavfile.write(ruta_audio, rate=sampling_rate, data=output.__array__()[0])
        
        # El resultado se empaqueta en el contenedor final para la interfaz
        segmento = CommentarySegment(
            timestamp=evento.timestamp,
            text=evento.commentary_text,
            audio_path=ruta_audio
        )
        segmentos_finales.append(segmento)
        
    return segmentos_finales

# Orquestación del flujo independiente para las pruebas de la Persona B
if __name__ == "__main__":
    print("\n🚀 INICIANDO PIPELINE DE SALIDA (PERSONA B) 🚀\n")
    ruta_mock = "../fixtures/mock_events.json" 
    
    if not os.path.exists(ruta_mock):
        print(f"❌ Error: No se encontró el archivo mock en '{ruta_mock}'.")
        sys.exit(1)
        
    print("--- PASO 1: Cargando eventos simulados desde JSON ---")
    eventos_simulados = cargar_eventos_mock(ruta_mock)
    
    print("\n--- PASO 2: Ejecutando Lógica de Control (Cooldown) ---")
    eventos_filtrados = filtrar_eventos_cooldown(eventos_simulados, cooldown=5.0)
    
    print("\n--- PASO 3: Generando síntesis de voz (Text-to-Speech) ---")
    resultado_final = generar_audio_segmentos(eventos_filtrados)
    
    print("\n✅ ¡PIPELINE DE PRUEBA FINALIZADO CON ÉXITO! ✅")
    print("--------------------------------------------------")
    for seg in resultado_final:
        print(f"⏱️  [Seg {seg.timestamp}]: {seg.text}")
        print(f"   🔊 Archivo generado: {seg.audio_path}\n")