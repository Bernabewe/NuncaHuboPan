import sys
sys.path.append("..")
import os
# --- COPIA Y PEGA ESTO PARA SILENCIAR LOS WARNINGS DE LA TERMINAL ---
import warnings
from transformers import logging

# Silencia las advertencias nativas de Python
warnings.filterwarnings("ignore")
# Silencia las advertencias de desarrollo de Hugging Face / Transformers
logging.set_verbosity_error()
# ---------------------
import torch
import scipy.io.wavfile as wavfile
# Cargamos los componentes nativos de Bark en lugar del pipeline genérico
from transformers import BarkModel, AutoProcessor
from typing import List

# Importación de contratos y de las funciones del módulo de lógica
from contracts import GameEvent, CommentarySegment
from control_logic import cargar_eventos_mock, filtrar_eventos_cooldown

# Inicialización explícita y nativa del modelo y su procesador
print("⏳ Cargando procesador y modelo de voz Bark (suno/bark-small)...")
processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small")

# Configuración del dispositivo de cómputo (usa GPU si está disponible, si no, CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def generar_audio_segmentos(eventos_filtrados: List[GameEvent]) -> List[CommentarySegment]:
    """
    Sintetiza el texto de los eventos aprobados usando Bark de forma nativa.
    Aprovecha los preajustes de voz oficiales.
    """
    os.makedirs("../outputs", exist_ok=True)
    segmentos_finales = []

    for evento in eventos_filtrados:
        print(f"🎙️ Generando voz humana para el segundo {evento.timestamp}...")
        
        # El procesador se encarga de estructurar el texto y el locutor correctamente
        inputs = processor(
            text=[evento.commentary_text],
            voice_preset="v2/es_speaker_4"
        ).to(device)
        
        # Generación directa de la onda de audio sin intermediarios
        with torch.no_grad():
            audio_array = model.generate(**inputs)
        
        # Convertimos la salida de PyTorch a un array numérico que Scipy pueda escribir
        audio_data = audio_array.cpu().numpy().squeeze()
        
        ruta_audio = f"../outputs/audio_{evento.timestamp}s.wav"
        
        # Guardamos el archivo con la tasa de muestreo nativa de Bark (24kHz)
        sampling_rate = 24000
        wavfile.write(ruta_audio, rate=sampling_rate, data=audio_data)
        
        segmento = CommentarySegment(
            timestamp=evento.timestamp,
            text=evento.commentary_text,
            audio_path=ruta_audio
        )
        segmentos_finales.append(segmento)
        
    return segmentos_finales

# Orquestación del flujo independiente para las pruebas de la Persona B
if __name__ == "__main__":
    print("\n🚀 INICIANDO PIPELINE DE SALIDA NATIVO (PERSONA B) 🚀\n")
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
    
    print("\n--- PASO 3: Generando síntesis de voz (Bark Nativo) ---")
    resultado_final = generar_audio_segmentos(eventos_filtrados)
    
    print("\n✅ ¡PIPELINE DE PRUEBA FINALIZADO CON ÉXITO! ✅")
    print("--------------------------------------------------")
    for seg in resultado_final:
        print(f"⏱️  [Seg {seg.timestamp}]: {seg.text}")
        print(f"   🔊 Archivo generado: {seg.audio_path}\n")