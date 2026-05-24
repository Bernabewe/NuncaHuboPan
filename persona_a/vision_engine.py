import sys
sys.path.append("..")
import os
import json
import torch
from PIL import Image
import cv2
# Usamos el procesador nativo de la arquitectura de Google
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

# --- SILENCIAR WARNINGS ---
import warnings
from transformers import logging as tf_logging
warnings.filterwarnings("ignore")
tf_logging.set_verbosity_error()
# --------------------------

from video_reader import extraer_keyframes_dinamicos
from contracts import GameEvent

# Identificador del modelo ultra-ligero de Google
MODEL_ID = "briaai/paligemma-3b-pt-448"

print("⏳ Cargando VLM de Google de alta eficiencia (PaliGemma)...")

# Cargamos el procesador y el modelo de forma nativa
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

def analizar_video_y_generar_json(ruta_video: str, ruta_salida_json: str):
    keyframes = extraer_keyframes_dinamicos(ruta_video)
    eventos_detectados = []
    
    print("\n🧠 --- PROCESAMIENTO ULTRA-RÁPIDO CON PALIGEMMA ---")
    # Prompt corto en inglés (PaliGemma responde mejor así y luego lo mapeamos o dejamos directo)
    prompt = "caption es" 

    for timestamp, frame in keyframes:
        print(f"🤖 Procesando segundo {timestamp}s...")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # El procesador de Google es extremadamente rápido empaquetando la imagen
        inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20)
            
        # Decodificamos el resultado recortando el prompt inicial
        descripcion = processor.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
        print(f"   ↳ [IA]: {descripcion}")
        
        evento = GameEvent(
            timestamp=float(timestamp),
            commentary_text=descripcion
        )
        
        eventos_detectados.append({
            "timestamp": evento.timestamp,
            "commentary_text": evento.commentary_text
        })
        
    os.makedirs(os.path.dirname(ruta_salida_json), exist_ok=True)
    with open(ruta_salida_json, "w", encoding="utf-8") as f:
        json.dump(eventos_detectados, f, indent=4, ensure_ascii=False)
        
    print(f"\n💾 Archivo guardado con éxito en: {ruta_salida_json}")

if __name__ == "__main__":
    VIDEO_DE_PRUEBA = "../fixtures/partida_test.mp4"
    JSON_DESTINO = "../fixtures/mock_events.json"
    
    if os.path.exists(VIDEO_DE_PRUEBA):
        analizar_video_y_generar_json(VIDEO_DE_PRUEBA, JSON_DESTINO)
        print("\n✅ El archivo JSON fue generado con éxito. Ya puedes ejecutar speech_engine.py para escuchar los audios rápidos.")