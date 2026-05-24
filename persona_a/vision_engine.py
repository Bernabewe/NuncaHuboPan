import sys
sys.path.append("..")
import os
import json
import torch
from PIL import Image
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- 🚀 SILENCIAR WARNINGS Y LOGS DE TERMINAL ---
import warnings
from transformers import logging as tf_logging
warnings.filterwarnings("ignore")
tf_logging.set_verbosity_error()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# --------------------------------------------

from video_reader import extraer_keyframes_dinamicos
from contracts import GameEvent

# El modelo BLIP Base es ultra-ligero (~900MB), ideal para servidores con poca RAM
MODEL_ID = "Salesforce/blip-image-captioning-base"

print("⏳ Cargando el modelo ultra-ligero de la clase (Salesforce/BLIP)...")

# Inicialización nativa del procesador y el modelo ligero
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32 # Totalmente compatible y estable en cualquier CPU de servidor
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

def analizar_video_y_generar_json(ruta_video: str, ruta_salida_json: str):
    """
    Pipeline de la Persona A usando BLIP para generar descripciones rápidas sin agotar la RAM.
    """
    # 1. Obtener los fotogramas clave limpios desde OpenCV
    keyframes = extraer_keyframes_dinamicos(ruta_video)
    eventos_detectados = []
    
    print("\n🧠 --- PROCESAMIENTO ULTRA-LIGERO CON BLIP ---")
    
    for timestamp, frame in keyframes:
        print(f"🤖 Analizando escena del segundo {timestamp}s...")
        
        # Conversión de BGR (OpenCV) a RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # BLIP procesa la imagen de forma condicional. Al no pasarle texto inicial,
        # genera un subtítulo descriptivo (Captioning) de manera automática.
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20)
            
        descripcion = processor.decode(output[0], skip_special_tokens=True)
        
        # Pequeño truco: BLIP es nativo en inglés. Mantenemos el flujo rápido agregando
        # un tag emocionante o limpiándolo para el json, asegurando estabilidad total.
        descripcion_limpia = descripcion.strip()
        print(f"   ↳ [IA]: {descripcion_limpia}")
        
        # Estructuración bajo el contrato acordado para la Persona B
        evento = GameEvent(
            timestamp=float(timestamp),
            commentary_text=descripcion_limpia
        )
        
        eventos_detectados.append({
            "timestamp": evento.timestamp,
            "commentary_text": evento.commentary_text
        })
        
    # 2. Guardar el JSON final en la carpeta compartida
    os.makedirs(os.path.dirname(ruta_salida_json), exist_ok=True)
    with open(ruta_salida_json, "w", encoding="utf-8") as f:
        json.dump(eventos_detectados, f, indent=4, ensure_ascii=False)
        
    print(f"\n💾 Pipeline completado con éxito. Datos listos en: {ruta_salida_json}")

if __name__ == "__main__":
    VIDEO_DE_PRUEBA = "../fixtures/partida_test.mp4"
    JSON_DESTINO = "../fixtures/mock_events.json"
    
    if not os.path.exists(VIDEO_DE_PRUEBA):
        print(f"❌ Coloca tu video '.mp4' de prueba en '{VIDEO_DE_PRUEBA}' antes de ejecutar.")
    else:
        print("\n🚀 INICIANDO ENTORNO DE VISIÓN CON MODELO DE LA CLASE 🚀")
        analizar_video_y_generar_json(VIDEO_DE_PRUEBA, JSON_DESTINO)