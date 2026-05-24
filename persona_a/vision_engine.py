import sys
sys.path.append("..")
import os
import json

# --- 🚀 SILENCIAR WARNINGS Y LOGS DE LA TERMINAL ---
import warnings
from transformers import logging as tf_logging
import logging

warnings.filterwarnings("ignore")
tf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# --------------------------------------------------

import torch
from PIL import Image
import cv2
from transformers import AutoModelForCausalLM, AutoTokenizer

# Importamos el lector y los contratos compartidos
from video_reader import extraer_keyframes_dinamicos
from contracts import GameEvent

# Usamos el identificador del modelo principal. Al remover la revisión vieja de mayo de 2024,
# Hugging Face jalará la última actualización del modelo que ya es compatible con transformers moderno.
MODEL_ID = "vikhyatk/moondream2"

print("⏳ Cargando VLM de alta velocidad desde Hugging Face (Moondream2)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Cargamos el modelo directamente de la rama principal (main), permitiendo que use
# los scripts de inicialización de Phi actualizados por el autor.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True, 
    torch_dtype=torch.float32 # Usamos float32 para asegurar compatibilidad total en CPU sin problemas de tipos
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

def analizar_video_y_generar_json(ruta_video: str, ruta_salida_json: str):
    """
    Ejecuta el pipeline de la Persona A procesando fotogramas clave con Moondream2.
    """
    # 1. Obtener solo los fotogramas con movimiento real desde OpenCV
    keyframes = extraer_keyframes_dinamicos(ruta_video)
    
    eventos_detectados = []
    
    print("\n🧠 --- PROCESAMIENTO INTELIGENTE CON MOONDREAM2 ---")
    prompt_shoutcaster = "Describe en una sola frase corta y emocionante en español qué acción está pasando en este videojuego."

    for timestamp, frame in keyframes:
        print(f"🤖 Analizando escena del segundo {timestamp}s...")
        
        # Conversión de BGR (OpenCV) a RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Codificación de la imagen en tensores utilizando la API nativa de Moondream
        image_embeds = model.encode_image(pil_image)
        
        # Generación del comentario por parte del VLM
        with torch.no_grad():
            descripcion = model.answer_question(image_embeds, prompt_shoutcaster, tokenizer)
            
        descripcion_limpia = descripcion.strip()
        print(f"   ↳ [IA]: {descripcion_limpia}")
        
        # Estructuración bajo el contrato acordado
        evento = GameEvent(
            timestamp=float(timestamp),
            commentary_text=descripcion_limpia
        )
        
        eventos_detectados.append({
            "timestamp": evento.timestamp,
            "commentary_text": evento.commentary_text
        })
        
    # 2. Guardar el archivo de salida para la Persona B
    os.makedirs(os.path.dirname(ruta_salida_json), exist_ok=True)
    with open(ruta_salida_json, "w", encoding="utf-8") as f:
        json.dump(eventos_detectados, f, indent=4, ensure_ascii=False)
        
    print(f"\n💾 ¡Ciclo de Persona A finalizado! Datos guardados en: {ruta_salida_json}")

if __name__ == "__main__":
    VIDEO_DE_PRUEBA = "../fixtures/partida_test.mp4"
    JSON_DESTINO = "../fixtures/mock_events.json"
    
    if not os.path.exists(VIDEO_DE_PRUEBA):
        print(f"❌ Coloca un archivo de video '.mp4' en '{VIDEO_DE_PRUEBA}' para iniciar el pipeline.")
    else:
        print("\n🚀 INICIANDO PIPELINE DE PRUEBA DE VISIÓN GLOBAL 🚀")
        analizar_video_y_generar_json(VIDEO_DE_PRUEBA, JSON_DESTINO)
        print("\n✅ El archivo JSON fue generado con éxito. Ya puedes ejecutar speech_engine.py para escuchar los audios rápidos.")