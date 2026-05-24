import sys
sys.path.append("..")
import os
import json
import torch
from PIL import Image
import cv2
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# --- SILENCIAR WARNINGS Y LOGS DE TERMINAL ---
import warnings
from transformers import logging as tf_logging
warnings.filterwarnings("ignore")
tf_logging.set_verbosity_error()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# --------------------------------------------

from video_reader import extraer_keyframes_dinamicos
from contracts import GameEvent

# Modelo de visión de alta velocidad y libre acceso de la serie Qwen
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

print("⏳ Cargando VLM de alta eficiencia libre de candados (Qwen2-VL)...")

# Inicialización del procesador y modelo nativo
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

def analizar_video_y_generar_json(ruta_video: str, ruta_salida_json: str):
    """
    Procesa el video filtrando frames estáticos y generando descripciones cortas.
    """
    # 1. Obtener los fotogramas clave limpios desde OpenCV
    keyframes = extraer_keyframes_dinamicos(ruta_video)
    eventos_detectados = []
    
    print("\n🧠 --- PROCESAMIENTO OPTIMIZADO CON QWEN2-VL ---")
    
    for timestamp, frame in keyframes:
        print(f"🤖 Analizando escena del segundo {timestamp}s...")
        
        # Conversión de BGR (OpenCV) a RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Estructura de mensajes estándar para modelos instructivos modernos
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": "Describe en una sola frase corta y emocionante en español qué acción está pasando en este videojuego."}
                ]
            }
        ]
        
        # Preparación del prompt estructurado
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = processor.image_processor(images=pil_image, videos=None), None
        
        inputs = processor(
            text=[text],
            images=pil_image,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Generación veloz restringiendo los tokens de salida
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=30)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            descripcion = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        descripcion_limpia = descripcion.strip()
        print(f"   ↳ [IA]: {descripcion_limpia}")
        
        # Construcción y apego estricto al contrato de datos
        evento = GameEvent(
            timestamp=float(timestamp),
            commentary_text=descripcion_limpia
        )
        
        eventos_detectados.append({
            "timestamp": evento.timestamp,
            "commentary_text": evento.commentary_text
        })
        
    # 2. Escritura del archivo JSON final para el uso de la Persona B
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
        print("\n🚀 INICIANDO ENTORNO DE VISIÓN CONTROLADO 🚀")
        analizar_video_y_generar_json(VIDEO_DE_PRUEBA, JSON_DESTINO)