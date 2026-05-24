################################################################################
# Coordina la extraccion de fotogramas, su descripcion 
# mediante un modelo de vision y la generacion de la narracion fluida del 
# video completo.
################################################################################

import os
import cv2
import time
import json
import base64
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from groq import Groq

# Configura variables globales y de entorno
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
VISION_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"
NARRADOR_MODEL = "llama-3.3-70b-versatile"

FRAME_INTERVAL       = 1.0
SIMILARITY_THRESHOLD = 0.05
API_DELAY            = 0.2
API_MAX_RETRIES      = 3

_client: Optional[Groq] = None

# Inicializa y retorna el cliente de la API de Groq
def _get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY no configurado.")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client

# Extrae los fotogramas del video en intervalos definidos descartando repeticiones
def extraer_frames(video_path: str, tmp_dir: Optional[str] = None):
    video_path = str(Path(video_path).resolve())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duracion = round(total_frames / fps, 1)

    interval_frames = max(1, int(fps * FRAME_INTERVAL))
    save_dir = tmp_dir or tempfile.mkdtemp(prefix="vp_frames_")

    frames: List[Dict] = []
    last_frame = None
    frame_idx  = 0
    saved_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval_frames == 0:
            if last_frame is not None:
                diff  = cv2.absdiff(frame, last_frame)
                score = diff.mean() / 255.0
                if score < SIMILARITY_THRESHOLD:
                    frame_idx += 1
                    continue

            timestamp = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2)
            path = str(Path(save_dir) / f"frame_{saved_idx:04d}.jpg")
            cv2.imwrite(path, frame)

            frames.append({"timestamp": timestamp, "path": path})

            last_frame = frame.copy()
            saved_idx += 1

        frame_idx += 1

    cap.release()
    return frames, duracion

# Genera una descripcion detallada para un fotograma especifico
def describir_frame(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    last_err: Exception = Exception("sin intentos")

    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            response = _get_client().chat.completions.create(
                model=VISION_MODEL,
                max_tokens=120,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                        },
                        {
                            "type": "text",
                            "text": "Describe en una frase concreta que accion ocurre en este frame de gameplay. Se muy especifico. Evita frases genericas. Responde solo con la descripcion, sin introduccion."
                        }
                    ]
                }]
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            last_err = e
            if attempt < API_MAX_RETRIES:
                espera = 2 ** attempt
                time.sleep(espera)

    return "escena no disponible"

# Crea una narracion continua a partir de las descripciones de los eventos
def generar_narracion(eventos: List[Dict], duracion: float) -> str:
    contexto = "\n".join(
        f"[{e['timestamp']:.0f}s] {e['descripcion_raw']}"
        for e in eventos
    )

    max_palabras = int(duracion * 3)

    prompt = f"Eres un narrador de videojuegos. Tienes {len(eventos)} momentos de una partida de {duracion:.0f} segundos:\n\n{contexto}\n\nEscribe la narracion completa en espanol. Maximo {max_palabras} palabras. Devuelve SOLO el texto de la narracion."

    try:
        response = _get_client().chat.completions.create(
            model=NARRADOR_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return ""

# Ejecuta el flujo de trabajo completo desde el video hasta la narracion
def procesar_video(video_path: str) -> Dict:
    frames, duracion = extraer_frames(video_path)

    if not frames:
        return {"eventos": [], "narracion": "", "duracion": 0}

    eventos: List[Dict] = []
    for i, f in enumerate(frames, 1):
        desc = describir_frame(f["path"])
        eventos.append({
            "timestamp":       f["timestamp"],
            "descripcion_raw": desc,
        })
        time.sleep(API_DELAY)

    narracion = generar_narracion(eventos, duracion)

    return {
        "duracion":  duracion,
        "narracion": narracion,
        "eventos":   eventos,
    }

# Guarda los resultados en un archivo JSON
def guardar_json(resultado: Dict, output_path: str = "mock_events.json") -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

def main() -> None:
    parser = argparse.ArgumentParser(description="Analiza y narra una partida con Groq Vision.")
    parser.add_argument("--video",  required=True,                       help="Ruta al archivo de video")
    parser.add_argument("--output", default="fixtures/mock_events.json", help="Ruta del JSON de salida")
    args = parser.parse_args()

    if not Path(args.video).exists():
        raise FileNotFoundError(f"Video no encontrado: {args.video}")

    resultado = procesar_video(args.video)
    guardar_json(resultado, args.output)