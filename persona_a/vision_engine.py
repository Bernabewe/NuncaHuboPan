"""
vision_pipeline.py
------------------
Extrae frames de un video, los analiza con Llama Vision via Groq
y genera una narración fluida de la partida.

Uso:
    python vision_pipeline.py --video ruta/al/video.mp4

Variables de entorno (.env):
    GROQ_API_KEY  → tu API key de https://console.groq.com/keys
"""

import os
import cv2
import time
import json
import base64
import argparse
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from groq import Groq

# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)

# =========================
# CONFIG
# =========================

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
VISION_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"
NARRADOR_MODEL = "llama-3.3-70b-versatile"

FRAME_INTERVAL       = 1.0
SIMILARITY_THRESHOLD = 0.05
API_DELAY            = 0.2
API_MAX_RETRIES      = 3

# =========================
# CLIENTE GROQ
# =========================

_client: Optional[Groq] = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise EnvironmentError(
                "GROQ_API_KEY no configurado.\n"
                "Crea un .env con: GROQ_API_KEY=gsk_tukey\n"
                "Obtén tu key en: https://console.groq.com/keys"
            )
        log.debug("Inicializando cliente Groq con key %s...", GROQ_API_KEY[:8])
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


# =========================
# FRAME EXTRACTION
# =========================

def extraer_frames(video_path: str, tmp_dir: Optional[str] = None):
    video_path = str(Path(video_path).resolve())
    log.info("Abriendo video: %s", video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duracion = round(total_frames / fps, 1)
    log.info("Duración del video: %.1fs", duracion)

    interval_frames = max(1, int(fps * FRAME_INTERVAL))
    save_dir = tmp_dir or tempfile.mkdtemp(prefix="vp_frames_")
    log.info("FPS=%.1f  intervalo=%d frames  carpeta=%s", fps, interval_frames, save_dir)

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
                    log.debug("  ↩ frame_%04d omitido (diff=%.4f)", saved_idx, score)
                    frame_idx += 1
                    continue

            timestamp = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2)
            path = str(Path(save_dir) / f"frame_{saved_idx:04d}.jpg")
            cv2.imwrite(path, frame)

            frames.append({"timestamp": timestamp, "path": path})
            log.info("  📸 frame_%04d  t=%.2fs", saved_idx, timestamp)

            last_frame = frame.copy()
            saved_idx += 1

        frame_idx += 1

    cap.release()
    log.info("Frames extraídos: %d  (%.1fs de video)", len(frames), duracion)
    return frames, duracion


# =========================
# PASO 1 — DESCRIBIR CADA FRAME (detallado)
# =========================

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
                            "text": (
                                "Describe en una frase concreta qué acción ocurre en este frame de gameplay. "
                                "Sé muy específico: ¿qué hace el jugador exactamente en este momento? "
                                "¿Dispara, se mueve, saquea, recibe daño, elimina a alguien, recoge algo? "
                                "¿Hay algún texto importante en pantalla como kills, loot, zona, etc.? "
                                "Evita frases genéricas como 'el jugador apunta' sin más detalle. "
                                "Responde solo con la descripción, sin introducción."
                            )
                        }
                    ]
                }]
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            last_err = e
            log.warning("  ⚠️ intento %d/%d falló: %s", attempt, API_MAX_RETRIES, repr(e))
            log.debug("  Traceback:", exc_info=True)
            if attempt < API_MAX_RETRIES:
                espera = 2 ** attempt
                log.info("  ⏳ Reintentando en %ds…", espera)
                time.sleep(espera)

    log.error("  ❌ Frame sin describir: %s", repr(last_err))
    return "escena no disponible"


# =========================
# PASO 2 — NARRACIÓN FLUIDA
# =========================

def generar_narracion(eventos: List[Dict], duracion: float) -> str:
    log.info("🎙️ Generando narración fluida...")

    contexto = "\n".join(
        f"[{e['timestamp']:.0f}s] {e['descripcion_raw']}"
        for e in eventos
    )

    max_palabras = int(duracion * 3)

    prompt = f"""Eres un narrador de videojuegos estilo caster de esports, con energía, ritmo y carisma.
Tienes los siguientes {len(eventos)} momentos de una partida de {duracion:.0f} segundos:

{contexto}

Tu tarea: escribe la narración completa en español como si estuvieras comentando en vivo, momento a momento.

Reglas estrictas:
- Máximo {max_palabras} palabras en total
- Cubre TODOS los momentos importantes, sin saltarte nada relevante
- Conecta los momentos con transiciones naturales ("de repente", "acto seguido", "sin perder tiempo"...)
- Varía el ritmo: oraciones cortas y rápidas en combate, algo más descriptivo al saquear o moverse
- Usa vocabulario de gamer: "elimina", "saquea", "zona", "blindaje", "scope", "down", etc.
- NO incluyas timestamps ni corchetes en el resultado
- Devuelve SOLO el texto de la narración, sin títulos ni explicaciones"""

    try:
        response = _get_client().chat.completions.create(
            model=NARRADOR_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        narracion = response.choices[0].message.content.strip()
        log.info("✅ Narración generada (%d palabras)", len(narracion.split()))
        return narracion

    except Exception as e:
        log.error("❌ Error generando narración: %s", repr(e))
        return ""


# =========================
# PIPELINE COMPLETO
# =========================

def procesar_video(video_path: str) -> Dict:
    log.info("=== PIPELINE VISIÓN + NARRACIÓN (Groq) ===")

    frames, duracion = extraer_frames(video_path)

    if not frames:
        log.warning("No se extrajeron frames del video.")
        return {"eventos": [], "narracion": "", "duracion": 0}

    # Paso 1: describir cada frame con detalle
    eventos: List[Dict] = []
    for i, f in enumerate(frames, 1):
        log.info("[%d/%d] Describiendo t=%.2fs …", i, len(frames), f["timestamp"])
        desc = describir_frame(f["path"])
        log.info("  → %s", desc)
        eventos.append({
            "timestamp":       f["timestamp"],
            "descripcion_raw": desc,
        })
        time.sleep(API_DELAY)

    # Paso 2: narración fluida con todo el contexto
    narracion = generar_narracion(eventos, duracion)

    log.info("\n" + "="*50)
    log.info("🎙️  NARRACIÓN FINAL:\n\n%s\n", narracion)
    log.info("="*50)

    return {
        "duracion":  duracion,
        "narracion": narracion,
        "eventos":   eventos,
    }


# =========================
# OUTPUT
# =========================

def guardar_json(resultado: Dict, output_path: str = "mock_events.json") -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    log.info("💾 Guardado en: %s", out)


# =========================
# CLI
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="Analiza y narra una partida con Groq Vision.")
    parser.add_argument("--video",  required=True,                       help="Ruta al archivo de video")
    parser.add_argument("--output", default="fixtures/mock_events.json", help="Ruta del JSON de salida")
    args = parser.parse_args()

    if not Path(args.video).exists():
        raise FileNotFoundError(f"Video no encontrado: {args.video}")

    resultado = procesar_video(args.video)
    guardar_json(resultado, args.output)
    log.info("✅ DONE")


if __name__ == "__main__":
    main()