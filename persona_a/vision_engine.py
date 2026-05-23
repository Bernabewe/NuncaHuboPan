# vision_engine.py
import os
import yaml
from enum import Enum
from typing import List, Optional

from PIL import Image

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from contracts import GameEvent
from persona_a.video_reader import extract_frames


# ---------------------------------------------------------------------------
# 1. CLASIFICADOR DE ESTADO DE JUEGO
# ---------------------------------------------------------------------------

class EstadoJuego(Enum):
    EXPLORACION = "exploracion"
    COMBATE     = "combate"
    VICTORIA    = "victoria"
    DERROTA     = "derrota"
    NEUTRO      = "neutro"

_KEYWORDS_ESTADO: dict[EstadoJuego, list[str]] = {
    EstadoJuego.VICTORIA: [
        "victory", "win", "winning", "won", "winner", "triumph",
        "defeated", "game over", "champion", "celebration", "podium",
    ],
    EstadoJuego.DERROTA: [
        "defeat", "dead", "death", "dying", "eliminated", "respawn",
        "game over screen", "skull",
    ],
    EstadoJuego.COMBATE: [
        "gun", "shooting", "shot", "weapon", "fight", "fighting", "battle",
        "firing", "bullet", "explode", "explosion", "knife", "sword",
        "attack", "attacking", "aiming", "rifle", "pistol", "grenade",
        "blood", "kill", "enemy",
    ],
    EstadoJuego.EXPLORACION: [
        "walking", "running", "hallway", "corridor", "looking",
        "sneaking", "moving", "exploring", "background", "standing",
        "crouching", "map", "person", "player", "character", "holding",
        "screen", "game",
    ],
}

def clasificar_estado(vision_caption: str) -> EstadoJuego:
    caption_lower = vision_caption.lower()
    for estado in [EstadoJuego.VICTORIA, EstadoJuego.DERROTA, EstadoJuego.COMBATE, EstadoJuego.EXPLORACION]:
        for kw in _KEYWORDS_ESTADO[estado]:
            if kw in caption_lower:
                return estado
    return EstadoJuego.NEUTRO


# ---------------------------------------------------------------------------
# 2. GENERADOR CREATIVO (LLM INSTRUCT)
# ---------------------------------------------------------------------------

def generate_commentary(
    vision_caption: str,
    estado_actual: EstadoJuego,
    estado_anterior: EstadoJuego,
    comentario_anterior: Optional[str],
    gpt_model,
    gpt_tokenizer,
    device: str,
    max_new_tokens: int = 100,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> str:
    
    # 1. Construir la memoria 
    contexto = ""
    if comentario_anterior:
        contexto = f"Contexto anterior: '{comentario_anterior}'. Sigue narrando sin repetir lo que ya dijiste."
    else:
        contexto = "Esta es la primera jugada."

    # 2. System Prompt Ultra-Estricto
    instrucciones = (
        "Eres un comentarista profesional de eSports. Tu trabajo es narrar la escena descrita."
        "REGLAS INQUEBRANTABLES:\n"
        "1. Escribe UNA SOLA ORACIÓN, máximo dos. Ve directo al grano.\n"
        "2. NUNCA menciones que el jugador está 'frente a una pantalla', 'jugando un videojuego' o 'tecleando'. "
        "Narra como si el mundo del juego fuera la vida real.\n"
        "3. Usa jerga gamer competitiva (pushear, holdear, respawn, rush, lootear).\n"
        "4. NO inventes cosas que no estén en la descripción visual."
    )

    messages = [
        {"role": "system", "content": instrucciones},
        {"role": "user", "content": f"{contexto}\n\nAcción actual en pantalla: '{vision_caption}'. ¡Narra ya!"}
    ]

    try:
        text = gpt_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = gpt_tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = gpt_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=gpt_tokenizer.eos_token_id,
                repetition_penalty=1.15
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        comentario_final = gpt_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        comentario_final = comentario_final.replace('"', '').replace('\n', ' ')
        return f"¡{comentario_final}" if not comentario_final.startswith("¡") else comentario_final

    except Exception as e:
        print(f"  [Error LLM] {e}")
        return f"¡Tensión máxima en este momento de {estado_actual.value}!"


# ---------------------------------------------------------------------------
# 3. CONFIGURACIÓN
# ---------------------------------------------------------------------------

def load_config() -> dict:
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    defaults = {
        "video": {"frame_interval": 1.0},
        "models": {
            "blip": {"name": "vikhyatk/moondream2"},
            "gpt2": {
                "name": "Qwen/Qwen2.5-1.5B-Instruct",
                "max_new_tokens": 100,
                "temperature": 0.6,
                "top_p": 0.9,
            },
        },
    }
    if not os.path.exists(config_path):
        return defaults
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg if cfg else defaults
    except Exception:
        return defaults


# ---------------------------------------------------------------------------
# 4. FUNCIÓN PRINCIPAL EXPUESTA A PUNTO5.PY
# ---------------------------------------------------------------------------

def procesar_video_real(video_path: str) -> List[GameEvent]:
    print(f"\n[Persona A] Iniciando procesamiento del video: {video_path}")

    cfg = load_config()
    frame_interval     = cfg["video"].get("frame_interval", 1.0)
    vision_model_name  = cfg["models"]["blip"].get("name", "vikhyatk/moondream2")
    gpt2_model_name    = cfg["models"]["gpt2"].get("name", "Qwen/Qwen2.5-1.5B-Instruct")
    max_new_tokens     = cfg["models"]["gpt2"].get("max_new_tokens", 100)
    temperature        = cfg["models"]["gpt2"].get("temperature", 0.6)
    top_p              = cfg["models"]["gpt2"].get("top_p", 0.9)

    # 1. Extracción de fotogramas
    print(f"[Persona A] Paso 1: Extrayendo fotogramas cada {frame_interval}s...")
    frames = extract_frames(video_path, frame_interval)
    if not frames:
        print("[Persona A] Sin fotogramas. Abortando.")
        return []
    print(f"[Persona A] {len(frames)} fotogramas extraídos.")

    # 2. Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Persona A] Hardware: {device.upper()}")

    # 3. Inicialización de modelos
    # Nota: trust_remote_code=True es necesario para arquitecturas nuevas como Moondream
    print(f"[Persona A] Paso 2: Cargando Moondream2 ({vision_model_name})...")
    vision_model = AutoModelForCausalLM.from_pretrained(
        vision_model_name, trust_remote_code=True, revision="2024-08-26"
    ).to(device)
    vision_tokenizer = AutoTokenizer.from_pretrained(vision_model_name, revision="2024-08-26")

    print(f"[Persona A] Paso 3: Cargando Modelo de Narración ({gpt2_model_name})...")
    gpt_tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
    gpt_model     = AutoModelForCausalLM.from_pretrained(gpt2_model_name).to(device)

    # 4. Pipeline
    print("[Persona A] Paso 4: Analizando fotogramas con narrador contextual...")
    eventos: List[GameEvent] = []
    estado_anterior  = EstadoJuego.NEUTRO
    comentario_anterior: Optional[str] = None

    for timestamp, pil_image in frames:
        # A) Entendimiento Visual con Moondream2
        try:
            # Codificamos la imagen
            enc_image = vision_model.encode_image(pil_image)
            # Pregunta súper específica para evitar que describa al jugador real
            pregunta = "This is a screenshot of a video game. Describe strictly what is happening INSIDE the game world. What is the character doing or holding?"
            raw_caption = vision_model.answer_question(enc_image, pregunta, vision_tokenizer)
        except Exception as e:
            print(f"  [Advertencia Visión] {timestamp}s: {e}")
            raw_caption = "moving around the map"

        # B) Clasificar el estado
        estado_actual = clasificar_estado(raw_caption)

        # C) Generar comentario usando Qwen
        comentario = generate_commentary(
            vision_caption=raw_caption,
            estado_actual=estado_actual,
            estado_anterior=estado_anterior,
            comentario_anterior=comentario_anterior,
            gpt_model=gpt_model,
            gpt_tokenizer=gpt_tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        estado_emoji = {
            EstadoJuego.EXPLORACION: "🔍",
            EstadoJuego.COMBATE:     "🔫",
            EstadoJuego.VICTORIA:    "🏆",
            EstadoJuego.DERROTA:     "💀",
            EstadoJuego.NEUTRO:      "❓",
        }.get(estado_actual, "")

        print(f"  [{timestamp}s] {estado_emoji} Moondream: \"{raw_caption}\"")
        print(f"         -> Comentario: \"{comentario}\"")

        # D) Guardar evento
        eventos.append(GameEvent(
            timestamp=timestamp,
            raw_caption=raw_caption,
            commentary_text=comentario,
        ))

        estado_anterior = estado_actual
        comentario_anterior = comentario

    print(f"\n[Persona A] Procesamiento completado. {len(eventos)} eventos generados.")
    return eventos