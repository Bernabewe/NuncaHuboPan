# contracts.py
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class FramePrediction:
    """
    📌 RELACIÓN CON EL PROYECTO:
    • Corresponde al: PUNTO #1 (Módulo de Captura/Muestreo) y PUNTO #2 (Módulo de Visión).
    • Tipo de dato: Es la SALIDA del bloque de Visión (BLIP) y la ENTRADA directa del bloque de Narración (GPT-2).
    
    🔗 CÓMO SE RELACIONA:
    La Persona A usa OpenCV para extraer un cuadro del video (Punto 1) y se lo pasa a BLIP (Punto 2). 
    BLIP procesa la imagen y empaqueta el resultado en esta clase.
    """
    timestamp: float         # Segundo exacto del video donde se tomó la captura (ej: 4.5)
    raw_caption: str         # La descripción cruda en inglés que escupe BLIP (ej: "a character fighting a monster")
    confidence: float = 1.0  # Nivel de certeza del modelo de visión (útil para descartar frames borrosos)


@dataclass
class GameEvent:
    """
    📌 RELACIÓN CON EL PROYECTO:
    • Corresponde al: PUNTO #2 (Módulo de Narración - Texto / GPT-2).
    • Tipo de dato: Es la SALIDA final del trabajo de la Persona A y la ENTRADA del trabajo de la Persona B.
    
    🔗 CÓMO SE RELACIONA:
    La Persona A toma el 'FramePrediction' anterior, le pasa el 'raw_caption' a GPT-2 para generar 
    el comentario creativo en español y guarda todo aquí. Una lista de estos objetos es lo que 
    la Persona A le entrega a la Persona B (o lo que se simula en el archivo JSON mock).
    """
    timestamp: float         # Hereda el segundo exacto del video original
    raw_caption: str         # Se conserva la frase en inglés (sirve a la Persona B para su heurística de similitud)
    commentary_text: str     # El comentario divertido, creativo y en español generado por el LLM


@dataclass
class CommentarySegment:
    """
    📌 RELACIÓN CON EL PROYECTO:
    • Corresponde al: PUNTO #5 (Lógica de Control) y PUNTO #4 (Módulo de Narración - Voz / TTS).
    • Tipo de dato: Es la SALIDA final del trabajo de la Persona B.
    
    🔗 CÓMO SE RELACIONA:
    La Persona B recibe los 'GameEvent' de la Persona A. Primero los pasa por el filtro de cooldown (Punto 5). 
    Los eventos que sobreviven al filtro se mandan al modelo MMS-TTS-SPA (Punto 4) para generar el archivo de audio. 
    El texto final aprobado junto con la ruta de su audio se guardan en esta clase.
    """
    timestamp: float         # Segundo en el que el bot detectó el cambio significativo y decidió hablar
    text: str                # El comentario en español que se va a mostrar en la pantalla
    audio_path: str          # Ruta local donde la Persona B guardó el archivo de voz (ej: 'outputs/audio_4.5s.wav')


@dataclass
class PipelineResult:
    """
    📌 RELACIÓN CON EL PROYECTO:
    • Corresponde al: PUNTO #5 (Interfaz / Gradio o Streamlit) e Integración Final.
    • Tipo de dato: Es la SALIDA total de todo el sistema unificado.
    
    🔗 CÓMO SE RELACIONA:
    Cuando el video termina de procesarse por completo, este objeto junta las métricas generales del sistema 
    y la lista de todos los 'CommentarySegment' generados. Es la estructura limpia que leerá la interfaz gráfica 
    para pintar los textos en pantalla y reproducir los audios de manera sincronizada.
    """
    video_path: str                  # Ruta del video que subió el usuario en la UI
    total_frames_processed: int       # Cuántos fotogramas se analizaron en total (para estadísticas)
    execution_time_seconds: float     # Cuánto tiempo tardó la IA en procesar todo el video
    segments: List[CommentarySegment] = field(default_factory=list) # Lista con todos los audios y textos aprobados