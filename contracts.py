# contracts.py
from dataclasses import dataclass

@dataclass
class GameEvent:
    """Clase que representa lo que la IA 've' en un momento específico."""
    timestamp: float    # El segundo exacto del video (ej: 12.5)
    description: str    # La descripción creativa en español generada por GPT-2

@dataclass
class CommentarySegment:
    """Clase que representa el resultado final listo para la interfaz."""
    timestamp: float    # El segundo en que debe reproducirse
    text: str           # El texto del comentario
    audio_path: str     # La ruta del archivo .wav generado para escucharse