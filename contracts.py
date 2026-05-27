################################################################################
# Define las estructuras de datos principales para la aplicacion. 
# Contiene las clases que representan las predicciones de fotogramas, eventos de 
# juego, segmentos de comentarios y resultados del pipeline.
################################################################################

from dataclasses import dataclass, field
from typing import List, Optional

# Define la estructura de un evento de juego procesado
@dataclass
class GameEvent:
    timestamp: float
    raw_caption: str
    commentary_text: str

# Define la estructura para un segmento de comentario en audio
@dataclass
class CommentarySegment:
    timestamp: float
    text: str
    audio_path: str