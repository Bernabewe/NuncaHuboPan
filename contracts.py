################################################################################
# Define las estructuras de datos principales para la aplicacion. 
# Contiene las clases que representan las predicciones de fotogramas, eventos de 
# juego, segmentos de comentarios y resultados del pipeline.
################################################################################

from dataclasses import dataclass, field
from typing import List, Optional

# Define la estructura para la prediccion de un fotograma
@dataclass
class FramePrediction:
    timestamp: float
    raw_caption: str
    confidence: float = 1.0

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

# Define la estructura del resultado final del procesamiento
@dataclass
class PipelineResult:
    video_path: str
    total_frames_processed: int
    execution_time_seconds: float
    segments: List[CommentarySegment] = field(default_factory=list)