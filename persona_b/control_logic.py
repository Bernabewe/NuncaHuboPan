import sys
sys.path.append("C:\\libs")  
sys.path.append("..")
import json
import os
from typing import List

# Importación de las estructuras de datos compartidas
from contracts import GameEvent

def cargar_eventos_mock(ruta_json: str) -> List[GameEvent]:
    """ Lee el archivo JSON simulado y lo transforma en objetos GameEvent. """
    with open(ruta_json, 'r', encoding='utf-8') as f:
        datos = json.load(f)
    
    # Convierte cada elemento del JSON al contrato establecido
    return [
        GameEvent(
            timestamp=item['timestamp'],
            raw_caption=item.get('raw_caption', ''),
            commentary_text=item['description']
        ) for item in datos
    ]

def filtrar_eventos_cooldown(eventos: List[GameEvent], cooldown: float = 5.0) -> List[GameEvent]:
    """ 
    Evalúa la marca de tiempo de cada evento para decidir si la IA debe 
    hablar o ignorar la escena con base en un tiempo de espera mínimo.
    """
    eventos_aprobados = []
    ultimo_timestamp_hablado = -float('inf')

    for evento in eventos:
        # El algoritmo verifica si transcurrió el tiempo configurado desde el último comentario
        if evento.timestamp - ultimo_timestamp_hablado >= cooldown:
            eventos_aprobados.append(evento)
            ultimo_timestamp_hablado = evento.timestamp
        else:
            print(f"🚫 Evento ignorado en seg {evento.timestamp}: Bloqueado por Cooldown.")
            
    return eventos_aprobados