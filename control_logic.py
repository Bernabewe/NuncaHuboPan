################################################################################
# Contiene la logica de control para cargar eventos simulados
# y filtrar eventos segun un tiempo de enfriamiento (cooldown).
################################################################################

import sys
import json
import os
from typing import List

from contracts import GameEvent

# Carga eventos simulados desde un archivo JSON
def cargar_eventos_mock(ruta_json: str) -> List[GameEvent]:
    with open(ruta_json, 'r', encoding='utf-8') as f:
        datos = json.load(f)
    
    return [
        GameEvent(
            timestamp=item['timestamp'],
            raw_caption=item.get('raw_caption', ''),
            commentary_text=item['description']
        ) for item in datos
    ]

# Filtra eventos para asegurar que no se solapen en funcion de un tiempo de espera
def filtrar_eventos_cooldown(eventos: List[GameEvent], cooldown: float = 5.0) -> List[GameEvent]:
    eventos_aprobados = []
    ultimo_timestamp_hablado = -float('inf')

    for evento in eventos:
        if evento.timestamp - ultimo_timestamp_hablado >= cooldown:
            eventos_aprobados.append(evento)
            ultimo_timestamp_hablado = evento.timestamp
            
    return eventos_aprobados