################################################################################
# Gestiona la sintesis de voz utilizando el modelo VITS. 
# Convierte el texto de los eventos filtrados en archivos de audio de forma 
# optimizada.
################################################################################

import sys
import os
import warnings
from transformers import logging

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

import torch
import scipy.io.wavfile as wavfile
from transformers import VitsModel, AutoTokenizer
from typing import List

from contracts import GameEvent, CommentarySegment
from control_logic import cargar_eventos_mock, filtrar_eventos_cooldown

# Inicializa el modelo de sintesis de voz
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-spa")
model = VitsModel.from_pretrained("facebook/mms-tts-spa")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Genera archivos de audio a partir de los eventos usando el modelo VITS
def generar_audio_segmentos(eventos_filtrados: List[GameEvent]) -> List[CommentarySegment]:
    os.makedirs("outputs", exist_ok=True)
    segmentos_finales = []

    for evento in eventos_filtrados:
        inputs = tokenizer(evento.commentary_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model(**inputs).waveform
        
        audio_data = output.cpu().numpy().squeeze()
        
        ruta_audio = f"outputs/audio_{evento.timestamp}s.wav"
        
        sampling_rate_original = model.config.sampling_rate
        wavfile.write(ruta_audio, rate=sampling_rate_original, data=audio_data)
        
        segmento = CommentarySegment(
            timestamp=evento.timestamp,
            text=evento.commentary_text,
            audio_path=ruta_audio
        )
        segmentos_finales.append(segmento)
        
    return segmentos_finales