# control_logic.py
import time
from typing import List
from contracts import GameEvent, CommentarySegment, PipelineResult

def filtrar_y_generar_voz(eventos: List[GameEvent]) -> PipelineResult:
    """
    📌 ESTRUCTURA MOCK / PLACEHOLDER (Responsabilidad de Persona B)
    Este método es un puente provisional que simula la lógica de filtrado (cooldown)
    y generación de audio de la Persona B, permitiendo que Punto5.py funcione
    de punta a punta con el trabajo completado de la Persona A.
    """
    print("\n[Persona B - Mock] Recibiendo eventos reales de Persona A...")
    
    start_time = time.time()
    segmentos = []
    
    # Simular lógica de cooldown: ignorar eventos que estén a menos de 1.5s del anterior
    last_timestamp = -999.0
    for ev in eventos:
        time_diff = ev.timestamp - last_timestamp
        if time_diff >= 1.5:
            # Pasa el filtro
            print(f"  [Filtro] Evento aprobado en {ev.timestamp}s: \"{ev.commentary_text}\"")
            
            # Simular la ruta de audio correspondiente
            audio_mock_path = f"outputs/audio_{ev.timestamp}s.wav"
            
            segmento = CommentarySegment(
                timestamp=ev.timestamp,
                text=ev.commentary_text,
                audio_path=audio_mock_path
            )
            segmentos.append(segmento)
            last_timestamp = ev.timestamp
        else:
            print(f"  [Filtro] Evento descartado en {ev.timestamp}s (cooldown de {time_diff:.1f}s)")
            
    execution_time = round(time.time() - start_time, 2)
    
    # Empaquetar el resultado final del Pipeline
    resultado = PipelineResult(
        video_path="partida.mp4",
        total_frames_processed=len(eventos),
        execution_time_seconds=execution_time,
        segments=segmentos
    )
    
    print(f"[Persona B - Mock] Simulación completada. {len(segmentos)} segmentos de comentario aprobados.")
    return resultado
