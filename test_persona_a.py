# test_persona_a.py
import os
import cv2
import numpy as np

# Importar las clases de contrato y la función de Persona A
from contracts import GameEvent
from persona_a.vision_engine import procesar_video_real

def create_dummy_video(path="partida.mp4", duration_seconds=3.0, fps=10):
    """
    Crea un video mp4 de prueba de forma programática.
    """
    print(f"[Test] Creando video de prueba '{path}' de {duration_seconds}s...")
    width, height = 320, 240
    # Usar el codec universal mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    total_frames = int(duration_seconds * fps)
    for i in range(total_frames):
        # Generar un fotograma con cambios de color y texto dinámico
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Simular escenas diferentes en el video
        if i < total_frames // 3:
            # Escena 1: Un pasillo oscuro (Fondo gris)
            frame[:, :] = [40, 40, 40]
            cv2.putText(frame, "Scene 1: Walking", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.rectangle(frame, (100, 100), (220, 180), (100, 100, 100), -1)
        elif i < 2 * total_frames // 3:
            # Escena 2: Un combate con disparos (Fondo rojo)
            frame[:, :] = [0, 0, 150]
            cv2.putText(frame, "Scene 2: Shooting", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.circle(frame, (160, 120), 40, (0, 255, 255), -1)
        else:
            # Escena 3: Victoria final (Fondo azul/verde)
            frame[:, :] = [0, 150, 0]
            cv2.putText(frame, "Scene 3: Victory", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "WINNER!", (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 3)
            
        out.write(frame)
        
    out.release()
    print("[Test] ¡Video de prueba creado exitosamente!")

def run_test():
    test_video = "partida.mp4"
    
    # Crear video si no existe
    if not os.path.exists(test_video):
        create_dummy_video(test_video, duration_seconds=3.0, fps=10)
        
    print("\n[Test] Iniciando prueba del pipeline de Persona A...")
    try:
        eventos = procesar_video_real(test_video)
        
        print("\n" + "="*50)
        print("RESULTADOS DE LA PERSONA A")
        print("="*50)
        for i, ev in enumerate(eventos):
            print(f"Evento #{i+1}:")
            print(f"  - Segundo: {ev.timestamp}s")
            print(f"  - Descripción visual (BLIP): \"{ev.raw_caption}\"")
            print(f"  - Comentario gamer (GPT-2):  \"{ev.commentary_text}\"")
            
            # Validaciones de tipos y contrato
            assert isinstance(ev, GameEvent), "¡El evento no cumple con el contrato GameEvent!"
            
        print("="*50)
        print("¡TODO CORRECTO! Los contratos se cumplen y el pipeline funciona perfectamente.")
        
    except Exception as e:
        print(f"\n[Test] ❌ ERROR DURANTE LA PRUEBA: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
