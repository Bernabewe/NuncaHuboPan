# video_reader.py
import cv2
from PIL import Image
import os
from typing import List, Tuple

def extract_frames(video_path: str, interval_seconds: float = 1.0) -> List[Tuple[float, Image.Image]]:
    """
    📌 PASO #1: Módulo de Captura/Muestreo
    Extrae fotogramas de un archivo de video en intervalos constantes de tiempo.
    
    Args:
        video_path (str): Ruta al archivo de video (.mp4, .avi, etc.).
        interval_seconds (float): Intervalo de tiempo en segundos entre fotogramas.
        
    Returns:
        List[Tuple[float, Image.Image]]: Una lista de tuplas conteniendo (segundo_exacto, PIL.Image.Image).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError(f"Invalid video metadata: FPS={fps}, Total Frames={total_frames}")
        
    duration = total_frames / fps
    frames = []
    
    # Muestrear fotogramas en marcas de tiempo específicas (0s, 0s + intervalo, etc.)
    current_time = 0.0
    while current_time < duration:
        frame_idx = int(current_time * fps)
        if frame_idx >= total_frames:
            break
            
        # Posicionar el lector de video en el fotograma calculado para mayor velocidad
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        
        if not success:
            # Si falla la lectura por saltos, se detiene o continúa según disponibilidad
            break
            
        # OpenCV lee en formato BGR, convertimos a RGB para trabajar con PIL y BLIP
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Guardar el segundo exacto y la imagen
        frames.append((round(current_time, 2), pil_img))
        current_time += interval_seconds
        
    cap.release()
    return frames
