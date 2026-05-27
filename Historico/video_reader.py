################################################################################
# Procesa archivos de video para extraer fotogramas clave. 
# Aplica filtros para ignorar escenas repetitivas basandose en 
# la similitud de histogramas.
################################################################################

import cv2
import os
from typing import List, Tuple

# Compara la similitud de color entre dos imagenes
def calcular_similitud_histograma(img1, img2) -> float:
    if img1 is None or img2 is None:
        return 0.0
        
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    similitud = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return max(0.0, similitud)

# Extrae fotogramas dinamicos de un video ignorando imagenes estaticas
def extraer_keyframes_dinamicos(ruta_video: str, umbral_similitud: float = 0.95) -> List[Tuple[float, cv2.Mat]]:
    if not os.path.exists(ruta_video):
        raise FileNotFoundError(f"No se encontro el archivo de video en: {ruta_video}")
        
    cap = cv2.VideoCapture(ruta_video)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    
    if fps_video == 0:
        raise ValueError("No se pudo leer la tasa de FPS del video.")

    keyframes_validos = []
    ultimo_frame_procesado = None
    segundo_actual = 0.0
    
    while cap.isOpened():
        frame_id = int(segundo_actual * fps_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        
        exito, frame = cap.read()
        if not exito:
            break 
            
        if ultimo_frame_procesado is not None:
            similitud = calcular_similitud_histograma(ultimo_frame_procesado, frame)
            
            if similitud >= umbral_similitud:
                segundo_actual += 1.0
                continue
                
        keyframes_validos.append((segundo_actual, frame))
        ultimo_frame_procesado = frame.copy()
        
        segundo_actual += 1.0
        
    cap.release()
    return keyframes_validos