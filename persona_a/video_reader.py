import cv2
import os
from typing import List, Tuple

def calcular_similitud_histograma(img1, img2) -> float:
    """
    Compara matemáticamente dos imágenes usando sus histogramas de color en canales HSV.
    Devuelve un valor entre 0.0 (completamente diferentes) y 1.0 (idénticas).
    """
    if img1 is None or img2 is None:
        return 0.0
        
    # Convertimos a HSV para que los cambios de iluminación afecten menos que los cambios de contenido
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    # Calculamos el histograma para los canales de Matiz (H) y Saturación (S)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    
    # Normalizamos los histogramas
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    # Comparamos usando Correlación (1.0 es coincidencia perfecta)
    similitud = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return max(0.0, similitud)

def extraer_keyframes_dinamicos(ruta_video: str, umbral_similitud: float = 0.95) -> List[Tuple[float, cv2.Mat]]:
    """
    Procesa el video extrayendo solo 1 fotograma por segundo y aplicando el filtro
    de estancamiento de pantalla. Devuelve una lista de tuplas (segundo, imagen).
    """
    if not os.path.exists(ruta_video):
        raise FileNotFoundError(f"No se encontró el archivo de video en: {ruta_video}")
        
    cap = cv2.VideoCapture(ruta_video)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    
    if fps_video == 0:
        raise ValueError("No se pudo leer la tasa de FPS del video.")

    keyframes_validos = []
    ultimo_frame_procesado = None
    segundo_actual = 0.0
    
    print("\n🔍 --- FILTRO OPENCV: MUESTREO Y DETECCIÓN DE ESTANCAMIENTO ---")
    
    while cap.isOpened():
        # Calculamos el número exacto del frame correspondiente al segundo actual
        frame_id = int(segundo_actual * fps_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        
        exito, frame = cap.read()
        if not exito:
            break # Fin del video
            
        # Aplicamos el filtro de histograma si ya tenemos un frame previo de referencia
        if ultimo_frame_procesado is not None:
            similitud = calcular_similitud_histograma(ultimo_frame_procesado, frame)
            
            if similitud >= umbral_similitud:
                print(f"🚫 Seg {segundo_actual:.1f}s: Pantalla estática ({similitud*100:.1f}% idéntica). Ignorado.")
                segundo_actual += 1.0
                continue
                
        # Si pasó el filtro, lo guardamos como un fotograma clave válido
        keyframes_validos.append((segundo_actual, frame))
        ultimo_frame_procesado = frame.copy()
        
        print(f"📸 Seg {segundo_actual:.1f}s: Cambio de pantalla detectado. Fotograma capturado.")
        segundo_actual += 1.0
        
    cap.release()
    print(f"📊 Filtro finalizado: De {int(segundo_actual * fps_video)} frames redujimos a solo {len(keyframes_validos)} para la IA.")
    return keyframes_validos

# --- MAIN DE PRUEBAS PARA LA PERSONA A ---
if __name__ == "__main__":
    # Ajusta aquí la ruta a tu video de 30 segundos para tus pruebas locales
    RUTA_TEST = "../fixtures/partida_test.mp4"
    
    # Creamos un video dummy rápido si no tienes uno para que el código no truene
    if not os.path.exists(RUTA_TEST):
        os.makedirs("../fixtures", exist_ok=True)
        print(f"⚠️ Coloca tu video real de 30 segundos en '{os.path.abspath(RUTA_TEST)}'")
    else:
        try:
            resultados = extraer_keyframes_dinamicos(RUTA_TEST)
            print(f"✅ ¡Prueba de VideoReader exitosa! Elementos extraídos: {len(resultados)}")
        except Exception as e:
            print(f"❌ Error en la prueba: {e}")