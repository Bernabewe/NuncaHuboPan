# pipeline_final.py
from persona_a.vision_engine import procesar_video_real
from persona_b.control_logic import filtrar_y_generar_voz

# 1. Persona A procesa el video real y genera los eventos
eventos_reales = procesar_video_real("partida.mp4")

# 2. Persona B recibe esos eventos reales en lugar del JSON mock
resultado_final = filtrar_y_generar_voz(eventos_reales)

# 3. ¡Listo! Esto se le entrega a la interfaz de Gradio para mostrarlo