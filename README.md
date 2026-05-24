# Narrador IA para Videojuegos

## ¿Qué hace este proyecto?
Este proyecto toma un clip de video de una partida (por ejemplo, de Warzone), lo analiza visualmente para entender qué está pasando en la pantalla, y genera de manera automática una narración épica y emocionante con voz, sincronizada perfectamente con la acción del video. Todo esto se presenta a través de una interfaz web sencilla y fácil de usar.

---

## Tecnologías y Librerías Utilizadas

- **Gradio:** Utilizada para construir la interfaz gráfica de usuario. Es la capa visible donde subes tu video y visualizas tanto la tabla de decisiones en tiempo real como el resultado final.
- **OpenCV (`cv2`):** Es la librería encargada del procesamiento del video. Extrae los fotogramas (imágenes) y detecta los cambios en la pantalla usando histogramas de color para ignorar escenas aburridas o repetitivas.
- **Groq API & Modelos Llama:** Actúan como los "ojos" y la "creatividad" del proyecto. Llama Vision interpreta las imágenes para entender la acción, y un modelo de lenguaje (LLM) transforma esa interpretación aburrida en un comentario épico de esports.
- **Transformers (VITS / `facebook/mms-tts-spa`):** Es el motor de habla. Sintetiza los textos generados por la IA en audios reales con una voz en español.
- **FFmpeg (vía subprocess):** Funciona como el editor de video en el fondo. Se encarga de mezclar el video original con todos los segmentos de audio generados, calculando los tiempos para que las voces no se encimen.
- **Pandas:** Se usa para almacenar y mostrar el registro histórico de los eventos y decisiones de la IA en forma de una tabla clara en la interfaz.

---

## Ciclo de Vida: Desde el Video hasta la Narración

El proyecto funciona como una línea de ensamblaje. Cuando subes un video a la interfaz y presionas iniciar, ocurre lo siguiente:

1. **`app.py` (El Inicio)**
   Es el director de orquesta. Levanta la interfaz gráfica, recibe tu video y llama a los demás archivos para comenzar el trabajo.

2. **`video_reader.py` (Extracción y Filtrado)**
   Toma el video y lo corta en múltiples imágenes (fotogramas). Compara una imagen con la anterior y descarta aquellas donde la pantalla está estática o no hay cambios importantes. Pasa solo los momentos de acción al siguiente paso.

3. **`vision_engine.py` (Los Ojos de la IA)**
   Recibe los fotogramas clave y los envía a la inteligencia artificial visual. La IA analiza cada imagen y genera una descripción cruda de la acción (ejemplo: "El jugador está disparando a un enemigo a lo lejos").

4. **`control_logic.py` (El Control de Ritmo)**
   Toma los eventos detectados y les aplica una regla de "enfriamiento" (*cooldown*). Se asegura de que el narrador tome aire y no hable sin parar, descartando eventos que ocurren demasiado rápido uno detrás del otro.

5. **`app.py` (La Magia del Comentarista)**
   Las descripciones que sobreviven al filtro regresan al archivo principal. Aquí, otra inteligencia artificial transforma el texto crudo en una frase de caster emocionante y directa (ejemplo: "¡Ráfaga letal, limpieza total en la zona!").

6. **`speech_engine.py` (La Voz)**
   Recibe estas frases heroicas y, utilizando modelos de inteligencia artificial locales (Transformers), genera de manera ultrarrápida archivos de audio `.wav` para cada frase.

7. **Montaje Final (dentro de `app.py`)**
   Finalmente, FFmpeg toma el video original y todos los pequeños archivos de voz. Calcula en qué milisegundo exacto debe entrar cada frase, aplica un filtro "anti-colisiones" para que no se hablen encima, y une todo en un solo video `.mp4` narrado que se te muestra en pantalla.

> **Nota Adicional:** El archivo **`contracts.py`** es utilizado por todos durante este viaje. Funciona como un molde de datos (contrato) para asegurar que la información (segundos, descripciones, rutas de audio) pase de un archivo a otro sin perder el formato.
