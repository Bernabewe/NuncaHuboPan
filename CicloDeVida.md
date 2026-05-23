# Ciclo de Vida de la Aplicación

Todo el ciclo de vida de la aplicación se activa en el momento en que el usuario entra a la interfaz y sube un archivo de video.

Para que lo veas con total claridad, imagínalo como una línea de producción en una fábrica donde el video es la materia prima, la Persona A construye la primera mitad de la máquina, y la Persona B construye la segunda mitad.

Aquí tienes el viaje completo del video paso a paso y en qué punto exacto entra el trabajo de cada uno:

---

## 📥 El Disparador (La Interfaz)

El usuario sube un clip de video de 20 segundos (por ejemplo, una jugada de League of Legends o Halo) en la pantalla de Gradio. Al dar clic en **"Procesar"**, el archivo se envía al código de fondo.

---

## 🏗️ Tramo 1: El Trabajo de la PERSONA A (Visión y Narración en Texto)

Aquí entra en acción el pipeline que programó la Persona A. Su trabajo consiste en recibir ese video crudo y transformarlo en ideas escritas:

*   **Extracción de Frames:** El script de la Persona A (`video_reader.py`) toma el video y, usando OpenCV, va extrayendo un fotograma (una imagen fija) cada 1 o 2 segundos. No analiza todo el video de golpe porque sería lentísimo; lo va dosificando.
*   **Entendimiento Visual (BLIP):** Esa imagen pasa al modelo de Inteligencia Artificial BLIP. BLIP "observa" la captura de pantalla y genera una etiqueta en inglés muy fría y descriptiva, por ejemplo: `"a character shooting a weapon"` (un personaje disparando un arma).
*   **Generación del Comentario (GPT-2):** La Persona A toma esa frase en inglés y se la pasa al modelo de lenguaje GPT-2 junto con una instrucción (un prompt). GPT-2 procesa la idea y redacta un comentario creativo, con estilo gamer y en español: `"¡Madre mía, qué ráfaga de disparos! ¡Está arrinconando al enemigo!"`
*   **El "Entregable" de la Persona A:** Al terminar de revisar el video, el código de la Persona A entrega una lista ordenada de objetos `GameEvent` (los contratos que definieron juntos). Cada evento dice el segundo exacto y el texto del comentario.

---

## 🚚 El Puente (Los Contratos)

En un flujo normal, la lista de comentarios en texto que generó la Persona A pasa de inmediato al código de la Persona B.

> [!NOTE]
> Durante los días de desarrollo, la Persona B usó el archivo JSON simulado para hacer pruebas sin necesitar este bloque de la Persona A.

---

## 🏗️ Tramo 2: El Trabajo de la PERSONA B (Filtros y Voz)

Aquí arranca el pipeline de la Persona B, cuyo trabajo es decidir qué comentarios valen la pena y darles una voz real:

*   **El Filtro Inteligente (Lógica de Control):** La Persona B recibe la lista de comentarios de la Persona A y le aplica su código de heurística (`control_logic.py`). Si en el segundo 4 el bot dijo algo emocionante, y en el segundo 5 el modelo de visión sigue repitiendo algo muy similar, la lógica de la Persona B aplica un cooldown (tiempo de espera). Dice: `"Silencio, el bot ya habló hace un segundo, ignoremos este frame para no saturar al usuario"`.
*   **Generación de Voz (MMS-TTS):** Los comentarios que sí lograron pasar el filtro de la Persona B son enviados al modelo de audio de Facebook. Este modelo toma el texto en español y genera un archivo de audio real (`.wav`) con la narración hablada.
*   **El "Entregable" de la Persona B:** Una lista limpia de objetos `CommentarySegment` que contienen el texto final y la ruta del archivo de audio grabado en el disco duro.

---

## 🏁 El Resultado Final (De vuelta a la Interfaz)

La aplicación toma los audios y textos que devolvió la Persona B y actualiza la pantalla de Gradio. El usuario ahora puede ver su video original y, en la sección de comentarios, ir escuchando y leyendo las reacciones que la IA generó de forma automática con las marcas de tiempo exactas.

¡Ese es todo el mapa del flujo! Como puedes ver, están perfectamente coordinados: **la Persona A** se encarga de que la IA entienda y escriba, y **la Persona B** se encarga de que la IA piense cuándo hablar y ejecute la voz.
