import os
import sys
import html
import re
import asyncio
from datetime import datetime
import feedparser
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# ---------------------------------------------------------------------------
# 1. Nombres Dinámicos por Fecha (Ej: episodio_ep05112026.mp3)
# ---------------------------------------------------------------------------

FECHA_STR = datetime.now().strftime("%d%m%Y")
ARCHIVO_SALIDA = f"episodio_ep{FECHA_STR}.mp3"
ARCHIVO_GUION = f"guion_ep{FECHA_STR}.txt"

RSS_FEEDS = [
    "https://www.unesco.org/en/rss/news/field_topic/education-38",
    "https://www.weforum.org/agenda/topics/education-and-skills/feed",
    "https://observatorio.tec.mx/feed/",
    "https://intef.es/feed/",
    "https://www.unam.mx/rss.xml",
    "https://www.edsurge.com/articles_rss",
    "https://www.insidehighered.com/rss.xml",
    "https://www.edweek.org/technology.rss",
    "https://www.k12dive.com/feeds/news/",
    "https://www.eschoolnews.com/category/top-news/feed/",
    "https://www.techlearning.com/feeds.xml",
    "https://www.kqed.org/mindshift/feed/",
    "https://mfeldstein.com/feed/"
]

MODELOS_TEXTO = [
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest"
]

MODELO_TTS = "gemini-3.8-flash-tts"

# ---------------------------------------------------------------------------
# 2. Extracción de RSS
# ---------------------------------------------------------------------------

def limpiar_texto(texto: str) -> str:
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = html.unescape(texto)
    return re.sub(r"\s+", " ", texto).strip()[:700]

def _parsear_feed(url: str) -> list[str]:
    try:
        feed = feedparser.parse(url)
        nombre_fuente = feed.feed.get('title', url)
        entradas = getattr(feed, "entries", [])
        items = []
        for entry in entradas[:2]:
            titulo = limpiar_texto(getattr(entry, "title", "Sin título"))
            resumen = limpiar_texto(getattr(entry, "summary", getattr(entry, "description", "Sin resumen")))
            link = getattr(entry, "link", "")
            items.append(
                f"FUENTE OFICIAL: {nombre_fuente}\n"
                f"ENLACE: {link}\n"
                f"TÍTULO: {titulo}\n"
                f"RESUMEN: {resumen}\n---"
            )
        return items
    except Exception as e:
        print(f"[Error RSS] {url}: {e}")
        return []

async def obtener_titulares() -> str:
    tareas = [asyncio.to_thread(_parsear_feed, url) for url in RSS_FEEDS]
    resultados = await asyncio.gather(*tareas)
    noticias_filtradas = [item for grupo in resultados for item in grupo if item]
    return "\n".join(noticias_filtradas[:12])

# ---------------------------------------------------------------------------
# 3. Prompt Maestro para "Eduradar" con Expresiones Emocionales (<breath>, <chuckle>)
# ---------------------------------------------------------------------------

def construir_prompt_maestro(texto_noticias: str) -> str:
    meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    ahora = datetime.now()
    fecha_hoy = f"{ahora.day} de {meses[ahora.month - 1]} de {ahora.year}"

    return f"""
Actúa como un equipo de dos periodistas y locutores profesionales de podcast sobre educación y EdTech:
- SOFÍA: Presentadora principal, analítica, apasionada por las políticas educativas y la pedagogía.
- ALEJANDRO: Co-presentador, dinámico, enfocado en tecnología de aprendizaje e impacto práctico.

Tu objetivo es redactar un guion en formato de DIÁLOGO FLUIDO Y EXPRESIVO para el podcast "Eduradar".

TARGET DE EXTENSIÓN: Entre 750 y 950 PALABRAS (5 a 7 minutos).

ESTRUCTURA Y EXPRESIVIDAD EXPLICITA (SÍNTESIS GEMINI TTS):
1. APERTURA: Saludo caluroso, nombre "Eduradar" y fecha exacta "{fecha_hoy}".
2. CITA DE FUENTES: Atribución explícita a la institución o fuente oficial en cada nota.
3. EXPRESIONES VOCALES: Integra de manera natural dentro del texto las etiquetas de inflexión vocal para el modelo de voz:
   - Usa `<breath>` para pausas de respiración naturales.
   - Usa `<chuckle>` para momentos amables o ligeros en el diálogo.
   - Usa `<enthusiastic>` para enfatizar noticias o hallazgos importantes.
   - Usa `<astonishment>` para datos sorprendentes.
4. CIERRE: Pregunta provocadora para la comunidad y llamado a la acción (CTA) para seguir el podcast.

RESTRICCIONES DE ETIQUETAS:
- Cada línea DEBE comenzar exactamente con `SOFÍA:` o `ALEJANDRO:`.
- NO agregues acotaciones escénicas ni texto entre corchetes rectos `[...]`.
- NO uses formato Markdown (sin asteriscos, sin negritas).

NOTICIAS FUENTE:
{texto_noticias}
"""

def _llamar_modelo(client: genai.Client, modelo: str, prompt: str) -> str:
    response = client.models.generate_content(
        model=modelo,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.65)
    )
    if not response.text:
        raise ValueError("La respuesta de la API volvió vacía.")
    return response.text

async def generar_guion(texto_noticias: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("CRÍTICO: La variable GEMINI_API_KEY no está configurada en los secretos de GitHub.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=120_000,
            retry_options=types.HttpRetryOptions(attempts=3, initial_delay=3.0, max_delay=15.0)
        )
    )

    prompt = construir_prompt_maestro(texto_noticias)
    errores = []

    for modelo in MODELOS_TEXTO:
        for intento in range(1, 3):
            print(f"   Intentando guion con modelo: {modelo} (Intento {intento}/2)...")
            try:
                guion_texto = await asyncio.to_thread(_llamar_modelo, client, modelo, prompt)
                print(f"   [Éxito] Guion generado con {modelo}.")
                return guion_texto
            except genai_errors.APIError as e:
                msg = f"{modelo} -> HTTP {e.code}: {e.message}"
                print(f"   [Aviso API] {msg}")
                errores.append(msg)
                if e.code in (401, 403):
                    break
                elif e.code == 503:
                    await asyncio.sleep(10)
            except Exception as e:
                msg = f"{modelo} -> {type(e).__name__}: {e}"
                print(f"   [Error] {msg}")
                errores.append(msg)
                break

    raise RuntimeError(f"Ningún modelo estuvo disponible. Resumen:\n" + "\n".join(errores))

def guardar_guion_texto(guion_texto: str) -> None:
    with open(ARCHIVO_GUION, "w", encoding="utf-8") as f:
        f.write(guion_texto)
    print(f"   [OK] Guion guardado en '{ARCHIVO_GUION}'.")

# ---------------------------------------------------------------------------
# 4. Síntesis Expresiva con Gemini 3.8 Flash TTS (Voces Jori y Rami)
# ---------------------------------------------------------------------------

def _sintetizar_bloque_gemini(client: genai.Client, texto: str, voz: str) -> bytes:
    # Configuración de voz con Gemini Flash TTS
    response = client.models.generate_content(
        model=MODELO_TTS,
        contents=texto,
        config=types.GenerateContentConfig(
            response_mime_type="audio/mp3",
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voz)
                )
            )
        )
    )
    return response.candidates[0].content.parts[0].inline_data.data

async def texto_a_audio_gemini_tts(guion_texto: str) -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    client = genai.Client(api_key=api_key)
    
    lineas = guion_texto.strip().split("\n")
    bloques_audio = []
    
    for idx, linea in enumerate(lineas):
        linea = linea.strip()
        if not linea:
            continue
            
        voz_activa = "Jori" # Voz por defecto para Sofía
        texto_hablado = linea
        
        if linea.startswith("SOFÍA:") or linea.startswith("SOFIA:"):
            voz_activa = "Jori"
            texto_hablado = re.sub(r"^SOFÍ?A:\s*", "", linea, flags=re.IGNORECASE)
        elif linea.startswith("ALEJANDRO:"):
            voz_activa = "Rami"
            texto_hablado = re.sub(r"^ALEJANDRO:\s*", "", linea, flags=re.IGNORECASE)

        texto_hablado = html.unescape(texto_hablado).strip()

        if len(texto_hablado) > 2:
            temp_filename = f"fragmento_{idx}.mp3"
            try:
                audio_bytes = await asyncio.to_thread(_sintetizar_bloque_gemini, client, texto_hablado, voz_activa)
                with open(temp_filename, "wb") as f:
                    f.write(audio_bytes)
                bloques_audio.append(temp_filename)
            except Exception as e:
                print(f"   [Aviso TTS Gemini] Error en bloque {idx}: {e}")

    if not bloques_audio:
        raise ValueError("No se generó ningún fragmento de audio válido.")

    # Concatenación de fragmentos en el archivo final con nombre por fecha
    with open(ARCHIVO_SALIDA, "wb") as outfile:
        for temp_file in bloques_audio:
            if os.path.exists(temp_file):
                with open(temp_file, "rb") as infile:
                    outfile.write(infile.read())
                os.remove(temp_file)

# ---------------------------------------------------------------------------
# Pipeline Principal
# ---------------------------------------------------------------------------

async def ejecutar_pipeline() -> None:
    print(f"[1/3] Obteniendo noticias RSS para el episodio {FECHA_STR}...")
    noticias = await obtener_titulares()

    print("[2/3] Generando guion expresivo con Gemini API...")
    guion = await generar_guion(noticias)
    guardar_guion_texto(guion)

    print(f"[3/3] Sintetizando audio expresivo (Gemini TTS: Jori y Rami) -> {ARCHIVO_SALIDA}...")
    await texto_a_audio_gemini_tts(guion)
    print(f"¡Hito completado! Archivos '{ARCHIVO_SALIDA}' y '{ARCHIVO_GUION}' listos.")

if __name__ == "__main__":
    try:
        asyncio.run(ejecutar_pipeline())
    except Exception as e:
        print(f"::error title=Fallo del Pipeline::{e}")
        sys.exit(1)
