import os
import sys
import html
import asyncio
import feedparser
import edge_tts
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

RSS_FEEDS = [
    "https://observatorio.tec.mx/feed/",
    "https://www.edsurge.com/articles_rss",
    "https://intef.es/feed/",
]

VOZ_TTS = "es-MX-DaliaNeural"
ARCHIVO_SALIDA = "episodio_hoy.mp3"

# Modelo actualizado recomendado explícitamente por el servicio de Gemini
MODELO_GEMINI = "gemini-3.8-flash"

# ---------------------------------------------------------------------------
# 1. Extracción de Noticias RSS
# ---------------------------------------------------------------------------

def limpiar_texto(texto: str) -> str:
    import re
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = html.unescape(texto)
    return re.sub(r"\s+", " ", texto).strip()[:600]

def _parsear_feed(url: str) -> list[str]:
    try:
        feed = feedparser.parse(url)
        entradas = getattr(feed, "entries", [])
        items = []
        for entry in entradas[:2]:
            titulo = limpiar_texto(getattr(entry, "title", "Sin título"))
            resumen = limpiar_texto(getattr(entry, "summary", getattr(entry, "description", "Sin resumen")))
            items.append(f"Título: {titulo}\nResumen: {resumen}\n---")
        return items
    except Exception as e:
        print(f"[Error RSS] {url}: {e}")
        return []

async def obtener_titulares() -> str:
    # Ejecuta el parseo de feeds en hilos separados de forma limpia
    tareas = [asyncio.to_thread(_parsear_feed, url) for url in RSS_FEEDS]
    resultados = await asyncio.gather(*tareas)
    return "\n".join(item for grupo in resultados for item in grupo)

# ---------------------------------------------------------------------------
# 2. Generación del Guion con Gemini API
# ---------------------------------------------------------------------------

def _generar_guion_sincrono(api_key: str, prompt: str) -> str:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODELO_GEMINI,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.7)
    )
    if not response.text:
        raise ValueError("La respuesta de la API de Gemini volvió vacía.")
    return response.text

async def generar_guion(texto_noticias: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("CRÍTICO: La variable GEMINI_API_KEY no está configurada en los secretos de GitHub.")

    prompt = f"""
    Actúa como un locutor profesional de podcast de noticias sobre educación y tecnología educativa.
    Con la información de las siguientes noticias, redacta un guion dinámico, claro y coherente a una sola voz, en español.

    RESTRICCIONES ESTRICTAS:
    1. La lectura del guion debe durar entre 3 y 4 minutos (aproximadamente 500 palabras).
    2. NO incluyas acotaciones de sonido, marcadores de tono, ni indicaciones entre paréntesis o corchetes.
    3. Escribe ÚNICAMENTE el texto que será leído textualmente por el sintetizador de voz.
    4. No uses formato Markdown (sin asteriscos ni viñetas).

    Noticias fuente:
    {texto_noticias}
    """

    print(f"   Invocando API de Gemini con modelo: {MODELO_GEMINI}...")
    try:
        # Ejecuta la llamada síncrona en un hilo separado para proteger el Event Loop
        guion_texto = await asyncio.to_thread(_generar_guion_sincrono, api_key, prompt)
        return guion_texto
    except genai_errors.APIError as e:
        print(f"[ERROR GEMINI API] Código HTTP {e.code} | Mensaje: {e.message}")
        raise RuntimeError(f"Fallo de API Gemini (HTTP {e.code}): {e.message}")
    except Exception as e:
        print(f"[ERROR EN GENERACIÓN DE GUION] {type(e).__name__}: {e}")
        raise e

# ---------------------------------------------------------------------------
# 3. Conversión de Texto a Audio MP3
# ---------------------------------------------------------------------------

async def texto_a_audio(texto_guion: str) -> None:
    comunicador = edge_tts.Communicate(texto_guion, VOZ_TTS)
    await comunicador.save(ARCHIVO_SALIDA)

# ---------------------------------------------------------------------------
# Ejecución Principal
# ---------------------------------------------------------------------------

async def ejecutar_pipeline() -> None:
    print("[1/3] Obteniendo noticias de fuentes RSS...")
    noticias = await obtener_titulares()
    if not noticias:
        noticias = "Título: Avances en IA Educativa.\nResumen: Herramientas emergentes para la innovación en el aprendizaje digital."

    print("[2/3] Generando guion con Gemini API...")
    guion = await generar_guion(noticias)
    print(f"   Guion redactado con éxito ({len(guion.split())} palabras).")

    print("[3/3] Sintetizando voz a MP3 con Edge-TTS...")
    await texto_a_audio(guion)
    print(f"¡Proceso completado! Archivo '{ARCHIVO_SALIDA}' generado correctamente.")

if __name__ == "__main__":
    try:
        asyncio.run(ejecutar_pipeline())
    except Exception as e:
        print(f"::error title=Fallo del Pipeline::{e}")
        sys.exit(1)
