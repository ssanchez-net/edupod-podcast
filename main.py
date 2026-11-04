import os
import sys
import html
import re
import asyncio
import feedparser
import edge_tts
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# ---------------------------------------------------------------------------
# 1. Configuración de Fuentes RSS Ampliadas (Globales y Latinoamérica)
# ---------------------------------------------------------------------------

RSS_FEEDS = [
    # Institucionales y Multilaterales
    "https://www.unesco.org/en/rss/news/field_topic/education-38",
    "https://www.weforum.org/agenda/topics/education-and-skills/feed",
    # América Latina y España
    "https://observatorio.tec.mx/feed/",
    "https://intef.es/feed/",
    "https://www.unam.mx/rss.xml",
    # Periodismo Especializado y Tecnología Educativa
    "https://www.edsurge.com/articles_rss",
    "https://www.insidehighered.com/rss.xml",
    "https://www.edweek.org/technology.rss",
    "https://www.k12dive.com/feeds/news/",
    "https://www.eschoolnews.com/category/top-news/feed/",
    "https://www.techlearning.com/feeds.xml",
    "https://www.kqed.org/mindshift/feed/",
    "https://mfeldstein.com/feed/"
]

# Configuración de Voces Neurónicas (Edge-TTS)
VOZ_FEMENINA = "es-MX-DaliaNeural"   # Sofía
VOZ_MASCULINA = "es-MX-JorgeNeural"   # Alejandro
ARCHIVO_SALIDA = "episodio_hoy.mp3"

# Candidatos de modelos compatibles
MODELOS_CANDIDATOS = [
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest"
]

# ---------------------------------------------------------------------------
# 2. Extracción de Noticias RSS con Atribución de Fuente
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
    return "\n".join(noticias_filtradas[:10])

# ---------------------------------------------------------------------------
# 3. Prompt Maestro para Podcast Conversacional (2 Voces)
# ---------------------------------------------------------------------------

def construir_prompt_maestro(texto_noticias: str) -> str:
    return f"""
Actúa como un equipo de dos periodistas y locutores de podcast sobre educación y EdTech: 
- SOFÍA: Presentadora principal, analítica, entusiasta de la pedagogía y la inclusión.
- ALEJANDRO: Co-presentador, enfocado en tecnología educativa, datos clave e impacto práctico en el aula.

Tu objetivo es redactar un guion en formato de DIÁLOGO FLUIDO Y NATURAL entre Sofía y Alejandro para un episodio de 5 a 7 minutos de duración (entre 750 y 950 palabras).

REGLAS ESTRICTAS DE REDACCIÓN Y ESTRUCTURA:
1. FORMATO DE ETIQUETAS: Cada intervención DEBE comenzar obligatoriamente con `SOFÍA:` o `ALEJANDRO:` en una nueva línea.
2. CITACIÓN OFICIAL OBLIGATORIA: En cada noticia explicada, uno de los locutores DEBE mencionar de forma explícita la fuente institucional o medio de donde se extrajo la información (ej. "Según publica hoy el Observatorio del Tec de Monterrey...", "Un informe del World Economic Forum destaca...", "En una reciente publicación del portal oficial de la UNAM...").
3. CONTEXTO COMPLETO: Incluyan detalles precisos como universidades, países, nombres de iniciativas, fechas y plataformas disponibles. No dejen información a medias.
4. DINÁMICA CONVERSACIONAL NATURAL: Intercambien comentarios, realicen preguntas retóricas entre ustedes y agreguen breves frases de transición ("Así es, Alejandro", "Un punto clave que mencionas, Sofía"). Incluyan un llamado a la acción (CTA) orgánico al público para consultar las fuentes.
5. RESTRICCIÓN DE FORMATO TTS: Escriban ÚNICAMENTE el diálogo con las etiquetas `SOFÍA:` y `ALEJANDRO:`. NO agreguen acotaciones entre paréntesis ni corchetes (nada de `[risas]` o `[pausa]`), ni usen formato Markdown (sin asteriscos, sin viñetas).

NOTICIAS FUENTE DISPONIBLES:
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
            timeout=110_000,
            retry_options=types.HttpRetryOptions(attempts=2, initial_delay=2.0, max_delay=10.0)
        )
    )

    prompt = construir_prompt_maestro(texto_noticias)

    errores = []
    for modelo in MODELOS_CANDIDATOS:
        print(f"   Intentando generación con modelo: {modelo}...")
        try:
            guion_texto = await asyncio.to_thread(_llamar_modelo, client, modelo, prompt)
            print(f"   [Éxito] Guion conversacional generado con {modelo}.")
            return guion_texto
        except genai_errors.APIError as e:
            msg = f"{modelo} -> HTTP {e.code}: {e.message}"
            print(f"   [Aviso API] {msg}")
            errores.append(msg)
            if e.code in (401, 403):
                break
        except Exception as e:
            msg = f"{modelo} -> {type(e).__name__}: {e}"
            print(f"   [Error] {msg}")
            errores.append(msg)

    raise RuntimeError(f"Ningún modelo estuvo disponible. Resumen de errores:\n" + "\n".join(errores))

# ---------------------------------------------------------------------------
# 4. Síntesis Multimóvil a Dos Voces (Ensemble de Audio)
# ---------------------------------------------------------------------------

async def texto_a_audio_dos_voces(guion_texto: str) -> None:
    lineas = guion_texto.strip().split("\n")
    bloques_audio = []
    
    # Procesamiento y separación de intervenciones por personaje
    for idx, linea in enumerate(lineas):
        linea = linea.strip()
        if not linea:
            continue
            
        voz_activa = None
        texto_hablado = ""
        
        if linea.startswith("SOFÍA:") or linea.startswith("SOFIA:"):
            voz_activa = VOZ_FEMENINA
            texto_hablado = re.sub(r"^SOFÍ?A:\s*", "", linea, flags=re.IGNORECASE)
        elif linea.startswith("ALEJANDRO:"):
            voz_activa = VOZ_MASCULINA
            texto_hablado = re.sub(r"^ALEJANDRO:\s*", "", linea, flags=re.IGNORECASE)
        else:
            # Si una línea no tiene etiqueta, hereda la voz por defecto (Sofía)
            voz_activa = VOZ_FEMENINA
            texto_hablado = linea

        if texto_hablado.strip():
            temp_filename = f"fragmento_{idx}.mp3"
            comunicador = edge_tts.Communicate(texto_hablado, voz_activa)
            await comunicador.save(temp_filename)
            bloques_audio.append(temp_filename)

    if not bloques_audio:
        raise ValueError("No se pudieron extraer bloques de diálogo para sintetizar.")

    # Concatena todos los fragmentos en un solo archivo final MP3
    with open(ARCHIVO_SALIDA, "wb") as outfile:
        for temp_file in bloques_audio:
            if os.path.exists(temp_file):
                with open(temp_file, "rb") as infile:
                    outfile.write(infile.read())
                os.remove(temp_file) # Limpiar archivo temporal

# ---------------------------------------------------------------------------
# Pipeline Principal
# ---------------------------------------------------------------------------

async def ejecutar_pipeline() -> None:
    print("[1/3] Obteniendo noticias de fuentes RSS (Ampliadas)...")
    noticias = await obtener_titulares()
    if not noticias:
        noticias = (
            "FUENTE OFICIAL: Observatorio del Instituto para el Futuro de la Educación del Tec de Monterrey\n"
            "TÍTULO: Innovación en la Educación Superior Latinoamericana\n"
            "RESUMEN: Análisis sobre la adopción de herramientas de IA en universidades de México y Chile.\n---"
        )

    print("[2/3] Generando guion conversacional (2 Voces) con Gemini API...")
    guion = await generar_guion(noticias)
    palabras = len(guion.split())
    print(f"   Guion redactado con éxito ({palabras} palabras, aprox. {round(palabras/140, 1)} minutos).")

    print("[3/3] Sintetizando voces neurónicas (Sofía y Alejandro) a MP3...")
    await texto_a_audio_dos_voces(guion)
    print(f"¡Proceso completado! Archivo '{ARCHIVO_SALIDA}' generado correctamente.")

if __name__ == "__main__":
    try:
        asyncio.run(ejecutar_pipeline())
    except Exception as e:
        print(f"::error title=Fallo del Pipeline::{e}")
        sys.exit(1)
