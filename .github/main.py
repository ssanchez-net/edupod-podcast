import os
import re
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

NOTICIAS_POR_FEED = 2
MAX_CARACTERES_RESUMEN = 700
TIMEOUT_RSS_SEG = 25
TIMEOUT_GEMINI_MS = 90_000      # el SDK usa milisegundos
TIMEOUT_TTS_SEG = 240

VOZ_TTS = "es-MX-DaliaNeural"
ARCHIVO_SALIDA = "episodio_hoy.mp3"

# Si defines GEMINI_MODEL en el workflow, ese modelo se prueba primero.
MODELO_FORZADO = os.environ.get("GEMINI_MODEL", "").strip()
MODELOS_BASE = [
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]


# ---------------------------------------------------------------------------
# 1. RSS (en paralelo)
# ---------------------------------------------------------------------------

def limpiar_html(texto: str) -> str:
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = html.unescape(texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto[:MAX_CARACTERES_RESUMEN]


async def leer_feed(url: str) -> list[str]:
    """Lee un feed sin bloquear el bucle de eventos y con límite de tiempo."""
    try:
        feed = await asyncio.wait_for(
            asyncio.to_thread(feedparser.parse, url), timeout=TIMEOUT_RSS_SEG
        )
    except asyncio.TimeoutError:
        print(f"[Error RSS] Tiempo de espera agotado: {url}")
        return []
    except Exception as e:
        print(f"[Error RSS] Fallo al leer {url}: {e}")
        return []

    entradas = getattr(feed, "entries", None)
    if not entradas:
        print(f"[Advertencia] No se encontraron entradas en: {url}")
        return []

    items = []
    for entry in entradas[:NOTICIAS_POR_FEED]:
        titulo = limpiar_html(getattr(entry, "title", "Sin título"))
        resumen = limpiar_html(
            getattr(entry, "summary", getattr(entry, "description", "Sin resumen"))
        )
        items.append(f"Título: {titulo}\nResumen: {resumen}\n---")
    return items


async def obtener_titulares() -> str:
    resultados = await asyncio.gather(*(leer_feed(u) for u in RSS_FEEDS))
    return "\n".join(item for grupo in resultados for item in grupo)


# ---------------------------------------------------------------------------
# 2. Guion con Gemini
# ---------------------------------------------------------------------------

def construir_prompt(texto_noticias: str) -> str:
    return f"""
    Actúa como un locutor profesional de podcast de noticias sobre educación y tecnología educativa.
    Con la información de las siguientes noticias, redacta un guion dinámico, claro y coherente a una sola voz, en español.

    RESTRICCIONES ESTRICTAS:
    1. La lectura del guion debe durar entre 3 y 4 minutos (aproximadamente entre 450 y 550 palabras).
    2. NO incluyas acotaciones de sonido, marcadores de tono, ni indicaciones entre paréntesis o corchetes.
    3. Escribe ÚNICAMENTE el texto que será leído textualmente por el sintetizador de voz.
    4. No uses formato Markdown (sin asteriscos, almohadillas ni viñetas).
    5. Estructura el contenido con una breve bienvenida, el desarrollo de las noticias principales y una despedida rápida.

    Noticias fuente:
    {texto_noticias}
    """


def limpiar_para_voz(texto: str) -> str:
    texto = re.sub(r"[*#_`]+", "", texto)
    texto = re.sub(r"\[[^\]]*\]|\([^)]*\)", "", texto)
    return re.sub(r"\n{3,}", "\n\n", texto).strip()


def descubrir_modelos(client: genai.Client) -> list[str]:
    """Pregunta a la API qué modelos 'flash' tiene realmente disponibles esta clave."""
    encontrados = []
    try:
        for m in client.models.list():
            nombre = (m.name or "").replace("models/", "")
            acciones = getattr(m, "supported_actions", None) or []
            excluidos = ("image", "tts", "live", "audio", "embedding", "robotics")
            if "flash" in nombre and "generateContent" in acciones and not any(x in nombre for x in excluidos):
                encontrados.append(nombre)
    except Exception as e:
        print(f"[Aviso] No se pudo listar modelos: {type(e).__name__}: {e}")
    return sorted(encontrados, reverse=True)


async def generar_guion(texto_noticias: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("CRÍTICO: GEMINI_API_KEY no está configurada en los secretos de GitHub.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=TIMEOUT_GEMINI_MS,
            # El SDK ya reintenta errores transitorios (429/5xx) con backoff.
            retry_options=types.HttpRetryOptions(attempts=2, initial_delay=2.0, max_delay=15.0),
        ),
    )

    disponibles = await asyncio.to_thread(descubrir_modelos, client)
    print(f"   Modelos flash visibles para esta clave: {disponibles or 'ninguno / no se pudo listar'}")

    candidatos, vistos = [], set()
    for m in [MODELO_FORZADO, *MODELOS_BASE, *disponibles]:
        if m and m not in vistos:
            vistos.add(m)
            candidatos.append(m)

    prompt = construir_prompt(texto_noticias)
    config = types.GenerateContentConfig(temperature=0.7)
    errores = []  # se conservan TODOS, no solo el último

    for modelo in candidatos:
        print(f"   Probando modelo: {modelo}")
        try:
            response = await client.aio.models.generate_content(
                model=modelo, contents=prompt, config=config
            )
            texto = limpiar_para_voz(response.text or "")
            if len(texto.split()) < 150:
                raise ValueError(f"Guion demasiado corto ({len(texto.split())} palabras).")
            print(f"   OK con {modelo}")
            return texto

        except genai_errors.APIError as e:
            linea = f"{modelo} -> HTTP {e.code} {e.status}: {str(e.message)[:200]}"
            print(f"   [Error] {linea}")
            errores.append(linea)
            if e.code in (401, 403):  # clave inválida o sin permisos: no tiene caso seguir
                break
        except Exception as e:
            linea = f"{modelo} -> {type(e).__name__}: {str(e)[:200]}"
            print(f"   [Error] {linea}")
            errores.append(linea)

    resumen = "\n".join(f"  - {x}" for x in errores)
    raise RuntimeError(f"Ningún modelo pudo generar el guion. Errores por modelo:\n{resumen}")


# ---------------------------------------------------------------------------
# 3. Voz
# ---------------------------------------------------------------------------

async def texto_a_audio(texto_guion: str, archivo_salida: str = ARCHIVO_SALIDA) -> None:
    ultimo_error = None
    for intento in range(1, 4):
        try:
            comunicador = edge_tts.Communicate(texto_guion, VOZ_TTS)
            await asyncio.wait_for(comunicador.save(archivo_salida), timeout=TIMEOUT_TTS_SEG)
            if not os.path.exists(archivo_salida) or os.path.getsize(archivo_salida) == 0:
                raise RuntimeError("El MP3 se generó vacío.")
            return
        except Exception as e:
            ultimo_error = e
            print(f"   [Aviso] edge-tts falló (intento {intento}): {type(e).__name__}: {e}")
            await asyncio.sleep(3 * intento)
    raise RuntimeError(f"No se pudo generar el audio: {ultimo_error}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def ejecutar_pipeline() -> None:
    print("[1/3] Extrayendo noticias de fuentes RSS...")
    noticias = await obtener_titulares()
    if not noticias:
        print("[Alerta] Sin noticias RSS, usando contenido de contingencia...")
        noticias = (
            "Título: Avances en Inteligencia Artificial y Educación.\n"
            "Resumen: Estrategias emergentes para el aprendizaje adaptativo y la "
            "personalización en el aula moderna."
        )

    print("[2/3] Generando guion con Gemini API...")
    guion = await generar_guion(noticias)
    print(f"Guion generado ({len(guion.split())} palabras).")

    print("[3/3] Sintetizando voz y generando MP3...")
    await texto_a_audio(guion)
    print(f"¡Listo! '{ARCHIVO_SALIDA}' generado.")


if __name__ == "__main__":
    try:
        asyncio.run(ejecutar_pipeline())
    except Exception as e:
        print(f"::error title=Fallo del pipeline::{type(e).__name__}")
        print(str(e))
        sys.exit(1)
