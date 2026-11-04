import os
import re
import html
import asyncio
import feedparser
import edge_tts
from google import genai
from google.genai import errors as genai_errors

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

RSS_FEEDS = [
    "https://observatorio.tec.mx/feed/",
    "https://www.edsurge.com/articles_rss",
    "https://intef.es/feed/",
]

# Modelo principal: se puede cambiar sin tocar el código definiendo la variable
# de entorno GEMINI_MODEL (por ejemplo en el workflow de GitHub Actions).
MODELO_PRINCIPAL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

# Si el modelo principal devuelve 404, se prueban estos en orden.
MODELOS_RESPALDO = [
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash",
]

VOZ_TTS = "es-MX-DaliaNeural"
ARCHIVO_SALIDA = "episodio_hoy.mp3"
NOTICIAS_POR_FEED = 2
MAX_CARACTERES_RESUMEN = 700


# ---------------------------------------------------------------------------
# 1. Extracción de noticias
# ---------------------------------------------------------------------------

def limpiar_html(texto: str) -> str:
    """Quita etiquetas HTML y entidades de los resúmenes RSS."""
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = html.unescape(texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto[:MAX_CARACTERES_RESUMEN]


def obtener_titulares() -> str:
    """Extrae las noticias más recientes de los feeds RSS configurados."""
    noticias = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            if getattr(feed, "entries", None):
                for entry in feed.entries[:NOTICIAS_POR_FEED]:
                    titulo = limpiar_html(getattr(entry, "title", "Sin título"))
                    resumen = limpiar_html(
                        getattr(entry, "summary", getattr(entry, "description", "Sin resumen"))
                    )
                    noticias.append(f"Título: {titulo}\nResumen: {resumen}\n---")
            else:
                print(f"[Advertencia] No se encontraron entradas en: {url}")
        except Exception as e:
            print(f"[Error RSS] Fallo al leer {url}: {e}")
    return "\n".join(noticias)


# ---------------------------------------------------------------------------
# 2. Generación del guion con Gemini
# ---------------------------------------------------------------------------

def construir_prompt(texto_noticias: str) -> str:
    return f"""
    Actúa como un locutor profesional de podcast de noticias sobre educación y tecnología educativa.
    Con la información de las siguientes noticias, redacta un guion dinámico, claro y coherente a una sola voz.

    RESTRICCIONES ESTRICTAS:
    1. La lectura del guion debe durar entre 3 y 4 minutos (aproximadamente entre 450 y 550 palabras).
    2. NO incluyas acotaciones de sonido, marcadores de tono, ni indicaciones entre paréntesis o corchetes.
    3. Escribe ÚNICAMENTE el texto que será leído textualmente por el sintetizador de voz.
    4. No uses formato Markdown (sin asteriscos, almohadillas ni viñetas).
    5. Estructura el contenido con una breve bienvenida, el desarrollo de las noticias principales y una despedida rápida.

    Noticias fuente:
    {texto_noticias}
    """


def lista_de_modelos() -> list[str]:
    """Modelo principal primero, luego respaldos sin repetir."""
    vistos, lista = set(), []
    for m in [MODELO_PRINCIPAL, *MODELOS_RESPALDO]:
        if m not in vistos:
            vistos.add(m)
            lista.append(m)
    return lista


async def generar_guion(texto_noticias: str) -> str:
    """Envía las noticias a Gemini y devuelve el guion. Prueba modelos de respaldo si hay 404."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "CRÍTICO: La variable GEMINI_API_KEY no está configurada en los secretos de GitHub."
        )

    client = genai.Client(api_key=api_key)
    prompt = construir_prompt(texto_noticias)
    ultimo_error = None

    for modelo in lista_de_modelos():
        for intento in range(1, 4):  # hasta 3 intentos por modelo
            try:
                print(f"   Usando modelo: {modelo} (intento {intento})")
                response = await client.aio.models.generate_content(
                    model=modelo,
                    contents=prompt,
                )
                if not response.text:
                    raise ValueError("La respuesta de Gemini volvió vacía.")
                return response.text.strip()

            except genai_errors.ClientError as e:
                ultimo_error = e
                if e.code == 404:
                    print(f"   [Aviso] Modelo no disponible: {modelo}. Probando el siguiente...")
                    break  # pasa al siguiente modelo
                if e.code == 429:
                    espera = 5 * intento
                    print(f"   [Aviso] Límite de cuota (429). Reintentando en {espera}s...")
                    await asyncio.sleep(espera)
                    continue
                raise  # 400, 401, 403: errores de configuración, no tiene caso reintentar

            except genai_errors.ServerError as e:
                ultimo_error = e
                espera = 5 * intento
                print(f"   [Aviso] Error del servidor de Gemini. Reintentando en {espera}s...")
                await asyncio.sleep(espera)

    raise RuntimeError(f"No se pudo generar el guion con ningún modelo. Último error: {ultimo_error}")


# ---------------------------------------------------------------------------
# 3. Síntesis de voz
# ---------------------------------------------------------------------------

async def texto_a_audio(texto_guion: str, archivo_salida: str = ARCHIVO_SALIDA) -> None:
    """Convierte el guion en un MP3 con Edge-TTS (con reintentos)."""
    ultimo_error = None
    for intento in range(1, 4):
        try:
            comunicador = edge_tts.Communicate(texto_guion, VOZ_TTS)
            await comunicador.save(archivo_salida)
            if not os.path.exists(archivo_salida) or os.path.getsize(archivo_salida) == 0:
                raise RuntimeError("El archivo de audio se generó vacío.")
            return
        except Exception as e:
            ultimo_error = e
            print(f"   [Aviso] Falló edge-tts (intento {intento}): {e}")
            await asyncio.sleep(3 * intento)
    raise RuntimeError(f"No se pudo generar el audio. Último error: {ultimo_error}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def ejecutar_pipeline() -> None:
    print("[1/3] Extrayendo noticias de fuentes RSS...")
    noticias = obtener_titulares()

    if not noticias:
        print("[Alerta] No se pudieron extraer noticias RSS, usando contenido de contingencia...")
        noticias = (
            "Título: Avances en Inteligencia Artificial y Educación.\n"
            "Resumen: Estrategias emergentes para el aprendizaje adaptativo y la "
            "personalización en el aula moderna."
        )

    print("[2/3] Generando guion con Gemini API...")
    guion = await generar_guion(noticias)
    print(f"Guion generado con éxito ({len(guion.split())} palabras).")

    print("[3/3] Sintetizando voz y generando archivo MP3...")
    await texto_a_audio(guion)
    print(f"¡Proceso completado! El archivo '{ARCHIVO_SALIDA}' ha sido generado exitosamente.")


if __name__ == "__main__":
    asyncio.run(ejecutar_pipeline())
