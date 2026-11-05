import os
import sys
import html
import re
import asyncio
from datetime import datetime
import feedparser
import edge_tts
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# ---------------------------------------------------------------------------
# 1. Configuración de Fuentes RSS Ampliadas
# ---------------------------------------------------------------------------

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

VOZ_FEMENINA = "es-MX-DaliaNeural"   # Sofía
VOZ_MASCULINA = "es-MX-JorgeNeural"   # Alejandro
ARCHIVO_SALIDA = "episodio_hoy.mp3"
ARCHIVO_GUION = "guion_episodio.txt"

MODELOS_CANDIDATOS = [
    "gemini-2.5-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest"
]

# ---------------------------------------------------------------------------
# 2. Extracción RSS
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
# 3. Prompt Maestro para "Eduradar" (5 a 7 Minutos)
# ---------------------------------------------------------------------------

def construir_prompt_maestro(texto_noticias: str) -> str:
    meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    ahora = datetime.now()
    fecha_hoy = f"{ahora.day} de {meses[ahora.month - 1]} de {ahora.year}"

    return f"""
Actúa como un equipo de dos periodistas y locutores profesionales de podcast sobre educación y EdTech:
- SOFÍA: Presentadora principal, analítica, apasionada por las políticas educativas, la inclusión y la pedagogía.
- ALEJANDRO: Co-presentador, dinámico, enfocado en tecnología de aprendizaje, datos duros e impacto práctico en el aula.

Tu objetivo es redactar un guion en formato de DIÁLOGO FLUIDO, DINÁMICO Y RIGUROSO entre Sofía y Alejandro para el podcast "Eduradar".

TARGET DE EXTENSIÓN Y DENSIDAD (5 A 7 MINUTOS):
- El guion DEBE contener entre 750 y 950 PALABRAS en total.

ESTRUCTURA DEL GUION:
1. APERTURA Y BIENVENIDA DIRECTA:
   - Bienvenida profesional.
   - Mencionen el nombre del podcast: "Eduradar".
   - Mencionen la fecha de hoy: "{fecha_hoy}".
2. DESARROLLO NOTICIOSO PERIODÍSTICO:
   - De 3 a 5 noticias clave del material fuente.
   - CITACIÓN OFICIAL OBLIGATORIA a la fuente emisora.
   - CONTEXTO COMPLETO (datos, países, universidades).
3. CIERRE DINÁMICO E INTERACTIVO:
   - PREGUNTA PROVOCADORA Y DEBATIBLE para los foros de la plataforma.
   - Llamado a la acción (CTA) para seguir el podcast.
   - Despedida rápida a dos voces.

RESTRICCIONES ESTRICTAS PARA TTS:
- Cada línea DEBE iniciar estrictamente con `SOFÍA:` o `ALEJANDRO:`.
- NO agregues acotaciones entre paréntesis ni corchetes (NADA de `[pausa]`, `[risas]`).
- NO uses formato Markdown (SIN asteriscos, SIN encabezados `#`, SIN viñetas `-`).
- Escribe ÚNICAMENTE el texto limpio para locución.

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
            timeout=120_000,
            retry_options=types.HttpRetryOptions(attempts=3, initial_delay=3.0, max_delay=15.0)
        )
    )

    prompt = construir_prompt_maestro(texto_noticias)
    errores = []

    for modelo in MODELOS_CANDIDATOS:
        for intento in range(1, 3):
            print(f"   Intentando generación con modelo: {modelo} (Intento {intento}/2)...")
            try:
                guion_texto = await asyncio.to_thread(_llamar_modelo, client, modelo, prompt)
                print(f"   [Éxito] Guion conversacional para 'Eduradar' generado con {modelo}.")
                return guion_texto
            except genai_errors.APIError as e:
                msg = f"{modelo} (intento {intento}) -> HTTP {e.code}: {e.message}"
                print(f"   [Aviso API] {msg}")
                errores.append(msg)
                
                if e.code in (401, 403):
                    break
                elif e.code == 503:
                    print("   [Servidor Saturado 503] Pausando 15 segundos para recuperar conexión...")
                    await asyncio.sleep(15)
            except Exception as e:
                msg = f"{modelo} -> {type(e).__name__}: {e}"
                print(f"   [Error] {msg}")
                errores.append(msg)
                break

    raise RuntimeError(f"Ningún modelo estuvo disponible tras reintentos por saturación. Resumen de errores:\n" + "\n".join(errores))

def guardar_guion_texto(guion_texto: str) -> None:
    """Guarda el guion generado en un archivo de texto (.txt) independiente."""
    with open(ARCHIVO_GUION, "w", encoding="utf-8") as f:
        f.write(guion_texto)
    print(f"   [OK] Guion guardado exitosamente en '{ARCHIVO_GUION}'.")

# ---------------------------------------------------------------------------
# 4. Síntesis Multimóvil Robusta a Dos Voces (Edge-TTS)
# ---------------------------------------------------------------------------

async def texto_a_audio_dos_voces(guion_texto: str) -> None:
    lineas = guion_texto.strip().split("\n")
    bloques_audio = []
    
    for idx, linea in enumerate(lineas):
        linea = linea.strip()
        if not linea:
            continue
            
        voz_activa = VOZ_FEMENINA
        texto_hablado = linea
        
        if linea.startswith("SOFÍA:") or linea.startswith("SOFIA:"):
            voz_activa = VOZ_FEMENINA
            texto_hablado = re.sub(r"^SOFÍ?A:\s*", "", linea, flags=re.IGNORECASE)
        elif linea.startswith("ALEJANDRO:"):
            voz_activa = VOZ_MASCULINA
            texto_hablado = re.sub(r"^ALEJANDRO:\s*", "", linea, flags=re.IGNORECASE)

        texto_hablado = re.sub(r"[*#_`]+", "", texto_hablado)
        texto_hablado = re.sub(r"\[.*?\]|\(.*?\)", "", texto_hablado)
        texto_hablado = html.unescape(texto_hablado).strip()

        if len(texto_hablado) > 2 and re.search(r"\w", texto_hablado):
            temp_filename = f"fragmento_{idx}.mp3"
            try:
                comunicador = edge_tts.Communicate(texto_hablado, voz_activa)
                await comunicador.save(temp_filename)
                if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 0:
                    bloques_audio.append(temp_filename)
            except Exception as e:
                print(f"   [Aviso TTS] No se pudo sintetizar fragmento {idx}: {e}")

    if not bloques_audio:
        raise ValueError("No se generó ningún bloque de audio válido para concatenar.")

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
    print("[1/3] Obteniendo noticias de fuentes RSS...")
    noticias = await obtener_titulares()
    if not noticias:
        noticias = (
            "FUENTE OFICIAL: Observatorio del Instituto para el Futuro de la Educación del Tec de Monterrey\n"
            "TÍTULO: Innovación en la Educación Superior Latinoamericana\n"
            "RESUMEN: Análisis sobre la adopción de herramientas de IA en universidades de México y Chile.\n---"
        )

    print("[2/3] Generando guion conversacional (Eduradar) con Gemini API...")
    guion = await generar_guion(noticias)
    guardar_guion_texto(guion)
    palabras = len(guion.split())
    minutos_estimados = round(palabras / 140, 1)
    print(f"   Guion redactado con éxito ({palabras} palabras, aprox. {minutos_estimados} minutos de audio).")

    print("[3/3] Sintetizando voces neurónicas a MP3...")
    await texto_a_audio_dos_voces(guion)
    print(f"¡Proceso completado! Archivos '{ARCHIVO_SALIDA}' y '{ARCHIVO_GUION}' generados correctamente.")

if __name__ == "__main__":
    try:
        asyncio.run(ejecutar_pipeline())
    except Exception as e:
        print(f"::error title=Fallo del Pipeline::{e}")
        sys.exit(1)
