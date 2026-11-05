import os
import sys
import html
import re
import json
import asyncio
from datetime import datetime
import feedparser
import edge_tts
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# ---------------------------------------------------------------------------
# 1. Configuración de Archivos y Fuentes
# ---------------------------------------------------------------------------

FECHA_STR = datetime.now().strftime("%d%m%Y")
ARCHIVO_SALIDA = f"episodio_ep{FECHA_STR}.mp3"
ARCHIVO_GUION = f"guion_ep{FECHA_STR}.txt"
ARCHIVO_HISTORICO = "historico_noticias.json"

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

MODELOS_TEXTO = [
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest"
]

# ---------------------------------------------------------------------------
# 2. Control de Historico Anti-Repetición
# ---------------------------------------------------------------------------

def cargar_historico() -> list[str]:
    if os.path.exists(ARCHIVO_HISTORICO):
        try:
            with open(ARCHIVO_HISTORICO, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def guardar_historico(nuevos_titulos: list[str]) -> None:
    historico = cargar_historico()
    historico.extend(nuevos_titulos)
    # Conservar únicamente los últimos 50 títulos para mantener el archivo ligero
    historico = historico[-50:]
    with open(ARCHIVO_HISTORICO, "w", encoding="utf-8") as f:
        json.dump(historico, f, ensure_ascii=False, indent=2)

def limpiar_texto(texto: str) -> str:
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = html.unescape(texto)
    return re.sub(r"\s+", " ", texto).strip()[:700]

def _parsear_feed(url: str, historico: list[str]) -> list[tuple[str, str]]:
    try:
        feed = feedparser.parse(url)
        nombre_fuente = feed.feed.get('title', url)
        entradas = getattr(feed, "entries", [])
        items = []
        for entry in entradas[:3]:
            titulo = limpiar_texto(getattr(entry, "title", "Sin título"))
            # Filtro anti-repetición: omitir si ya se usó recientemente
            if titulo in historico:
                continue
            resumen = limpiar_texto(getattr(entry, "summary", getattr(entry, "description", "Sin resumen")))
            link = getattr(entry, "link", "")
            bloque = (
                f"FUENTE OFICIAL: {nombre_fuente}\n"
                f"ENLACE: {link}\n"
                f"TÍTULO: {titulo}\n"
                f"RESUMEN: {resumen}\n---"
            )
            items.append((titulo, bloque))
        return items
    except Exception as e:
        print(f"[Error RSS] {url}: {e}")
        return []

async def obtener_titulares() -> tuple[str, list[str]]:
    historico = cargar_historico()
    tareas = [asyncio.to_thread(_parsear_feed, url, historico) for url in RSS_FEEDS]
    resultados = await asyncio.gather(*tareas)
    
    todos_los_items = [item for grupo in resultados for item in grupo]
    titulos_usados = [t[0] for t in todos_los_items[:12]]
    bloques_texto = [t[1] for t in todos_los_items[:12]]
    
    return "\n".join(bloques_texto), titulos_usados

# ---------------------------------------------------------------------------
# 3. Prompt Maestro Expresivo para "Edu-Radar"
# ---------------------------------------------------------------------------

def construir_prompt_maestro(texto_noticias: str) -> str:
    meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    ahora = datetime.now()
    fecha_hoy = f"{ahora.day} de {meses[ahora.month - 1]} de {ahora.year}"

    return f"""
Actúa como un equipo de dos periodistas y locutores profesionales de podcast sobre educación y EdTech:
- SOFÍA: Presentadora principal, analítica, apasionada por las políticas educativas e inclusión.
- ALEJANDRO: Co-presentador, dinámico, enfocado en tecnología de aprendizaje e impacto práctico.

Tu objetivo es redactar un guion en formato de DIÁLOGO SUMAMENTE EXPRESIVO, NATURAL Y ENTRETENIDO para el podcast "Edu-Radar".

TARGET DE EXTENSIÓN: Entre 750 y 950 PALABRAS (5 a 7 minutos).

REGLAS DE INTENSIÓN Y EMOCIÓN EN EL GUION (PARA SÍNTESIS DE VOZ):
1. PRONUNCIACIÓN DEL NOMBRE: Escribe SIEMPRE el nombre del podcast como `Edu-Radar` (separado con guión) para asegurar la correcta modulación fonética.
2. DENSIDAD EMOCIONAL Y RITMO:
   - Usa frases cortas y directas. Evita oraciones largas de más de 15 palabras.
   - Utiliza signos de interrogación (`¿...?`) para lanzar preguntas espontáneas entre los locutores.
   - Utiliza signos de admiración (`¡...!`) para dar énfasis y entusiasmo en noticias clave.
   - Usa puntos suspensivos (`...`) antes de revelar datos importantes para simular pausas de respiración y expectativa humana.
3. ESTRUCTURA:
   - APERTURA: Saludo caluroso, mención de "Edu-Radar" y la fecha exacta "{fecha_hoy}".
   - NOTICIAS: Citación explícita de la fuente oficial en cada nota.
   - CIERRE: Pregunta provocadora para la audiencia y llamada a la acción (CTA) para seguir el podcast.

RESTRICCIONES ESTRICTAS:
- Cada línea DEBE comenzar estrictamente con `SOFÍA:` o `ALEJANDRO:`.
- NO agregues acotaciones entre paréntesis o corchetes (nada de `[risas]`, `[pausa]`).
- NO uses formato Markdown (sin asteriscos, sin encabezados `#`, sin viñetas).
- Escribe ÚNICAMENTE el texto limpio para locución.

NOTICIAS FUENTE DISPONIBLES:
{texto_noticias}
"""

def _llamar_modelo(client: genai.Client, modelo: str, prompt: str) -> str:
    response = client.models.generate_content(
        model=modelo,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.7)
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
# 4. Síntesis Multimóvil Expresiva (Edge-TTS)
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

        # Reemplazo fonético preventivo
        texto_hablado = re.sub(r"\bEduradar\b", "Edu-Radar", texto_hablado, flags=re.IGNORECASE)
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
    print(f"[1/3] Obteniendo noticias RSS para el episodio {FECHA_STR} (Con filtro anti-repetición)...")
    noticias, titulos_usados = await obtener_titulares()

    print("[2/3] Generando guion conversacional expresivo con Gemini API...")
    guion = await generar_guion(noticias)
    guardar_guion_texto(guion)
    
    # Registrar las noticias elegidas en el histórico
    if titulos_usados:
        guardar_historico(titulos_usados)

    palabras = len(guion.split())
    minutos_estimados = round(palabras / 140, 1)
    print(f"   Guion redactado con éxito ({palabras} palabras, aprox. {minutos_estimados} minutos de audio).")

    print(f"[3/3] Sintetizando voces neurónicas a MP3 -> {ARCHIVO_SALIDA}...")
    await texto_a_audio_dos_voces(guion)
    print(f"¡Hito completado! Archivos '{ARCHIVO_SALIDA}' y '{ARCHIVO_GUION}' listos.")

if __name__ == "__main__":
    try:
        asyncio.run(ejecutar_pipeline())
    except Exception as e:
        print(f"::error title=Fallo del Pipeline::{e}")
        sys.exit(1)
