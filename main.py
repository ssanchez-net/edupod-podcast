import os
import sys
import html
import re
import json
import struct
import mimetypes
import asyncio
from datetime import datetime
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
import feedparser

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

# ---------------------------------------------------------------------------
# 1. Configuración Global
# ---------------------------------------------------------------------------

FECHA_STR = datetime.now().strftime("%d%m%Y")
ARCHIVO_VOZ_WAV = f"voz_raw_{FECHA_STR}.wav"
ARCHIVO_SALIDA_FINAL = f"episodio_ep{FECHA_STR}.mp3"
ARCHIVO_GUION = f"guion_ep{FECHA_STR}.txt"
ARCHIVO_HISTORICO = "historico_noticias.json"

ARCHIVO_INTRO = "intro.mp3"
ARCHIVO_OUTRO = "outro.mp3"
ARCHIVO_FONDO = "musica_fondo.mp3"

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
    "https://mfeldstein.com/feed/",
    "https://www.theguardian.com/education/rss",
    "https://rss.nytimes.com/services/xml/rss/nyt/Education.xml",
    "https://www.lemonde.fr/education/rss_full.xml",
    "https://elpais.com/rss/elpais/educacion.xml"
]

MODELO_TEXTO = "gemini-3.8-flash"
MODELO_TTS = "gemini-3.8-flash-tts"

# ---------------------------------------------------------------------------
# 2. Ingesta y Control de Histórico
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
    if not nuevos_titulos:
        return
    historico = cargar_historico()
    historico.extend(nuevos_titulos)
    historico = historico[-100:]
    with open(ARCHIVO_HISTORICO, "w", encoding="utf-8") as f:
        json.dump(historico, f, ensure_ascii=False, indent=2)

def limpiar_texto(texto: str) -> str:
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = html.unescape(texto)
    return re.sub(r"\s+", " ", texto).strip()[:700]

def _parsear_feed(url: str, historico: list[str]) -> list[tuple[str, str]]:
    try:
        feed = feedparser.parse(url)
        nombre_fuente = getattr(feed.feed, "title", url)
        entradas = getattr(feed, "entries", [])
        items = []
        for entry in entradas[:5]:
            titulo = limpiar_texto(getattr(entry, "title", "Sin título"))
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
    
    if not todos_los_items:
        print("   [Fallback RSS] No se detectaron noticias inéditas. Aplicando tema de respaldo EdTech...")
        bloque_fallback = (
            "FUENTE OFICIAL: Observatorio de Innovación Educativa\n"
            "TÍTULO: Avances en Inteligencia Artificial Generativa y Evaluación Formativa\n"
            "RESUMEN: Análisis global sobre cómo las instituciones de educación superior adoptan asistentes de IA "
            "para personalizar la retroalimentación estudiantil y optimizar la gestión docente.\n---"
        )
        return bloque_fallback, ["Avances en Inteligencia Artificial Generativa y Evaluación Formativa"]

    titulos_usados = [t[0] for t in todos_los_items[:12]]
    bloques_texto = [t[1] for t in todos_los_items[:12]]
    return "\n".join(bloques_texto), titulos_usados

# ---------------------------------------------------------------------------
# 3. Generación de Guion (Gemini API)
# ---------------------------------------------------------------------------

def construir_prompt_maestro(texto_noticias: str) -> str:
    meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    ahora = datetime.now()
    fecha_hoy = f"{ahora.day} de {meses[ahora.month - 1]} de {ahora.year}"

    return f"""
Actúa como un equipo de dos periodistas y locutores profesionales de podcast sobre educación y EdTech:
- SOFÍA (Speaker 1): Presentadora principal, analítica, apasionada por la pedagogía e inclusión. Tono: Friendly, amigable, cálido.
- ALEJANDRO (Speaker 2): Co-presentador, dinámico, enfocado en tecnología e impacto práctico. Tono: Friendly, cercano, empático.

Tu objetivo es redactar un guion COMPLETO en formato de DIÁLOGO CÁLIDO Y ENTERTAINING para el podcast "Edu Radar".

TARGET DE EXTENSIÓN: Entre 750 y 950 PALABRAS (5 a 7 minutos de locución).

REGLAS DE FORMATO Y EXPRESIVIDAD:
1. PRONUNCIACIÓN DEL NOMBRE: Escribe SIEMPRE el nombre del podcast separado con espacio como `Edu Radar` (SIN GUIÓN).
2. ETIQUETAS VOCALES: Integra de forma natural las siguientes marcas expresivas: `<breath>`, `<chuckle>`, `<enthusiastic>`, `<astonishment>`, `<sigh>`.
3. ESTRUCTURA OBLIGATORIA:
   - APERTURA: Saludo directo, mención de "Edu Radar" y fecha exacta "{fecha_hoy}".
   - DESARROLLO: Análisis fluido de las noticias con turnos secuenciales entre Sofía y Alejandro.
   - CIERRE: Pregunta provocadora, llamado a la acción (CTA) y despedida exacta: "¡Hasta la próxima emisión de Edu Radar!".

RESTRICCIONES ESTRICTAS:
- Cada intervención DEBE iniciar obligatoriamente con `SOFÍA:` o `ALEJANDRO:`.
- NO intercales roles en la misma línea.
- NO agregues acotaciones entre corchetes rectos `[...]`.
- NO uses formato Markdown (sin asteriscos, sin `#`, sin viñetas).

NOTICIAS FUENTE DISPONIBLES:
{texto_noticias}
"""

def _generar_texto_gemini(client: genai.Client, prompt: str) -> str:
    response = client.models.generate_content(
        model=MODELO_TEXTO,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.7)
    )
    if not response.text:
        raise ValueError("La respuesta del modelo de texto volvió vacía.")
    return response.text

async def generar_guion(client: genai.Client, texto_noticias: str) -> str:
    prompt = construir_prompt_maestro(texto_noticias)
    print("   [Gemini API] Solicitando generación de guion...")
    
    max_intentos = 3
    for intento in range(1, max_intentos + 1):
        try:
            guion_texto = await asyncio.to_thread(_generar_texto_gemini, client, prompt)
            print("   [Éxito] Guion generado correctamente.")
            return guion_texto
        except (genai_errors.APIError, Exception) as e:
            err_str = str(e)
            # Si es error 429 (cuota diaria agotada), abortar de inmediato para no gastar intentos
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                raise RuntimeError(f"Cuota diaria de Gemini API agotada (429). Abortando sin reintentos: {e}")
            
            is_server_busy = "503" in err_str or "UNAVAILABLE" in err_str
            if is_server_busy and intento < max_intentos:
                espera = intento * 20
                print(f"   [API Ocupada 503] Esperando {espera}s antes de reintentar (Intento {intento}/{max_intentos})...")
                await asyncio.sleep(espera)
            else:
                raise e

def guardar_guion_texto(guion_texto: str) -> None:
    with open(ARCHIVO_GUION, "w", encoding="utf-8") as f:
        f.write(guion_texto)
    print(f"   [OK] Guion guardado en '{ARCHIVO_GUION}'.")

# ---------------------------------------------------------------------------
# 4. Síntesis TTS y Máquina de Estados de Voces
# ---------------------------------------------------------------------------

def parse_audio_mime_type(mime_type: str) -> dict[str, int]:
    bits_per_sample = 16
    rate = 24000
    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate = int(param.split("=", 1)[1])
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size
    )
    return header + audio_data

def _sintetizar_audio_multivoz(client: genai.Client, guion_texto: str) -> tuple[bytes, str]:
    lineas = guion_texto.strip().split("\n")
    parts = []
    current_speaker = "Speaker 1"

    for linea in lineas:
        linea_limpia = linea.strip()
        if not linea_limpia:
            continue

        if re.match(r"^SOFÍ?A:\s*$", linea_limpia, flags=re.IGNORECASE):
            current_speaker = "Speaker 1"
            continue
        elif re.match(r"^ALEJANDRO:\s*$", linea_limpia, flags=re.IGNORECASE):
            current_speaker = "Speaker 2"
            continue
        elif linea_limpia.upper().startswith("SOFÍA:") or linea_limpia.upper().startswith("SOFIA:"):
            current_speaker = "Speaker 1"
            texto_hablado = re.sub(r"^SOFÍ?A:\s*", "", linea_limpia, flags=re.IGNORECASE)
        elif linea_limpia.upper().startswith("ALEJANDRO:"):
            current_speaker = "Speaker 2"
            texto_hablado = re.sub(r"^ALEJANDRO:\s*", "", linea_limpia, flags=re.IGNORECASE)
        else:
            texto_hablado = linea_limpia

        texto_hablado = re.sub(r"\bEdu-?Radar\b", "Edu Radar", texto_hablado, flags=re.IGNORECASE)
        texto_hablado = html.unescape(texto_hablado).strip()

        if texto_hablado:
            parts.append(
                types.Part(
                    text=texto_hablado,
                    speech_metadata=types.SpeechMetadata(speaker=current_speaker)
                )
            )

    contents = [types.Content(role="user", parts=parts)]

    generate_content_config = types.GenerateContentConfig(
        temperature=1.0,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(
                        speaker="Speaker 1",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Jori")
                        )
                    ),
                    types.SpeakerVoiceConfig(
                        speaker="Speaker 2",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Rami")
                        )
                    ),
                ]
            )
        )
    )

    audio_data = bytearray()
    mime_type = ""

    for chunk in client.models.generate_content_stream(
        model=MODELO_TTS,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.parts is None:
            continue
        if chunk.parts[0].inline_data and chunk.parts[0].inline_data.data:
            inline_data = chunk.parts[0].inline_data
            audio_data.extend(inline_data.data)
            mime_type = inline_data.mime_type

    if not audio_data:
        raise ValueError("No se recibieron datos de audio desde el modelo Gemini TTS.")

    data_buffer = bytes(audio_data)
    file_extension = mimetypes.guess_extension(mime_type)
    if file_extension is None or not mime_type.startswith("audio/wav"):
        data_buffer = convert_to_wav(bytes(audio_data), mime_type)

    return data_buffer, ARCHIVO_VOZ_WAV

async def generar_audio_expresivo(client: genai.Client, guion_texto: str) -> tuple[bytes, str]:
    print("   [Gemini TTS] Generando audio a dos voces (Sofía=Jori, Alejandro=Rami)...")
    max_intentos = 3
    for intento in range(1, max_intentos + 1):
        try:
            return await asyncio.to_thread(_sintetizar_audio_multivoz, client, guion_texto)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                raise RuntimeError(f"Cuota TTS agotada (429): {e}")
            if intento < max_intentos:
                espera = intento * 20
                print(f"   [Reintento TTS] Error temporal ({e}). Reintentando en {espera}s...")
                await asyncio.sleep(espera)
            else:
                raise e

# ---------------------------------------------------------------------------
# 5. Mezcla de Audio Profesional
# ---------------------------------------------------------------------------

def mezclar_audio_final(archivo_voz: str, archivo_salida: str) -> None:
    if AudioSegment is None:
        print("   [Aviso] pydub no disponible. Conservando audio sin cortinillas.")
        os.rename(archivo_voz, archivo_salida)
        return

    print("   [Audio Post-Procesamiento] Ensamblando cortinillas y música de fondo...")
    voz_segment = AudioSegment.from_file(archivo_voz)
    DURACION_FADE_MS = 3000

    if os.path.exists(ARCHIVO_FONDO):
        fondo = AudioSegment.from_file(ARCHIVO_FONDO) - 12
        if len(fondo) < len(voz_segment):
            repetir = (len(voz_segment) // len(fondo)) + 1
            fondo = fondo * repetir
        fondo = fondo[:len(voz_segment)].fade_in(DURACION_FADE_MS).fade_out(DURACION_FADE_MS)
        voz_segment = voz_segment.overlay(fondo)

    audio_final = AudioSegment.empty()
    if os.path.exists(ARCHIVO_INTRO):
        intro = AudioSegment.from_file(ARCHIVO_INTRO).fade_out(DURACION_FADE_MS)
        audio_final += intro

    audio_final += voz_segment

    if os.path.exists(ARCHIVO_OUTRO):
        outro = AudioSegment.from_file(ARCHIVO_OUTRO).fade_in(DURACION_FADE_MS)
        audio_final += outro

    audio_final.export(archivo_salida, format="mp3", bitrate="192k")
    if os.path.exists(archivo_voz):
        os.remove(archivo_voz)
    print(f"   [OK] Episodio masterizado guardado en '{archivo_salida}'.")

# ---------------------------------------------------------------------------
# 6. Punto de Entrada Principal
# ---------------------------------------------------------------------------

async def ejecutar_pipeline() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("CRÍTICO: GEMINI_API_KEY no está configurada en GitHub Secrets.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=300_000)
    )

    print(f"[1/3] Obteniendo noticias RSS para el episodio {FECHA_STR}...")
    noticias, titulos_usados = await obtener_titulares()

    print("[2/3] Generando guion expresivo con Gemini API...")
    guion = await generar_guion(client, noticias)
    guardar_guion_texto(guion)
    
    if titulos_usados:
        guardar_historico(titulos_usados)

    palabras = len(guion.split())
    print(f"   Guion redactado ({palabras} palabras).")

    print("[3/3] Sintetizando audio expresivo (Gemini 3.8 Flash TTS)...")
    data_buffer, nombre_archivo_raw = await generar_audio_expresivo(client, guion)
    with open(nombre_archivo_raw, "wb") as f:
        f.write(data_buffer)

    mezclar_audio_final(nombre_archivo_raw, ARCHIVO_SALIDA_FINAL)
    print(f"¡Proceso completado! Archivos '{ARCHIVO_SALIDA_FINAL}' y '{ARCHIVO_GUION}' generados.")

if __name__ == "__main__":
    try:
        asyncio.run(ejecutar_pipeline())
    except Exception as e:
        print(f"::error title=Fallo del Pipeline::{e}")
        sys.exit(1)
