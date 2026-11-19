import os
import sys
import html
import re
import json
import struct
import mimetypes
import asyncio
import email.utils
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
import feedparser

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

# ---------------------------------------------------------------------------
# 1. Configuración Global y Metadatos del Programa
# ---------------------------------------------------------------------------

BASE_URL = "https://ssanchez-net.github.io/edupod-podcast"
PODCAST_TITULO = "Edu Radar"
PODCAST_DESCRIPCION = (
    "Tu radar semanal sobre el futuro de la educación. Cada martes y viernes, Sofía y Alejandro "
    "analizan las noticias más relevantes en tecnología educativa (EdTech), inteligencia artificial "
    "aplicada al aula, pedagogía digital e innovación académica en instituciones de todo el mundo. "
    "Un espacio ágil y directo para docentes, directivos y apasionados por transformar el aprendizaje."
)
PODCAST_AUTOR = "Edumedios"
PODCAST_EMAIL = "ssanchez.net@gmail.com"
PODCAST_IDIOMA = "es-mx"
PODCAST_CATEGORIA = "Education"
PODCAST_SUBCATEGORIA = "Technology"
URL_PORTADA = f"{BASE_URL}/portada.jpg"

FECHA_STR = datetime.now().strftime("%d%m%Y")
ARCHIVO_VOZ_WAV = f"voz_raw_{FECHA_STR}.wav"
ARCHIVO_SALIDA_FINAL = f"episodio_ep{FECHA_STR}.mp3"
ARCHIVO_GUION = f"guion_ep{FECHA_STR}.txt"
ARCHIVO_HISTORICO = "historico_noticias.json"
ARCHIVO_EPISODIOS_DB = "episodios.json"
ARCHIVO_FEED_XML = "feed.xml"
ARCHIVO_INDEX_HTML = "index.html"

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

MODELO_TEXTO_PRIMARIO = "gemini-3.8-flash"
MODELO_TEXTO_RESPALDO = "gemini-3.5-flash-lite"
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
    historico = historico[-150:]
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

    titulos_usados = [t[0] for t in todos_los_items[:7]]
    bloques_texto = [t[1] for t in todos_los_items[:7]]
    return "\n".join(bloques_texto), titulos_usados

# ---------------------------------------------------------------------------
# 3. Generación Estructurada de Metadatos y Guion (Gemini API)
# ---------------------------------------------------------------------------

def construir_prompt_maestro(texto_noticias: str, numero_episodio: int) -> str:
    meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    ahora = datetime.now()
    fecha_hoy = f"{ahora.day} de {meses[ahora.month - 1]} de {ahora.year}"

    return f"""
Actúa como un equipo de dos periodistas y locutores profesionales de podcast sobre educación y EdTech:
- SOFÍA (Speaker 1): Presentadora principal, analítica, apasionada por la pedagogía e inclusión. Tono: Friendly, amigable, cálido.
- ALEJANDRO (Speaker 2): Co-presentador, dinámico, enfocado en tecnología e impacto práctico. Tono: Friendly, cercano, empático.

Debes devolver ÚNICAMENTE un objeto JSON válido (sin bloques de código markdown ```json) con exactamente tres claves: "titulo", "descripcion" y "guion".

ESPECIFICACIONES DE CADA CLAVE DEL JSON:
1. "titulo": Título atractivo y SEO para el episodio número {numero_episodio} (máximo 140 caracteres), iniciando exactamente con "Ep. {numero_episodio}: ".
2. "descripcion": Descripción enriquecida para Spotify y Apple Podcasts (entre 400 y 900 caracteres) que resuma los temas principales, incluya viñetas con "• " y cierre con la pregunta de debate del episodio y un llamado a suscribirse.
3. "guion": Guion COMPLETO en formato de DIÁLOGO CÁLIDO Y ENTERTAINING para el podcast "Edu Radar" con estas reglas estrictas:
   - TARGET DE EXTENSIÓN DEL GUION: Entre 750 y 950 PALABRAS (5 a 7 minutos de locución).
   - PRONUNCIACIÓN DEL NOMBRE: Escribe SIEMPRE el nombre del podcast separado con espacio como `Edu Radar` (SIN GUIÓN).
   - ETIQUETAS VOCALES: Integra de forma natural las marcas expresivas: `<breath>`, `<chuckle>`, `<enthusiastic>`, `<astonishment>`, `<sigh>`.
   - ESTRUCTURA OBLIGATORIA DEL GUION:
     * APERTURA: Saludo directo, mención de "Edu Radar" y fecha exacta "{fecha_hoy}".
     * DESARROLLO: Análisis fluido de las noticias con turnos secuenciales entre Sofía y Alejandro.
     * CIERRE: Pregunta provocadora, llamado a la acción (CTA) y despedida exacta: "¡Hasta la próxima emisión de Edu Radar!".
   - RESTRICCIONES DEL GUION: Cada intervención DEBE iniciar con `SOFÍA:` o `ALEJANDRO:` separado por saltos de línea (`\\n\\n`). NO intercales roles en la misma línea. NO uses acotaciones `[...]` ni formato Markdown (`*`, `#`).

NOTICIAS FUENTE DISPONIBLES:
{texto_noticias}
"""

def _generar_json_gemini(client: genai.Client, prompt: str, modelo: str) -> dict:
    response = client.models.generate_content(
        model=modelo,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            response_mime_type="application/json"
        )
    )
    if not response.text:
        raise ValueError(f"La respuesta del modelo {modelo} volvió vacía.")
    texto_limpio = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.text.strip())
    datos = json.loads(texto_limpio)
    if not all(k in datos and datos[k] for k in ("titulo", "descripcion", "guion")):
        raise ValueError("El JSON devuelto no contiene las claves requeridas (titulo, descripcion, guion).")
    return datos

async def generar_contenido_episodio(client: genai.Client, texto_noticias: str, numero_episodio: int) -> dict:
    prompt = construir_prompt_maestro(texto_noticias, numero_episodio)
    modelos_cascada = [MODELO_TEXTO_PRIMARIO, MODELO_TEXTO_RESPALDO]

    for modelo_actual in modelos_cascada:
        print(f"   [Gemini API] Solicitando metadatos y guion con {modelo_actual}...")
        for intento in range(1, 3):
            try:
                datos = await asyncio.to_thread(_generar_json_gemini, client, prompt, modelo_actual)
                print(f"   [Éxito] Contenido estructurado generado usando {modelo_actual}.")
                return datos
            except (genai_errors.APIError, Exception) as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    print(f"   [Cuota Agotada 429] {modelo_actual} sin cuota. Probando respaldo si existe...")
                    break
                if ("503" in err_str or "504" in err_str or "UNAVAILABLE" in err_str) and intento < 2:
                    tiempo_espera = 45
                    print(f"   [API Ocupada] {modelo_actual} saturado. Reintentando en {tiempo_espera}s...")
                    await asyncio.sleep(tiempo_espera)
                else:
                    print(f"   [Fallo en {modelo_actual}] {e}. Conmutando a modelo de respaldo...")
                    break

    raise RuntimeError("CRÍTICO: Todos los modelos de texto en cascada fallaron.")

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
        b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1,
        num_channels, sample_rate, byte_rate, block_align,
        bits_per_sample, b"data", data_size
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

def mezclar_audio_final(archivo_voz: str, archivo_salida: str) -> int:
    if AudioSegment is None:
        os.rename(archivo_voz, archivo_salida)
        return 360

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

    duracion_segundos = int(len(audio_final) / 1000)
    audio_final.export(archivo_salida, format="mp3", bitrate="192k")
    if os.path.exists(archivo_voz):
        os.remove(archivo_voz)
    print(f"   [OK] Episodio masterizado guardado en '{archivo_salida}' ({duracion_segundos}s).")
    return duracion_segundos

# ---------------------------------------------------------------------------
# 6. Gestión de Base de Episodios y Generación de Feed RSS + Web
# ---------------------------------------------------------------------------

def cargar_o_inicializar_episodios() -> list[dict]:
    if os.path.exists(ARCHIVO_EPISODIOS_DB):
        try:
            with open(ARCHIVO_EPISODIOS_DB, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    episodio_1 = {
        "numero": 1,
        "temporada": 1,
        "titulo": "Ep. 1: El fin del ensayo tradicional, arte contemporáneo y el futuro del talento internacional",
        "descripcion": (
            "En este episodio de Edu Radar, Sofía y Alejandro analizan cómo la inteligencia artificial generativa "
            "está obligando a transformar las tareas escolares y el fin del ensayo tradicional para llevar a casa. "
            "Además, reflexionamos sobre el pensamiento crítico a través del arte contemporáneo, los retos de la "
            "educación cívica y la libertad de expresión en las aulas, y la iniciativa clave en Filadelfia para "
            "retener el talento universitario internacional.\n\n"
            "Temas clave de la emisión:\n"
            "• La evolución de la pedagogía frente a la IA generativa.\n"
            "• El arte y la provocación como herramientas de pensamiento crítico.\n"
            "• Educación cívica, democracia y libertad de cátedra.\n"
            "• Integración y retención de talento internacional en la educación superior.\n\n"
            "Frente a un mundo saturado de respuestas automatizadas, ¿qué práctica estás implementando hoy para "
            "estimular el pensamiento propio de tus estudiantes? ¡Suscríbete y compártelo!"
        ),
        "archivo_mp3": "episodio_ep13112026.mp3",
        "bytes": os.path.getsize("episodio_ep13112026.mp3") if os.path.exists("episodio_ep13112026.mp3") else 8890000,
        "duracion_seg": 360,
        "pub_date": "Fri, 13 Nov 2026 06:15:00 +0000",
        "guid": "eduradar-ep-1-13112026"
    }
    return [episodio_1]

def generar_rss_y_web(episodios: list[dict]) -> None:
    ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"
    ATOM_NS = "http://www.w3.org/2005/Atom"
    ET.register_namespace("itunes", ITUNES_NS)
    ET.register_namespace("atom", ATOM_NS)

    rss = ET.Element("rss", {"version": "2.0"})
    channel = ET.SubElement(rss, "channel")

    ET.SubElement(channel, f"{{{ATOM_NS}}}link", {
        "href": f"{BASE_URL}/{ARCHIVO_FEED_XML}",
        "rel": "self",
        "type": "application/rss+xml"
    })
    ET.SubElement(channel, "title").text = PODCAST_TITULO
    ET.SubElement(channel, "link").text = BASE_URL
    ET.SubElement(channel, "language").text = PODCAST_IDIOMA
    ET.SubElement(channel, "description").text = PODCAST_DESCRIPCION
    ET.SubElement(channel, f"{{{ITUNES_NS}}}author").text = PODCAST_AUTOR
    ET.SubElement(channel, f"{{{ITUNES_NS}}}summary").text = PODCAST_DESCRIPCION
    ET.SubElement(channel, f"{{{ITUNES_NS}}}explicit").text = "false"
    ET.SubElement(channel, f"{{{ITUNES_NS}}}type").text = "episodic"

    owner = ET.SubElement(channel, f"{{{ITUNES_NS}}}owner")
    ET.SubElement(owner, f"{{{ITUNES_NS}}}name").text = PODCAST_AUTOR
    ET.SubElement(owner, f"{{{ITUNES_NS}}}email").text = PODCAST_EMAIL

    ET.SubElement(channel, f"{{{ITUNES_NS}}}image", {"href": URL_PORTADA})
    img_std = ET.SubElement(channel, "image")
    ET.SubElement(img_std, "url").text = URL_PORTADA
    ET.SubElement(img_std, "title").text = PODCAST_TITULO
    ET.SubElement(img_std, "link").text = BASE_URL

    cat = ET.SubElement(channel, f"{{{ITUNES_NS}}}category", {"text": PODCAST_CATEGORIA})
    ET.SubElement(cat, f"{{{ITUNES_NS}}}category", {"text": PODCAST_SUBCATEGORIA})

    for ep in reversed(episodios):
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = ep["titulo"]
        ET.SubElement(item, "description").text = ep["descripcion"]
        ET.SubElement(item, f"{{{ITUNES_NS}}}summary").text = ep["descripcion"]
        ET.SubElement(item, "pubDate").text = ep["pub_date"]
        ET.SubElement(item, "guid", {"isPermaLink": "false"}).text = ep["guid"]
        ET.SubElement(item, "enclosure", {
            "url": f"{BASE_URL}/{ep['archivo_mp3']}",
            "length": str(ep["bytes"]),
            "type": "audio/mpeg"
        })
        ET.SubElement(item, f"{{{ITUNES_NS}}}duration").text = str(ep["duracion_seg"])
        ET.SubElement(item, f"{{{ITUNES_NS}}}explicit").text = "false"
        ET.SubElement(item, f"{{{ITUNES_NS}}}episodeType").text = "full"
        ET.SubElement(item, f"{{{ITUNES_NS}}}season").text = str(ep.get("temporada", 1))
        ET.SubElement(item, f"{{{ITUNES_NS}}}episode").text = str(ep["numero"])
        ET.SubElement(item, f"{{{ITUNES_NS}}}image", {"href": URL_PORTADA})

    tree = ET.ElementTree(rss)
    ET.indent(tree, space="  ")
    tree.write(ARCHIVO_FEED_XML, encoding="utf-8", xml_declaration=True)
    print(f"   [OK] Canal RSS actualizado en '{ARCHIVO_FEED_XML}'.")

    items_html = ""
    for ep in reversed(episodios):
        items_html += f"""
        <article style="background:#111827;padding:20px;border-radius:12px;margin-bottom:16px;border:1px solid #1f2937;">
            <h2 style="margin:0 0 8px 0;color:#a3e635;font-size:1.2rem;">{html.escape(ep['titulo'])}</h2>
            <p style="color:#9ca3af;font-size:0.85rem;margin:0 0 12px 0;">{ep['pub_date']}</p>
            <p style="color:#e5e7eb;line-height:1.5;white-space:pre-line;">{html.escape(ep['descripcion'])}</p>
            <audio controls style="width:100%;margin-top:12px;" src="{ep['archivo_mp3']}"></audio>
        </article>
        """

    pagina_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{PODCAST_TITULO} - By {PODCAST_AUTOR}</title>
</head>
<body style="background:#030712;color:#f9fafb;font-family:system-ui,-apple-system,sans-serif;max-width:760px;margin:0 auto;padding:32px 16px;">
    <header style="display:flex;align-items:center;gap:20px;margin-bottom:32px;">
        <img src="portada.jpg" alt="{PODCAST_TITULO}" style="width:120px;height:120px;border-radius:16px;">
        <div>
            <h1 style="margin:0;font-size:2rem;">{PODCAST_TITULO}</h1>
            <p style="color:#a3e635;margin:4px 0 8px 0;font-weight:600;">By {PODCAST_AUTOR}</p>
            <a href="{ARCHIVO_FEED_XML}" style="display:inline-block;background:#2563eb;color:#fff;padding:6px 14px;border-radius:8px;text-decoration:none;font-size:0.9rem;">Suscribirse vía RSS</a>
        </div>
    </header>
    <main>{items_html}</main>
</body>
</html>"""
    with open(ARCHIVO_INDEX_HTML, "w", encoding="utf-8") as f:
        f.write(pagina_html)
    print(f"   [OK] Sitio web actualizado en '{ARCHIVO_INDEX_HTML}'.")

# ---------------------------------------------------------------------------
# 7. Punto de Entrada Principal
# ---------------------------------------------------------------------------

async def ejecutar_pipeline() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("CRÍTICO: GEMINI_API_KEY no está configurada en GitHub Secrets.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=300_000)
    )

    episodios = cargar_o_inicializar_episodios()
    siguiente_num = max(ep["numero"] for ep in episodios) + 1

    print(f"[1/4] Obteniendo noticias RSS para el Episodio {siguiente_num} ({FECHA_STR})...")
    noticias, titulos_usados = await obtener_titulares()

    print("[2/4] Generando metadatos SEO y guion expresivo con Gemini API...")
    contenido = await generar_contenido_episodio(client, noticias, siguiente_num)
    guion = contenido["guion"]
    guardar_guion_texto(guion)

    if titulos_usados:
        guardar_historico(titulos_usados)

    palabras = len(guion.split())
    print(f"   Título: {contenido['titulo']}")
    print(f"   Guion redactado ({palabras} palabras).")

    print("[3/4] Sintetizando audio expresivo (Gemini 3.8 Flash TTS)...")
    data_buffer, nombre_archivo_raw = await generar_audio_expresivo(client, guion)
    with open(nombre_archivo_raw, "wb") as f:
        f.write(data_buffer)

    duracion_seg = mezclar_audio_final(nombre_archivo_raw, ARCHIVO_SALIDA_FINAL)

    print("[4/4] Actualizando repositorio y canal RSS (feed.xml)...")
    nuevo_episodio = {
        "numero": siguiente_num,
        "temporada": 1,
        "titulo": contenido["titulo"],
        "descripcion": contenido["descripcion"],
        "archivo_mp3": ARCHIVO_SALIDA_FINAL,
        "bytes": os.path.getsize(ARCHIVO_SALIDA_FINAL),
        "duracion_seg": duracion_seg,
        "pub_date": email.utils.format_datetime(datetime.now(timezone.utc)),
        "guid": f"eduradar-ep-{siguiente_num}-{FECHA_STR}"
    }
    episodios.append(nuevo_episodio)

    with open(ARCHIVO_EPISODIOS_DB, "w", encoding="utf-8") as f:
        json.dump(episodios, f, ensure_ascii=False, indent=2)

    generar_rss_y_web(episodios)
    print(f"¡Proceso completado! Episodio {siguiente_num} publicado en {BASE_URL}/{ARCHIVO_FEED_XML}")

if __name__ == "__main__":
    try:
        asyncio.run(ejecutar_pipeline())
    except Exception as e:
        print(f"::error title=Fallo del Pipeline::{e}")
        sys.exit(1)
