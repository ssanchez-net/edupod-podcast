import os
import asyncio
import feedparser
import edge_tts
from google import genai

# Lista de feeds RSS verificados para la extracción de noticias
RSS_FEEDS = [
    "https://observatorio.tec.mx/feed/",
    "https://www.edsurge.com/articles_rss",
    "https://intef.es/feed/"
]

def obtener_titulares():
    """Extrae las noticias más recientes de los feeds RSS configurados."""
    noticias_acumuladas = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            if hasattr(feed, 'entries') and feed.entries:
                for entry in feed.entries[:2]:
                    titulo = getattr(entry, 'title', 'Sin título')
                    resumen = getattr(entry, 'summary', getattr(entry, 'description', 'Sin resumen'))
                    noticias_acumuladas.append(f"Título: {titulo}\nResumen: {resumen}\n---")
            else:
                print(f"[Advertencia] No se encontraron entradas en el feed: {url}")
        except Exception as e:
            print(f"[Error RSS] Error al leer la fuente {url}: {e}")
            
    return "\n".join(noticias_acumuladas)

async def generar_guion(texto_noticias):
    """Envía las noticias extraídas a Gemini API para redactar el guion."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("CRÍTICO: La variable GEMINI_API_KEY no está configurada o está vacía en los secretos del repositorio.")
        
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    Actúa como un locutor profesional de podcast de noticias sobre educación y tecnología educativa.
    Con la información de las siguientes noticias, redacta un guion dinámico, claro y coherente a una sola voz.
    
    RESTRICCIONES ESTRICTAS:
    1. La lectura del guion debe durar entre 3 y 4 minutos (aproximadamente entre 450 y 550 palabras).
    2. NO incluyas acotaciones de sonido, marcadores de tono, ni indicaciones entre paréntesis o corchetes.
    3. Escribe ÚNICAMENTE el texto que será leído textualmente por el sintetizador de voz.
    4. Estructura el contenido con una breve bienvenida, el desarrollo de las noticias principales y una despedida rápida.

    Noticias fuente:
    {texto_noticias}
    """
    
    try:
        # Uso del modelo flash estándar respaldado por la librería
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        if not response.text:
            raise ValueError("La respuesta de Gemini volvió vacía.")
        return response.text
    except Exception as e:
        print(f"[Error Gemini API] Fallo al generar guion con el modelo predeterminado: {e}")
        # Intento de respaldo con modelo 1.5-flash
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt,
        )
        return response.text

async def texto_a_audio(texto_guion, archivo_salida="episodio_hoy.mp3"):
    """Convierte el texto del guion en un archivo de audio MP3 usando Edge-TTS."""
    voz = "es-MX-DaliaNeural"
    comunicador = edge_tts.Communicate(texto_guion, voz)
    await comunicador.save(archivo_salida)

async def ejecutar_pipeline():
    print("[1/3] Extrayendo noticias de fuentes RSS...")
    noticias = obtener_titulares()
    
    if not noticias:
        print("[Alerta] No se pudieron extraer noticias RSS, utilizando noticia de respaldo...")
        noticias = "Título: Avances en Inteligencia Artificial Aplicada a la Educación en 2026.\nResumen: Las herramientas de IA continúan transformando la personalización del aprendizaje en el aula global."
        
    print("[2/3] Generando guion con Gemini API...")
    guion = await generar_guion(noticias)
    print(f"Guion generado con éxito ({len(guion.split())} palabras).")
    
    print("[3/3] Sintetizando voz y generando archivo MP3...")
    await texto_a_audio(guion)
    print("¡Proceso completado! El archivo 'episodio_hoy.mp3' ha sido generado exitosamente.")

if __name__ == "__main__":
    asyncio.run(ejecutar_pipeline())
