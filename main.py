import os
import asyncio
import feedparser
import edge_tts
from google import genai

RSS_FEEDS = [
    "https://observatorio.tec.mx/feed/",
    "https://www.edsurge.com/articles_rss",
    "https://intef.es/feed/"
]

def obtener_titulares():
    noticias_acumuladas = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            if hasattr(feed, 'entries') and feed.entries:
                for entry in feed.entries[:2]:
                    titulo = getattr(entry, 'title', 'Sin título')
                    resumen = getattr(entry, 'summary', getattr(entry, 'description', 'Sin resumen'))
                    noticias_acumuladas.append(f"Título: {titulo}\nResumen: {resumen}\n---")
        except Exception as e:
            print(f"Error al leer {url}: {e}")
            
    return "\n".join(noticias_acumuladas)

async def generar_guion(texto_noticias):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY no configurada.")
        
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
    
    # Modelo compatible con el SDK google-genai
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    return response.text

async def texto_a_audio(texto_guion, archivo_salida="episodio_hoy.mp3"):
    voz = "es-MX-DaliaNeural"
    comunicador = edge_tts.Communicate(texto_guion, voz)
    await comunicador.save(archivo_salida)

async def ejecutar_pipeline():
    print("[1/3] Extrayendo noticias...")
    noticias = obtener_titulares()
    
    if not noticias:
        noticias = "Título: Innovación educativa.\nResumen: Avances en aprendizaje digital."
        
    print("[2/3] Generando guion...")
    guion = await generar_guion(noticias)
    
    print("[3/3] Generando audio MP3...")
    await texto_a_audio(guion)
    print("¡Proceso completado exitosamente!")

if __name__ == "__main__":
    asyncio.run(ejecutar_pipeline())
