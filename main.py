import os
import asyncio
import feedparser
import edge_tts
import google.generativeai as genai

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
            # Extraer las 2 publicaciones más recientes de cada fuente
            for entry in feed.entries[:2]:
                titulo = entry.get('title', 'Sin título')
                resumen = entry.get('summary', 'Sin resumen disponible')
                noticias_acumuladas.append(f"Título: {titulo}\nResumen: {resumen}\n---")
        except Exception as e:
            print(f"Error al leer la fuente {url}: {e}")
            
    return "\n".join(noticias_acumuladas)

async def generar_guion(texto_noticias):
    """Envía las noticias extraídas a Gemini API para redactar el guion."""
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Actúa como un locutor profesional de podcast de noticias sobre educación y tecnología educativa.
    Con la información de las siguientes noticias, redacta un guion dinámico, claro y coherente a una sola voz.
    
    RESTRICCIONES ESTRICTAS:
    1. La lectura del guion debe durar entre 3 y 4 minutos (aproximadamente entre 450 y 550 palabras).
    2. NO incluyas acotaciones de sonido, marcadores de tono, ni indicaciones entre paréntesis o corchetes.
    3. Escribe ÚNICAMENTE el texto que será leído textualmente por el sintetizador de voz.
    4. Estructura el contenido con una breve bienvenida, el desarrollo de las 3 notas más importantes y una despedida rápida.

    Noticias fuente:
    {texto_noticias}
    """
    
    response = model.generate_content(prompt)
    return response.text

async def texto_a_audio(texto_guion, archivo_salida="episodio_hoy.mp3"):
    """Convierte el texto del guion en un archivo de audio MP3 usando Edge-TTS."""
    # Voz neuronal en español de México (Dalia)
    voz = "es-MX-DaliaNeural"
    comunicador = edge_tts.Communicate(texto_guion, voz)
    await comunicador.save(archivo_salida)

async def ejecutar_pipeline():
    print("[1/3] Extrayendo noticias de fuentes RSS...")
    noticias = obtener_titulares()
    
    if not noticias:
        raise ValueError("No se pudieron extraer noticias de las fuentes RSS.")
        
    print("[2/3] Generando guion con Gemini 1.5 Flash...")
    guion = await generar_guion(noticias)
    
    print("[3/3] Sintonizando voz y convirtiendo a audio MP3...")
    await texto_a_audio(guion)
    print("¡Proceso completado! El archivo 'episodio_hoy.mp3' ha sido generado exitosamente.")

if __name__ == "__main__":
    asyncio.run(ejecutar_pipeline())
