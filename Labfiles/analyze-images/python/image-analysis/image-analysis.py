from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import sys
from matplotlib import pyplot as plt
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

def main():
    # Limpiar consola
    os.system('cls' if os.name=='nt' else 'clear')

    try:
        # Cargar configuración
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # Validar que las credenciales existan
        if not ai_endpoint or not ai_key:
            print("Error: Faltan las variables de entorno AI_SERVICE_ENDPOINT o AI_SERVICE_KEY.")
            return

        # Obtener imagen
        image_file = 'images/street.jpg'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        # Autenticar cliente
        client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key)
        )

        # Leer imagen
        with open(image_file, "rb") as f:
            image_data = f.read()

        print(f'Analizando {image_file}...')

        # Analizar imagen
        # --- MODIFICACIÓN ---
        # Eliminamos CAPTION y DENSE_CAPTIONS porque la suscripción
        # "Azure for Students" no los soporta en las regiones permitidas.
        result = client.analyze(
            image_data=image_data,
            visual_features=[
                # VisualFeatures.CAPTION,      <--- ELIMINADO
                VisualFeatures.TAGS,
                # VisualFeatures.DENSE_CAPTIONS, <--- ELIMINADO
                VisualFeatures.OBJECTS,
                VisualFeatures.PEOPLE
            ],
        )

        # 1. Mostrar Caption principal (Eliminado)
        # if result.caption is not None: ...

        # 2. Mostrar Dense Captions (Eliminado)
        # if result.dense_captions is not None: ...


        # 3. Mostrar Tags
        if result.tags is not None:
            print("\n--- Tags ---")
            for tag in result.tags.list:
                print("'{}' (confianza: {:.2f}%)".format(tag.name, tag.confidence * 100))

        # 4. Procesar Objetos (Llamada a la función)
        if result.objects is not None:
            print("\n--- Objetos Detectados ---")
            for detected_object in result.objects.list:
                print("'{}' (confianza: {:.2f}%)".format(detected_object.tags[0].name, detected_object.tags[0].confidence * 100))
            
            # Llamamos a tu función para generar la imagen
            show_objects(image_file, result.objects.list)

        # 5. Procesar Personas (Llamada a la función)
        if result.people is not None:
            print("\n--- Personas Detectadas ---")
            for detected_person in result.people.list:
                print("Persona (confianza: {:.2f}%)".format(detected_person.confidence * 100))
            
            # Llamamos a tu función para generar la imagen
            show_people(image_file, result.people.list)

    except Exception as ex:
        print(f"Ocurrió un error: {ex}")

def show_objects(image_filename, detected_objects):
    print("\nGuardando imagen de objetos...")
    image = Image.open(image_filename)
    fig = plt.figure(figsize=(image.width/100, image.height/100))
    plt.axis('off')
    draw = ImageDraw.Draw(image)
    color = 'cyan'

    for detected_object in detected_objects:
        r = detected_object.bounding_box
        bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height)) 
        draw.rectangle(bounding_box, outline=color, width=3)
        plt.annotate(detected_object.tags[0].name, (r.x, r.y), backgroundcolor=color)

    plt.imshow(image)
    plt.tight_layout(pad=0)
    objectfile = 'objects(exp1).jpg'
    fig.savefig(objectfile)
    print('  Resultados guardados en', objectfile)

def show_people(image_filename, detected_people):
    print("\nGuardando imagen de personas...")
    image = Image.open(image_filename)
    fig = plt.figure(figsize=(image.width/100, image.height/100))
    plt.axis('off')
    draw = ImageDraw.Draw(image)
    color = 'cyan'

    for detected_person in detected_people:
        if detected_person.confidence > 0.5: # Ajusté un poco la confianza
            r = detected_person.bounding_box
            bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
            draw.rectangle(bounding_box, outline=color, width=3)

    plt.imshow(image)
    plt.tight_layout(pad=0)
    peoplefile = 'people(exp1).jpg'
    fig.savefig(peoplefile)
    print('  Resultados guardados en', peoplefile)

if __name__ == "__main__":
    main()