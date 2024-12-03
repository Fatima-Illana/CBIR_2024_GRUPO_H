# Sistema CBIR para Imágenes Artísticas

Este proyecto implementa un sistema de Recuperación de Imágenes Basado en Contenido (CBIR) orientado a obras de arte. Permite buscar y recuperar imágenes similares usando técnicas de extracción de características y aprendizaje automático.

## Aspectos a tener en cuenta para la ejecución
- Para ejecutarlo basta con ejecutar el archivo Interfaz.py que contiene el código de la interfaz.
- Comando de ejecución: py -m streamlit run interfaz.py
- Verisón keras utilizada: 3.7.0
- Imágenes de prueba para ingresar en la interfaz están disponibles en la carpeta Test del DatasetArteTrainTest.zip
- Los ficheros de extractores contiene el proceso seguido para la generación de los índices.
- El fichero Bag_Of_Words.npy contiene el diccionario de palabras visuales utilizado en el extractor 3 (Bag_Of_Words).
- El fichero autoencoder.keras contiene el modelo generado para el extractor 5 (Autoencoder).
- El cuaderno de preprocesado contiene el proceso de división del dataset y la creación de la base de datos.
- El fichero resultados.ipynb contiene los cálculos realizados para extraer los resultados.
- El fichero ResultadosCBIR.xlxs contiene los resultados obtenidos de todas las imágenes de test.
- La carpeta Dataset Arte contiene todas las imágenes utilizadas, separadas por movimientos.
- La carpeta DatasetArteTrainTest contiene las imágenes ya divididas en Train y Test y con el nombre cambiado. 

## Dataset
- **Fuente**: Dataset reducido de [WikiArt Art Movements/Styles](https://www.kaggle.com/datasets/sivarazadi/wikiart-art-movementsstyles).
