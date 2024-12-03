# Sistema CBIR para Imágenes Artísticas

Este proyecto implementa un sistema de Recuperación de Imágenes Basado en Contenido (CBIR) orientado a obras de arte. Permite buscar y recuperar imágenes similares usando técnicas de extracción de características y aprendizaje automático.

## Características
- **Métodos de extracción de características**:
  - **Histograma de Color**: Detecta similitudes basadas en la paleta cromática.
  - **Histograma de Textura**: Identifica patrones texturales.
  - **Bag of Words**: Encuentra elementos visuales recurrentes.
  - **CNN (VGG19)**: Extrae características profundas con redes convolucionales.
  - **Autoencoder**: Genera representaciones compactas y abstractas.
- **Indexación**: Uso de FAISS (Facebook AI Similarity Search) para búsquedas eficientes.
- **Interfaz gráfica**: Selecciona imágenes de prueba o carga nuevas y elige el método de comparación.

## Dataset
- **Fuente**: Dataset reducido de [WikiArt Art Movements/Styles](https://www.kaggle.com/datasets/sivarazadi/wikiart-art-movementsstyles).
- **Movimientos artísticos**: Modernismo, Barroco, Arte Japonés, Realismo, Rococó y Medieval Occidental.
