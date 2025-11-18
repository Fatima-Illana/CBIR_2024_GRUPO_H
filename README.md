# CONTENT BASED IMAGE RETRIEVAL FOR ART MOVEMENT STYLE CLASSIFICATION

This project implements a **Content-Based Image Retrieval (CBIR)** system focused on artworks classification. The objective is to retrieve images that belong to the same art movement as the one given as input by using different feature extraction techniques and machine learning.

## Project Preview

## Project Description

## Styles Considered

- `Art Nouveau`
- `Baroque`
- `Japanese Art`
- `Realism`
- `Rococo`
- `Western Medieval`

## Feature Extraction Methods Used

- `Color Histogram`
- `Textures Extractor`
- `Bag of Words`
- `Convolutional Neural Network (CNN)`: Specifically VGG19.
- `Autoencoder`

## Search Method:

- `Facebook AI Similarity Search (FAISS)`: for fast searches

## Results

## Files

- **database Folder**: Folder with the files used to work with the database of images and the indexes.
- **Bag_Of_Words.npy**: Contains the visual words dictionary used in Extractor 3 (Bag_Of_Words).
- **Dataset Arte.zip**: Contains all the images used, organized by art movements.  
- **DatasetArteTrainTest.zip**: Contains the images already split into Train and Test sets with renamed files.
- The extractor files contain the processes followed to generate the indexes:
  - **Extractor1-HistogramaColor.ipynb**
  - **Extractor2-Texturas.ipynb**
  - **Extractor3-BagOfWords.ipynb**
  - **Extractor4-CNN-VGG19.ipynb**
  - **Extractor5-Autoencoder.ipynb**
- **Interfaz.py**: Contains the interface code.
- **PreprocesadoImagenes.ipynb**: Contains the process of dataset splitting and database creation.  
- **Resultados.ipynb**: Contains the calculations performed to extract the results.  
- **ResultadosCBIR.xlsx**: Contains the results obtained for all test images.  
- **autoencoder.keras**: Contains the model generated for Extractor 5 (Autoencoder).
- **CBIR_Report.pdf**: Report with all the process followed to preprocess the images, create each extractor and the results obtained.

The contents are only available in spanish.

## Execution

- To run the project, simply execute the file `Interfaz.py`, which contains the interface code.  
- Execution command: `py -m streamlit run interfaz.py`   
- Sample images to use in the interface are available in the `Test` folder of `DatasetArteTrainTest.zip`.      

Download all files and store them in the same folder. Otherwise, modify the file paths in the main code and run it.

## Requirements

- Python 3.7 or higher
- Keras version used: 3.7.0 

## Dataset

- **Source**: Reduced dataset from [WikiArt Art Movements/Styles](https://www.kaggle.com/datasets/sivarazadi/wikiart-art-movementsstyles)

## Authors

- [Fátima Fuchun Illana Guerra](https://github.com/Fatima-Illana)
- [Cristina Fernández Gómez](https://github.com/crisfernandez)
- [Ester Esteban Bruña](https://github.com/esteresteban)
