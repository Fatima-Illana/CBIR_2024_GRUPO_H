# CONTENT BASED IMAGE RETRIEVAL FOR ART MOVEMENT IDENTIFICATION

This project implements a **Content-Based Image Retrieval (CBIR)** system focused on art movement similarity search. The objective is to retrieve images that belong to the same art movement as the one given as input by using machine learning and different feature extraction techniques.

## Project Preview

<img src="./Images/Interfaz_CBIR.png" alt="Interface" width="1000"/>

## Project Description

The aim of this project is to facilitate an interface that, based on a given work of art image, retrieves similiar images stored in the database based on content. The main objective is to return images that belogn to the same art movement, although we have seen that it is not always the case, as the content of the images is not always related to the artistic style. 

The presented tool allows to select 5 different feature extraction methods, some of them working better than others in specific art movements. Once you have selected the method, you will have to upload an artistic image and you will be able to select the specific area you want the system to take into account.

Once the extraction of images is done you will receive 11 images, along with the corresponding precision related to the number of images returned. Most methods will lose accuracy as the number of matches increases.

## Styles Considered

- `Art Nouveau`
- `Baroque`
- `Japanese Art`
- `Realism`
- `Rococo`
- `Western Medieval`

<img src="./Images/Movimientos_CBIR.png" alt="ArtMovements" width="1000"/>

## Feature Extraction Methods

- `Color Histogram`
- `Texture Histogram`
- `Bag of Words`
- `Convolutional Neural Network (CNN)`: Specifically VGG19.
- `Autoencoder`

<img src="./Images/Autoencoder_CBIR.png" alt="Autoencoder" width="1000"/>

## Search Method:

- `Facebook AI Similarity Search (FAISS)`: for fast searches

## Results

As we have seen, the method that obtains the best results is the Autoencoder.

| **EXTRACTOR** | **AVERAGE ACCURACY** |
|:----------:|:-----------:|
| Color Histogram   | 0.65    |
| Texture Histogram   | 0.27    |
| Bag of Words   | 0.44    |
| VGG19   | 0.53    |
| Autoencoder   | 0.83    |

Sample of the results obtained with Autoencoder:

<img src="./Images/Resultados_Autoencoder.png" alt="Results" width="1000"/>

*A deeper analysis of the results obtained is detailed in the report.*

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

## Interface Code Author

- [Ángel Mario García Pedrero](https://github.com/amgp-upm)
