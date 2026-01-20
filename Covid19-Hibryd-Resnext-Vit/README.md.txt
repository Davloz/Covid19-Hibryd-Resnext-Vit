# Detección automática de COVID-19 en radiografías de tórax usando ResNeXt50 y Vision Transformers

Este repositorio contiene la implementación del Trabajo de Integración Curricular (TIC),
cuyo objetivo es el desarrollo y evaluación de un modelo híbrido basado en la arquitectura
ResNeXt50_32x4d integrada con Vision Transformers (ViT) para la detección automática de
COVID-19 a partir de radiografías de tórax (Chest X-Ray, CXR).

El repositorio está orientado a garantizar la reproducibilidad del proceso experimental
descrito en el documento del TIC, incluyendo la preparación de datos, el entrenamiento
del modelo, la evaluación del desempeño y el análisis estadístico de los resultados.

---

## Contexto del dataset utilizado

En la fase de planificación del proyecto (PTIC) se contempló el uso del conjunto de datos
**BIMCV-COVID19**, un repositorio clínico de gran escala orientado a estudios de imagen
médica para COVID-19:

https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/

No obstante, debido a limitaciones prácticas asociadas al tamaño del dataset, así como a
los requerimientos de descarga y almacenamiento, la ejecución experimental del Trabajo de
Integración Curricular (TIC) se realizó utilizando el **COVID-19 Radiography Dataset**,
un conjunto de datos público ampliamente utilizado en investigaciones de clasificación de
radiografías de tórax.

El dataset empleado se encuentra disponible en Kaggle:

https://www.kaggle.com/datasets/preetviradiya/covid19-radiography-dataset

Por razones de licencia y tamaño, el conjunto de datos no se incluye en este repositorio.
Una vez descargado, debe ubicarse dentro de la carpeta `data/`.
---

## Estructura del repositorio

El proyecto se organiza mediante notebooks secuenciales que representan cada una de las
etapas del pipeline experimental descrito en el TIC.

### 📁 notebooks/

- `01_config_datos_F.ipynb`  
  Notebook correspondiente a la fase de preparación y configuración de los datos. Incluye
  la carga del conjunto de datos, la organización de las imágenes por clases, el
  preprocesamiento inicial de las radiografías de tórax y la verificación de la estructura
  del dataset. Esta etapa es fundamental para asegurar la consistencia de los datos
  utilizados durante el entrenamiento y la evaluación del modelo.

- `02_train_modelo_A.ipynb`  
  Implementa el entrenamiento del modelo híbrido ResNeXt50_32x4d con Vision Transformers
  bajo la **Configuración A**, utilizada como referencia para analizar el comportamiento
  inicial del modelo durante el proceso de aprendizaje.

- `02_train_modelo_B.ipynb`  
  Contiene el entrenamiento del modelo híbrido bajo la **Configuración B**, seleccionada
  como el modelo final del proyecto. En este notebook se realiza el entrenamiento completo
  del modelo, el almacenamiento de los checkpoints y el seguimiento del desempeño durante
  el proceso de entrenamiento.

- `02_train_modelo_C.ipynb`  
  Corresponde al entrenamiento del modelo bajo la **Configuración C**, diseñada para
  analizar variaciones en el comportamiento del modelo frente a cambios en la
  configuración de entrenamiento o en la arquitectura del modelo.

- `03_evaluacion_inferencia_F.ipynb`  
  Notebook destinado a la evaluación final del modelo mediante inferencia. Incluye la
  generación de predicciones sobre el conjunto de prueba, el cálculo de métricas de
  desempeño (precisión, sensibilidad, especificidad y F1-score), así como la construcción
  de matrices de confusión y curvas ROC y Precision-Recall. Los resultados obtenidos en
  este notebook corresponden a los reportados en la sección de Resultados del TIC.

- `04_estadistica_checkpoints_F.ipynb`  
  En este notebook se realiza el análisis estadístico de los resultados obtenidos a partir
  de los modelos entrenados, considerando los checkpoints generados durante el proceso de
  entrenamiento. El análisis permite evaluar la estabilidad de los resultados y reforzar
  la confiabilidad del desempeño reportado en el TIC.

---

## Configuración experimental

El modelo propuesto combina una red convolucional profunda ResNeXt50_32x4d como extractor
de características locales con un bloque basado en Vision Transformers encargado de
modelar dependencias espaciales globales. Esta integración permite capturar tanto patrones
locales como relaciones de largo alcance presentes en las radiografías de tórax.

El entrenamiento del modelo se realizó utilizando PyTorch. Se emplearon técnicas estándar
de entrenamiento supervisado, incluyendo partición del conjunto de datos en subconjuntos
de entrenamiento y prueba, así como el uso de funciones de pérdida y optimizadores
adecuados para tareas de clasificación binaria en imágenes médicas.

Debido a las demandas computacionales del modelo híbrido, el uso de GPU es recomendado
para reducir los tiempos de entrenamiento, aunque no es estrictamente obligatorio para la
ejecución de los notebooks.

