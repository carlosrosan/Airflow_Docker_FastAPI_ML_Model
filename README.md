# Implementación de un Modelo de detección de números escritos a mano, basado en MNIST.
### AMq2 - CEIA - FIUBA

[MNIST Database of handrwitten digits](https://archive.ics.uci.edu/dataset/683/mnist+database+of+handwritten+digits).

La implementación incluye:

- En Apache Airflow, un DAG que obtiene los datos del repositorio, realiza limpieza y 
feature engineering, y guarda en el bucket `s3://data` los datos separados para entrenamiento 
y pruebas. MLflow hace seguimiento de este procesamiento.
- Un servicio de API del modelo, que toma el artefacto de MLflow y lo expone para realizar 
predicciones.
