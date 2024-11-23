#!/usr/bin/env python
# coding: utf-8

# # **Sistema recomendador basado en NMF usando el dataset movielens**

# # Importar librerias y Cargar Datos
# Se cargan las calificaciones y detalles de películas.

# In[8]:


from surprise import NMF
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

# Cargar el dataset de calificaciones desde un archivo CSV
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

### 1. Cargar los datos en el formato requerido por Surprise

reader = Reader(rating_scale=(0.5, 5))  # Rango de calificaciones en el dataset
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)


# # Dividir los datos en entrenamiento y prueba

# In[9]:


trainset, testset = train_test_split(data, test_size=0.2, random_state=42)


# In[10]:


# Entrenar el modelo


# In[11]:


model = NMF(n_factors=20, random_state=42)  # NMF con 20 factores latentes
model.fit(trainset)


# # Evaluar el modelo en el conjunto de prueba

# In[12]:


# Realizar predicciones en el conjunto de prueba
predictions = model.test(testset)

# Calcular métricas de evaluación
mse = accuracy.mse(predictions, verbose=True)

### 5. Función para Predecir Calificaciones


# # Función para Predicción
# Predice la calificación que un usuario daría a una película específica. Retorna un objeto Predicción, del cual se extrae el valor predicho (est).

# In[13]:


def predecir_calificacion(user_id, movie_id):
    """
    Predice la calificación de un usuario para una película específica.
    
    Parámetros:
    - user_id: ID del usuario
    - movie_id: ID de la película
    
    Retorna:
    - Calificación predicha.
    """
    try:
        prediction = model.predict(user_id, movie_id)
        return prediction.est
    except Exception as e:
        print(f"Error al predecir: {e}")
        return None


# # Función para Generar Recomendaciones
# - Filtrar películas vistas: Se excluyen las películas que el usuario ya ha calificado.
# - Predecir calificaciones para películas no vistas: Usa la función predecir_calificacion para calcular las calificaciones predichas.
# - Ordenar y seleccionar Top-N: Ordena las películas no vistas por calificación predicha y selecciona las mejores

# In[14]:


def recomendar_top_n(user_id, num_recommendations=10):
    """
    Genera recomendaciones para un usuario específico.
    
    Parámetros:
    - user_id: ID del usuario
    - num_recommendations: Número de recomendaciones a devolver.
    
    Retorna:
    - DataFrame con las películas recomendadas.
    """
    # Obtener todas las películas del dataset
    all_movies = movies['movieId'].unique()
    
    # Filtrar películas ya vistas por el usuario
    peliculas_vistas = ratings[ratings['userId'] == user_id]['movieId'].values
    peliculas_no_vistas = [movie for movie in all_movies if movie not in peliculas_vistas]
    
    # Predecir calificaciones para todas las películas no vistas
    recomendaciones = []
    for movie_id in peliculas_no_vistas:
        prediccion = predecir_calificacion(user_id, movie_id)
        if prediccion is not None:
            recomendaciones.append((movie_id, prediccion))
    
    # Ordenar las películas por calificación predicha en orden descendente
    recomendaciones = sorted(recomendaciones, key=lambda x: x[1], reverse=True)[:num_recommendations]
    
    # Crear un DataFrame con los títulos de las películas recomendadas
    recomendaciones_df = movies[movies['movieId'].isin([rec[0] for rec in recomendaciones])].copy()
    recomendaciones_df['Predicted Rating'] = [rec[1] for rec in recomendaciones]
    
    return recomendaciones_df


# In[ ]:







# # Ejemplo de Uso
# Predicción de calificación y Recomendación de películas

# In[15]:


# Predicción de calificación
user_id = 1  # Cambia el ID del usuario para probar con otros usuarios
movie_id = 2  # Cambia el ID de la película para probar con otras películas
prediccion = predecir_calificacion(user_id, movie_id)
print(f"\nPredicción de calificación para el usuario {user_id} en la película {movie_id}: {prediccion:.2f}" if prediccion else "No se pudo predecir la calificación.")

# Recomendación de películas
num_recommendations = 10
print(f"\nTop-{num_recommendations} recomendaciones para el usuario {user_id}:")
print(recomendar_top_n(user_id, num_recommendations))


# # Mostrar resultados de evaluación

# In[16]:


print(f"\nEvaluación del modelo NMF:")
print(f" - MSE: {mse:.4f}")
#print(f" - MAE: {mae:.4f}")


# In[ ]:




