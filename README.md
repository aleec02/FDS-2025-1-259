# FDS-2025-1-259

# Análisis de Tendencias de Videos en YouTube India  
Sección 259 - Grupo 4 | Fundamentos de Data Science | UPC 2025-1

## Objetivo del Proyecto

Este proyecto aplica la metodología CRISP-DM para analizar un conjunto de datos de videos en tendencia en YouTube India, con el objetivo de extraer conocimiento valioso sobre el comportamiento de los usuarios, preferencias de contenido y dinámicas de interacción. La finalidad es generar valor accionable para empresas del rubro digital y marketing, fundamentando decisiones estratégicas mediante evidencia cuantitativa.


## Descripción del Dataset

El conjunto de datos original fue extraído de [Kaggle - YouTube Trending Video Statistics](https://www.kaggle.com/datasets/datasnaek/youtube-new). Para este proyecto se utilizó la versión correspondiente a India: `INvideos_cc50_202101.csv`, complementado con el archivo `IN_category_id.json`.

Además, se incorporaron las siguientes columnas en una versión modificada del dataset:

- `state`: estado del país asignado aleatoriamente
- `lat`, `lon`: coordenadas geográficas de cada estado
- `geometry`: geometría geoespacial para visualización opcional

## Metodología Aplicada: CRISP-DM

### 1. Comprensión del Negocio

- **Objetivo general**: Comprender qué tipo de contenido destaca en YouTube India y cómo varía su popularidad según categoría, canal, tiempo y ubicación.
- **Requerimientos planteados por el cliente**:
  - Identificar las categorías de mayor y menor aceptación.
  - Evaluar el engagement a través de likes, dislikes, comentarios y vistas.
  - Explorar tendencias temporales y geográficas.
  - Determinar la posibilidad de predicción sobre vistas y me gusta.

### 2. Comprensión de los Datos

- Se identificaron columnas clave como: `views`, `likes`, `dislikes`, `comment_count`, `category_id`, `channel_title`, `trending_date`.
- Se examinó la estructura, tipos de datos, valores faltantes y valores extremos.
- Se verificó la calidad de los datos: integridad, duplicados, codificación y consistencia semántica.

### 3. Preparación de los Datos

- Eliminación de duplicados y valores nulos.
- Conversión de tipos: numéricos, booleanos y categóricos.
- Tratamiento de outliers usando el rango intercuartílico (IQR).
- Creación de nuevas variables derivadas para enriquecer el análisis:

| Variable Derivada         | Descripción                                                  |
|---------------------------|--------------------------------------------------------------|
| `engagement_rate`         | Interacciones (likes + dislikes) divididas por vistas        |
| `like_dislike_ratio`      | Porcentaje de likes sobre el total de interacciones          |
| `comments_per_view`       | Comentarios en proporción a las vistas                       |
| `performance_tier`        | Segmentación por cuartiles de vistas (bajo, medio, alto)     |
| `views_comments_ratio`    | Indicador de participación basado en comentarios             |

Estas variables permiten responder no solo a preguntas descriptivas, sino también explorar relaciones explicativas para futuros modelos predictivos.

### 4. Análisis Exploratorio

El análisis exploratorio abordó ocho requerimientos planteados:

1. Categorías más frecuentes en tendencia
2. Categorías que más gustan y que menos gustan
3. Mejor ratio like/dislike por categoría
4. Mejores categorías en términos de comentarios por vista
5. Evolución semanal del volumen de videos en tendencia
6. Canales más frecuentes y menos frecuentes en tendencias
7. Estados con mayor volumen de vistas, likes y dislikes
8. Relación entre nivel de rendimiento y cantidad de comentarios

## Visualizaciones Generadas

Las visualizaciones generadas fueron exportadas automáticamente a la carpeta `data/graficos/`. Cada imagen refleja un análisis clave del conjunto de datos:

| Archivo                             | Descripción                                         |
|-------------------------------------|-----------------------------------------------------|
| 01_categorias_mayor_tendencia.png   | Top de categorías según frecuencia de aparición     |
| 02_categorias_mas_menos_gustan.png  | Promedio de likes por categoría (top y bottom)      |
| 03_mejor_ratio_likes_dislikes.png   | Mejor proporción de likes vs total interacciones    |
| 04_ratio_vistas_comentarios.png     | Relación entre vistas y comentarios por categoría    |
| 05_volumen_tendencia_tiempo.png     | Evolución semanal del número de videos en tendencia |
| 06_canales_tendencia_distribucion.png | Canales con más y menos videos en tendencia      |
| 07_estados_vistas_interacciones.png | Comparación entre estados según métricas clave      |

Además, se generó el archivo `07_estados_metricas.csv` con métricas por estado para potencial análisis geoespacial.

## Estructura del Repositorio

```
.
├── upc-2025-01-259-4-tf.py             # Script principal de análisis
├── data/
│   ├── all-data/
│   │   ├── INvideos_cc50_202101.csv    # Dataset base
│   │   └── IN_category_id.json         # Archivo de categorías
│   ├── graficos/                       # Visualizaciones generadas
│   └── geo/                            # Resultados por estado
└── README.md
```

## Conclusiones

- Las categorías más frecuentes y mejor valoradas incluyen música y entretenimiento.
- El `engagement_rate` y `comments_per_view` resultaron ser métricas más informativas que las vistas totales.
- Muchos canales aparecen solo una vez, lo cual indica una alta competitividad y rotación en la sección de tendencias.
- La distribución temporal muestra picos y caídas semanales que podrían asociarse a eventos particulares o estacionalidad.
- La visualización de métricas por estado revela diferencias notables en interacción por ubicación, útil para estrategias regionales de marketing.
- Se concluye que es viable preparar los datos para tareas de modelado supervisado, como regresión para estimar vistas o clasificación por nivel de rendimiento.

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulte el archivo [LICENSE](./LICENSE.md) para más detalles.

