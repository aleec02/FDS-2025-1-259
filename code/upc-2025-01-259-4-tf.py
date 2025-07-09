import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium
import json
import warnings
import os
from datetime import datetime
from tqdm import tqdm

# ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# CONFIGURACIÓN GLOBAL 

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('default')
sns.set_palette("husl")

import plotly.io as pio
pio.renderers.default = "browser"

VERDE = '\033[92m'
AZUL = '\033[94m'
ROJO = '\033[91m'
MAGENTA = '\033[95m'
AMARILLO = '\033[93m'
ENDC = '\033[0m'

SEPARADOR = "=" * 70

CODIGO_PAIS = "IN"

# Nombres de países
NOMBRES_PAISES = {
    "DE": "Alemania", "FR": "Francia", "GB": "Gran Bretaña", "IN": "India"
}

# Estructura de carpetas 
CARPETA_DATOS = "./data/all-data"
CARPETA_PROCESADOS = "./data/processed"
CARPETA_SALIDAS = "./data/outputs"
CARPETA_GRAFICOS = "./data/graficos"

# Crear directorios si no existen
os.makedirs(CARPETA_PROCESADOS, exist_ok=True)
os.makedirs(CARPETA_SALIDAS, exist_ok=True)
os.makedirs(CARPETA_GRAFICOS, exist_ok=True)

PATRON_ARCHIVO_VIDEOS = "{country}videos_cc50_202101.csv"
PATRON_ARCHIVO_CATEGORIAS = "{country}_category_id.json"

INFO_PROYECTO = {
    'titulo': 'Análisis de Videos de YouTube en Tendencia',
    'seccion': '259',
    'metodologia': 'CRISP-DM',
    'objetivo': 'Analizar patrones de videos de YouTube en tendencia y crear modelos predictivos',
    'repositorio': 'FDS-2025-1-259'
}

PREGUNTAS_INVESTIGACION = {
    1: "¿Qué categorías de videos son las de mayor tendencia?",
    2: "¿Qué categorías de videos son los que más gustan? ¿Y las que menos gustan?",
    3: "¿Qué categorías de videos tienen la mejor proporción (ratio) de 'Me gusta' / 'No me gusta'?",
    4: "¿Qué categorías de videos tienen la mejor proporción (ratio) de 'Vistas' / 'Comentarios'?",
    5: "¿Cómo ha cambiado el volumen de los videos en tendencia a lo largo del tiempo?",
    6: "¿Qué Canales de YouTube son tendencia más frecuentemente? ¿Y cuáles con menos frecuencia?",
    7: "¿En qué Estados se presenta el mayor número de 'Vistas', 'Me gusta' y 'No me gusta'?",
    8: "¿Los videos en tendencia son los que mayor cantidad de comentarios positivos reciben?",
    9: "¿Es factible predecir el número de 'Vistas' o 'Me gusta' o 'No me gusta'?"
}

def obtener_archivos_pais(codigo_pais=CODIGO_PAIS):
    """Obtener rutas de archivos para el país seleccionado"""
    archivo_videos = os.path.join(CARPETA_DATOS, PATRON_ARCHIVO_VIDEOS.format(country=codigo_pais))
    archivo_categorias = os.path.join(CARPETA_DATOS, PATRON_ARCHIVO_CATEGORIAS.format(country=codigo_pais))
    
    return {
        'archivo_videos': archivo_videos,
        'archivo_categorias': archivo_categorias,
        'nombre_pais': NOMBRES_PAISES.get(codigo_pais, codigo_pais),
        'codigo_pais': codigo_pais
    }

def mostrar_configuracion_actual():
    """Mostrar configuración actual del proyecto"""
    archivos = obtener_archivos_pais()
    print(f"\nConfiguración actual del proyecto:")
    print(f"País: {VERDE}{archivos['nombre_pais']}{ENDC} ({VERDE}{archivos['codigo_pais']}{ENDC})")
    print(f"Archivo de videos: {MAGENTA}{os.path.basename(archivos['archivo_videos'])}{ENDC}")
    print(f"Archivo de categorías: {MAGENTA}{os.path.basename(archivos['archivo_categorias'])}{ENDC}")
    print(f"Carpeta de datos: {MAGENTA}{CARPETA_DATOS}{ENDC}")
    print(f"Carpeta procesados: {MAGENTA}{CARPETA_PROCESADOS}{ENDC}")
    print(f"Carpeta de salidas: {MAGENTA}{CARPETA_SALIDAS}{ENDC}")

def cargar_y_limpiar_datos():
    """Carga y limpia los datos de YouTube con variables derivadas fundamentales"""
    ruta_videos = os.path.join(CARPETA_DATOS, f'{CODIGO_PAIS}videos_cc50_202101.csv')
    ruta_categorias = os.path.join(CARPETA_DATOS, f'{CODIGO_PAIS}_category_id.json')
    
    print(f"{SEPARADOR}\n{AZUL}Cargando y limpiando datos de YouTube {NOMBRES_PAISES[CODIGO_PAIS]}{ENDC}")
    
    print(f"{AZUL}Cargando archivo CSV...{ENDC}")
    df = pd.read_csv(ruta_videos, low_memory=False)
    
    # mapeo de categorías
    print(f"{AZUL}Cargando categorías...{ENDC}")
    with open(ruta_categorias, 'r', encoding='utf-8') as f:
        categorias = json.load(f)
    mapeo_categorias = {int(item['id']): item['snippet']['title'] for item in categorias['items']}
    
    print(f"{AZUL}Limpiando datos...{ENDC}")
    columnas_numericas = ['views', 'likes', 'dislikes', 'comment_count', 'category_id']
    
    with tqdm(total=7, desc="Procesando limpieza") as pbar:
        df['category_id'] = pd.to_numeric(df['category_id'], errors='coerce').fillna(0).astype(int)
        df['categoria_nombre'] = df['category_id'].map(mapeo_categorias)
        pbar.update(1)
        
        # Limpiar valores numéricos
        for col in columnas_numericas:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        pbar.update(1)
        
        # Limpiar valores de texto
        for col in ['title', 'channel_title', 'description', 'tags']:
            df[col] = df[col].fillna('')
        pbar.update(1)
        
        # Limpiar valores booleanos
        for col in ['comments_disabled', 'ratings_disabled', 'video_error_or_removed']:
            df[col] = df[col].fillna(False)
        pbar.update(1)
        
        # Eliminar duplicados
        df = df.drop_duplicates()
        pbar.update(1)
        
        # Tratar outliers
        for col in ['views', 'likes', 'dislikes', 'comment_count']:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            limite_inferior = q1 - 3 * iqr
            limite_superior = q3 + 3 * iqr
            df[col] = np.clip(df[col], limite_inferior, limite_superior)
        pbar.update(1)
        
        # Crear variables derivadas
        print(f"\n{AZUL}Creando variables derivadas fundamentales para el análisis...{ENDC}")
        
        df['total_interactions'] = df['likes'] + df['dislikes']
        df['engagement_rate'] = np.where(df['views'] > 0, df['total_interactions'] / df['views'], 0)
        df['like_dislike_ratio'] = np.where(df['total_interactions'] > 0, 
                                           df['likes'] / df['total_interactions'], 0.5)
        df['views_comments_ratio'] = np.where(df['comment_count'] > 0, 
                                             df['views'] / df['comment_count'], df['views'])
        df['likes_per_view'] = np.where(df['views'] > 0, df['likes'] / df['views'], 0)
        df['comments_per_view'] = np.where(df['views'] > 0, df['comment_count'] / df['views'], 0)
        df['performance_tier'] = pd.cut(
            df['views'],
            bins=[0, df['views'].quantile(0.25), df['views'].quantile(0.75), float('inf')],
            labels=['bajo', 'medio', 'alto'],
            include_lowest=True
        )
        pbar.update(1)
    
    print(f"{VERDE}Dataset procesado exitosamente: {len(df):,} registros con {len(df.columns)} columnas{ENDC}")
    print(f"{VERDE}Variables derivadas creadas para análisis avanzado de engagement y performance{ENDC}")
    
    return df

def guardar_datos_procesados(df, nombre_archivo, descripcion=""):
    """Guardar datos procesados en la carpeta correspondiente"""
    ruta_archivo = os.path.join(CARPETA_PROCESADOS, f"{CODIGO_PAIS}_{nombre_archivo}.csv")
    try:
        df.to_csv(ruta_archivo, index=False, encoding='utf-8')
        print(f"Datos guardados: {VERDE}{nombre_archivo}{ENDC} - {descripcion}")
        print(f"Ubicación: {MAGENTA}{ruta_archivo}{ENDC}")
        return ruta_archivo
    except Exception as e:
        print(f"Error guardando datos: {ROJO}{e}{ENDC}")
        return None


# =============================
# FUNCIÓN DE MAPA INTERACTIVO 
# =============================

def crear_mapa_ratio_like_dislike(df):
    """Crear mapa interactivo único para ratio like/dislike por estado"""
    print(f"\n{AZUL}Creando mapa interactivo de ratio like/dislike por estado...{ENDC}")
    
    geo_data = df.groupby(['state', 'lat', 'lon']).agg({
        'like_dislike_ratio': 'mean',
        'likes': 'sum',
        'dislikes': 'sum',
        'video_id': 'count'
    }).reset_index()
    
    geo_data['ratio_global'] = geo_data['likes'] / (geo_data['likes'] + geo_data['dislikes'])
    geo_data = geo_data.dropna()
    
    # mapa centrado en India
    m = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles='OpenStreetMap'
    )
    
    for idx, row in geo_data.iterrows():
        popup_text = f"""
        <b>{row['state']}</b><br>
        Videos: {row['video_id']:,}<br>
        Likes: {row['likes']:,}<br>
        Dislikes: {row['dislikes']:,}<br>
        Ratio L/D: {row['ratio_global']:.3f} ({row['ratio_global']*100:.1f}%)
        """
        
        # Determinar color basado en ratio like/dislike
        if row['ratio_global'] > 0.85:
            color = 'green'
            categoria = 'Muy Positivo'
        elif row['ratio_global'] > 0.75:
            color = 'lightgreen'
            categoria = 'Positivo'
        elif row['ratio_global'] > 0.65:
            color = 'orange'
            categoria = 'Neutral'
        else:
            color = 'red'
            categoria = 'Negativo'
            
        radius = min(max(row['video_id'] / 50, 5), 25)
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=radius,
            popup=popup_text,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
    
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 150px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px; border-radius: 5px;">
    <h4 style="margin-top:0;">Ratio Like/Dislike</h4>
    <p style="margin:5px 0;"><i class="fa fa-circle" style="color:green"></i> <b>Muy Positivo:</b> >85%</p>
    <p style="margin:5px 0;"><i class="fa fa-circle" style="color:lightgreen"></i> <b>Positivo:</b> 75-85%</p>
    <p style="margin:5px 0;"><i class="fa fa-circle" style="color:orange"></i> <b>Neutral:</b> 65-75%</p>
    <p style="margin:5px 0;"><i class="fa fa-circle" style="color:red"></i> <b>Negativo:</b> <65%</p>
    <p style="margin:5px 0; font-size:10px;"><i>Tamaño = # videos</i></p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Guardar mapa
    nombre_mapa = f'{CODIGO_PAIS}_mapa_ratio_like_dislike.html'
    ruta_mapa = os.path.join(CARPETA_SALIDAS, nombre_mapa)
    m.save(ruta_mapa)
    
    print(f"Mapa interactivo guardado: {VERDE}{nombre_mapa}{ENDC}")
    print(f"Ubicación: {MAGENTA}{ruta_mapa}{ENDC}")
    
    return geo_data

# =======================
# FUNCIONES DE ANÁLISIS 
# =======================

def pre1_categorias_tendencia(df):
    """Pregunta 1: Que categorías de videos son las de mayor tendencia?"""
    print(f"\n{AZUL}Pregunta 1: Que categorías de videos son las de mayor tendencia?{ENDC}")
    
    tendencia_categorias = df['categoria_nombre'].value_counts().head(10)
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(tendencia_categorias)))
    
    bars = plt.barh(tendencia_categorias.index[::-1], tendencia_categorias.values[::-1], 
                   color=colors, edgecolor='navy', linewidth=1.2, alpha=0.8)
    
    plt.xlabel('Número de videos en tendencia', fontsize=13, fontweight='bold')
    plt.title('Top 10 Categorías con Mayor Frecuencia en Tendencias', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # add valores en las barras
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                f'{int(width):,}', va='center', ha='left', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    filename = os.path.join(CARPETA_GRAFICOS, '01_categorias_mayor_tendencia.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Gráfico guardado en: {MAGENTA}{filename}{ENDC}")
    print(f"La categoría con más videos en tendencia es: {VERDE}{tendencia_categorias.index[0]}{ENDC}")
    print(f"Total de videos: {VERDE}{tendencia_categorias.iloc[0]:,}{ENDC}")
    print(f"Las top 3 categorías representan {VERDE}{(tendencia_categorias.head(3).sum()/len(df)*100):.1f}%{ENDC} del total")

def pre2_categorias_mayor_menor_gusto(df):
    """Pregunta 2: Que categorías de videos son los que más gustan? Y las que menos gustan?"""
    print(f"\n{AZUL}Pregunta 2: Que categorías de videos son los que más gustan? Y las que menos gustan?{ENDC}")
    
    # Calcular likes promedio por categoría
    likes_por_categoria = df.groupby('categoria_nombre')['likes'].agg(['mean', 'count']).reset_index()
    likes_por_categoria = likes_por_categoria[likes_por_categoria['count'] >= 10]
    likes_por_categoria = likes_por_categoria.sort_values('mean', ascending=False)
    
    top_5_mas_gustan = likes_por_categoria.head(5)
    top_5_menos_gustan = likes_por_categoria.tail(5)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    colors_mas = plt.cm.Greens(np.linspace(0.5, 0.9, 5))
    colors_menos = plt.cm.Reds(np.linspace(0.5, 0.9, 5))
    
    bars1 = ax1.barh(top_5_mas_gustan['categoria_nombre'][::-1], 
                     top_5_mas_gustan['mean'][::-1], 
                     color=colors_mas, edgecolor='darkgreen', linewidth=1.2, alpha=0.8)
    ax1.set_xlabel('Likes promedio', fontsize=12, fontweight='bold')
    ax1.set_title('Top 5 Categorías que MAS Gustan', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, bar in enumerate(bars1):
        ax1.text(bar.get_width() + bar.get_width()*0.02, bar.get_y() + bar.get_height()/2,
                f'{int(bar.get_width()):,}', va='center', ha='left', fontsize=10, fontweight='bold')
    
    # Gráfico de categorías que menos gustan  
    bars2 = ax2.barh(top_5_menos_gustan['categoria_nombre'][::-1], 
                     top_5_menos_gustan['mean'][::-1], 
                     color=colors_menos, edgecolor='darkred', linewidth=1.2, alpha=0.8)
    ax2.set_xlabel('Likes promedio', fontsize=12, fontweight='bold')
    ax2.set_title('Top 5 Categorías que MENOS Gustan', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, bar in enumerate(bars2):
        ax2.text(bar.get_width() + bar.get_width()*0.02, bar.get_y() + bar.get_height()/2,
                f'{int(bar.get_width()):,}', va='center', ha='left', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filename = os.path.join(CARPETA_GRAFICOS, '02_categorias_mas_menos_gustan.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Gráfico guardado en: {MAGENTA}{filename}{ENDC}")
    print(f"Categoría que MAS gusta: {VERDE}{top_5_mas_gustan.iloc[0]['categoria_nombre']}{ENDC}")
    print(f"Promedio de likes: {VERDE}{top_5_mas_gustan.iloc[0]['mean']:,.0f}{ENDC}")
    print(f"Categoría que MENOS gusta: {AMARILLO}{top_5_menos_gustan.iloc[-1]['categoria_nombre']}{ENDC}")
    print(f"Promedio de likes: {AMARILLO}{top_5_menos_gustan.iloc[-1]['mean']:,.0f}{ENDC}")

def pre3_mejor_ratio_likes_dislikes(df):
    """Pregunta 3: Que categorías de videos tienen la mejor proporción (ratio) de Me gusta / No me gusta?"""
    print(f"\n{AZUL}Pregunta 3: Que categorías de videos tienen la mejor proporción (ratio) de Me gusta / No me gusta?{ENDC}")
    
    # Calcular ratio promedio por categoría
    ratio_por_categoria = df[df['total_interactions'] > 0].groupby('categoria_nombre').agg({
        'like_dislike_ratio': 'mean',
        'video_id': 'count'
    }).reset_index()
    
    ratio_por_categoria = ratio_por_categoria[ratio_por_categoria['video_id'] >= 15]
    ratio_por_categoria = ratio_por_categoria.sort_values('like_dislike_ratio', ascending=False)
    
    top_5 = ratio_por_categoria.head(5)
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_5)))
    
    bars = plt.barh(top_5['categoria_nombre'][::-1], 
                   top_5['like_dislike_ratio'][::-1],
                   color=colors, edgecolor='darkblue', linewidth=1.2, alpha=0.85)
    
    plt.xlabel('Ratio Me gusta / Total interacciones', fontsize=13, fontweight='bold')
    plt.title('Top 5 Categorías con Mejor Ratio Me gusta / No me gusta', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, bar in enumerate(bars):
        ratio_val = bar.get_width()
        percentage = ratio_val * 100
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{ratio_val:.3f} ({percentage:.1f}%)', 
                va='center', ha='left', fontsize=10, fontweight='bold')
    
    plt.xlim(0, 1.05)
    plt.tight_layout()
    filename = os.path.join(CARPETA_GRAFICOS, '03_mejor_ratio_likes_dislikes.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Gráfico guardado en: {MAGENTA}{filename}{ENDC}")
    print(f"Mejor categoría en ratio likes/dislikes: {VERDE}{top_5.iloc[0]['categoria_nombre']}{ENDC}")
    print(f"Ratio: {VERDE}{top_5.iloc[0]['like_dislike_ratio']:.3f}{ENDC}")
    print(f"Esto significa que {VERDE}{top_5.iloc[0]['like_dislike_ratio']*100:.1f}%{ENDC} de las interacciones son likes")

def pre4_ratio_vistas_comentarios(df):
    """Pregunta 4: Que categorías de videos tienen la mejor proporción (ratio) de Vistas / Comentarios?"""
    print(f"\n{AZUL}Pregunta 4: Que categorías de videos tienen la mejor proporción (ratio) de Vistas / Comentarios?{ENDC}")
    
    # Calcular métricas por categoría
    df_con_comentarios = df[df['comment_count'] > 0]
    ratio_vistas_comentarios = df_con_comentarios.groupby('categoria_nombre').agg({
        'views_comments_ratio': 'mean',
        'comments_per_view': 'mean', 
        'video_id': 'count'
    }).reset_index()
    
    ratio_vistas_comentarios = ratio_vistas_comentarios[ratio_vistas_comentarios['video_id'] >= 10]
    ratio_vistas_comentarios = ratio_vistas_comentarios.sort_values('comments_per_view', ascending=False)
    
    top_10_comentarios = ratio_vistas_comentarios.head(10)
    
    # Crear scatterplot
    plt.figure(figsize=(14, 9))
    
    # Normalizar tamaños para mejor visualización
    sizes = (top_10_comentarios['video_id'] - top_10_comentarios['video_id'].min() + 10) * 5
    
    scatter = plt.scatter(top_10_comentarios['views_comments_ratio'], 
                         top_10_comentarios['comments_per_view'] * 100,  # convertir a porcentaje
                         s=sizes,
                         c=range(len(top_10_comentarios)), 
                         cmap='plasma', alpha=0.7, edgecolors='black', linewidth=1.5)
    
    for i, row in top_10_comentarios.iterrows():
        plt.annotate(row['categoria_nombre'], 
                    (row['views_comments_ratio'], row['comments_per_view'] * 100),
                    xytext=(5, 5), textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                             edgecolor='gray', alpha=0.8))
    
    plt.xlabel('Ratio Vistas / Comentarios (menor = más engagement)', fontsize=13, fontweight='bold')
    plt.ylabel('Comentarios por Vista (%)', fontsize=13, fontweight='bold') 
    plt.title('Categorías con Mejor Engagement en Comentarios\n(tamaño = número de videos)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Ranking', fontsize=11)
    
    plt.tight_layout()
    filename = os.path.join(CARPETA_GRAFICOS, '04_ratio_vistas_comentarios.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Gráfico guardado en: {MAGENTA}{filename}{ENDC}")
    print(f"Mejor categoría en engagement de comentarios: {VERDE}{top_10_comentarios.iloc[0]['categoria_nombre']}{ENDC}")
    print(f"Comentarios por vista: {VERDE}{top_10_comentarios.iloc[0]['comments_per_view']*100:.2f}%{ENDC}")
    print(f"Ratio vistas/comentarios: {VERDE}{top_10_comentarios.iloc[0]['views_comments_ratio']:.1f}{ENDC}")


def pre5_volumen_tendencia_tiempo(df):
    """Pregunta 5: Como ha cambiado el volumen de los videos en tendencia a lo largo del tiempo?"""
    print(f"\n{AZUL}Pregunta 5: Como ha cambiado el volumen de los videos en tendencia a lo largo del tiempo?{ENDC}")
    
    if 'trending_date' in df.columns:
        # Convertir trending_date a datetime
        df['fecha_trending'] = pd.to_datetime(df['trending_date'], errors='coerce')
        df_fechas = df.dropna(subset=['fecha_trending'])
        
        # Agrupar por semana
        df_fechas['semana'] = df_fechas['fecha_trending'].dt.to_period('W')
        volumen_semanal = df_fechas.groupby('semana').size().reset_index(name='num_videos')
        
        # Crear gráfico de línea mejorado
        plt.figure(figsize=(15, 8))
        
        # Convertir periodos a fechas
        fechas = [p.start_time for p in volumen_semanal['semana']]
        
        # Gráfico principal
        plt.plot(fechas, volumen_semanal['num_videos'], 
                linewidth=3, color='steelblue', marker='o', markersize=6,
                markeredgecolor='darkblue', markeredgewidth=1.5, alpha=0.9)
        
        # Línea de promedio
        promedio = volumen_semanal['num_videos'].mean()
        
        # Área sombreada
        plt.fill_between(fechas,
                 promedio - 10,
                 promedio + 10,
                 color='lightblue',
                 alpha=0.25,
                 label='Zona promedio ±10')
        
        plt.axhline(y=promedio, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Promedio: {promedio:.0f} videos')
        
        plt.xlabel('Fecha (semanas)', fontsize=13, fontweight='bold')
        plt.ylabel('Número de videos en tendencia', fontsize=13, fontweight='bold')
        plt.title('Evolución Temporal del Volumen de Videos en Tendencia\n(Agregación Semanal)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45)
        plt.legend(fontsize=12)
        
        # Resaltar pico y mínimo
        max_idx = volumen_semanal['num_videos'].idxmax()
        min_idx = volumen_semanal['num_videos'].idxmin()
        
        plt.annotate(f'Pico: {volumen_semanal.iloc[max_idx]["num_videos"]} videos',
                    xy=(fechas[max_idx], volumen_semanal.iloc[max_idx]['num_videos']),
                    xytext=(10, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
        
        plt.annotate(f'Mínimo: {volumen_semanal.iloc[min_idx]["num_videos"]} videos',
                    xy=(fechas[min_idx], volumen_semanal.iloc[min_idx]['num_videos']),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3'))
        
        plt.tight_layout()
        filename = os.path.join(CARPETA_GRAFICOS, '05_volumen_tendencia_tiempo.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Gráfico guardado en: {MAGENTA}{filename}{ENDC}")
        print(f"Semana con mayor volumen: {VERDE}{volumen_semanal.iloc[max_idx]['semana']}{ENDC}")
        print(f"Número de videos: {VERDE}{volumen_semanal.iloc[max_idx]['num_videos']}{ENDC}")
        print(f"Promedio semanal: {VERDE}{volumen_semanal['num_videos'].mean():.1f}{ENDC} videos")
        
    else:
        print(f"{ROJO}No se encontró la columna trending_date en los datos{ENDC}")


def pre6_canales_mayor_menor_tendencia(df):
    """Analiza los canales con más apariciones en tendencias"""
    print(f"\n{AZUL}Pregunta 6: ¿Qué canales de YouTube son tendencia más frecuentemente? y cuáles con menos frecuencia?{ENDC}")
    
    # Limpieza de nombres de canal
    df['channel_title'] = df['channel_title'].str.strip()
    df = df[df['channel_title'].astype(bool)]
    
    # Análisis de frecuencia
    canales_trending = df['channel_title'].value_counts()
    top_canales = canales_trending.head(5)
    top_channel = top_canales.index[0]
    top_count = top_canales.iloc[0]
    canales_unicos = canales_trending[canales_trending == 1]
    
    plt.figure(figsize=(14, 8))
    plt.rcParams['font.family'] = 'dejavu sans'
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(top_canales)))
    bars = plt.barh(top_canales.index[::-1], top_canales.values[::-1], 
                   color=colors, edgecolor='navy', linewidth=1.2, alpha=0.8)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 5, bar.get_y() + bar.get_height()/2,
                f'{int(width)}', 
                va='center', ha='left', 
                fontsize=10, fontweight='bold')
    
    plt.xlabel('Número de videos en tendencia', fontsize=12, fontweight='bold')
    plt.ylabel('Canales', fontsize=12, fontweight='bold')
    plt.title('Top 5 Canales con Más Videos en Tendencia', 
             fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    filename = os.path.join(CARPETA_GRAFICOS, '06_canales_tendencia_distribucion.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n{VERDE}Resultados clave:{ENDC}")
    print(f"Canal con más videos en tendencia: {VERDE}{top_channel}{ENDC}")
    print(f"Número de videos en tendencia: {VERDE}{top_count}{ENDC}")
    print(f"\n{VERDE}Distribución de canales:{ENDC}")
    print(f"- Canales con 1 aparición: {AMARILLO}{len(canales_unicos):,}{ENDC} ({len(canales_unicos)/len(canales_trending)*100:.1f}%)")
    print(f"- Canales con 2-5 apariciones: {AMARILLO}{len(canales_trending[(canales_trending > 1) & (canales_trending <= 5)]):,}{ENDC}")
    print(f"- Canales con 6-10 apariciones: {AMARILLO}{len(canales_trending[(canales_trending > 5) & (canales_trending <= 10)]):,}{ENDC}")
    print(f"- Canales con más de 10 apariciones: {AMARILLO}{len(canales_trending[canales_trending > 10]):,}{ENDC}")
    print(f"\n{VERDE}Total de canales analizados: {AMARILLO}{len(canales_trending):,}{ENDC}")
    print(f"\nGráfico guardado en: {MAGENTA}{filename}{ENDC}")

def pre7_estados_vistas_interacciones(df):
    """Pregunta 7: En que Estados se presenta el mayor número de Vistas, Me gusta y No me gusta?"""
    print(f"\n{AZUL}Pregunta 7: En que Estados se presenta el mayor número de Vistas, Me gusta y No me gusta?{ENDC}")
    
    if 'state' in df.columns:
        # Agregar métricas por estado
        estados_stats = df.groupby('state').agg({
            'views': 'sum',
            'likes': 'sum', 
            'dislikes': 'sum',
            'video_id': 'count'
        }).reset_index()
        
        estados_stats = estados_stats.sort_values('views', ascending=False).head(5)
        
        # Stacked bar chart
        fig, ax = plt.subplots(figsize=(15, 9))
        
        width = 0.75
        x = np.arange(len(estados_stats))
        
        # Normalizar para mejor visualización (en millones)
        views_m = estados_stats['views'] / 1e6
        likes_m = estados_stats['likes'] / 1e6  
        dislikes_m = estados_stats['dislikes'] / 1e6
        
        # Crear barras apiladas
        p1 = ax.bar(x, views_m, width, label='Vistas (M)', 
                   color='steelblue', alpha=0.8, edgecolor='darkblue', linewidth=1.2)
        p2 = ax.bar(x, likes_m, width, bottom=views_m, label='Likes (M)', 
                   color='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=1.2)
        p3 = ax.bar(x, dislikes_m, width, bottom=views_m + likes_m, label='Dislikes (M)', 
                   color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=1.2)
        
        ax.set_xlabel('Estados', fontsize=13, fontweight='bold')
        ax.set_ylabel('Cantidad (Millones)', fontsize=13, fontweight='bold')
        ax.set_title('Top 5 Estados por Vistas, Likes y Dislikes\n(valores en millones)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(estados_stats['state'], rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Agregar anotaciones en barras principales
        for i, (idx, row) in enumerate(estados_stats.iterrows()):
            total = views_m.iloc[i] + likes_m.iloc[i] + dislikes_m.iloc[i]
            ax.text(i, total + total*0.01, f'{total:.1f}M', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        filename = os.path.join(CARPETA_GRAFICOS, '07_estados_vistas_interacciones.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Guardar datos geográficos
        geo_filename = os.path.join(CARPETA_PROCESADOS, '07_estados_metricas.csv')
        estados_stats.to_csv(geo_filename, index=False)
        
        print(f"Gráfico guardado en: {MAGENTA}{filename}{ENDC}")
        print(f"Datos geográficos guardados en: {MAGENTA}{geo_filename}{ENDC}")
        print(f"Estado con más vistas: {VERDE}{estados_stats.iloc[0]['state']}{ENDC}")
        print(f"Total de vistas: {VERDE}{estados_stats.iloc[0]['views']:,}{ENDC}")
        print(f"Total de videos: {VERDE}{estados_stats.iloc[0]['video_id']:,}{ENDC}")
        
    else:
        print(f"{ROJO}No se encontró la columna state en los datos{ENDC}")

def pre8_tendencia_vs_comentarios_positivos(df):
    """Pregunta 8: Los videos en tendencia son los que mayor cantidad de comentarios positivos reciben?"""
    print(f"\n{AZUL}Pregunta 8: Los videos en tendencia son los que mayor cantidad de comentarios positivos reciben?{ENDC}")
    
    # Analizar relación entre performance tier y comentarios
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Preparar datos para boxplot
    data_boxplot = []
    labels_boxplot = []
    colors_boxplot = ['#ffcccc', '#ffffcc', '#ccffcc']
    
    for tier, color in zip(['bajo', 'medio', 'alto'], colors_boxplot):
        data = df[df['performance_tier'] == tier]['comment_count'].values
        if len(data) > 0:
            data_boxplot.append(data)
            labels_boxplot.append(f'{tier.capitalize()}\nrendimiento')
    
    # Boxplot
    bp = ax1.boxplot(data_boxplot, labels=labels_boxplot, patch_artist=True, 
                     showmeans=True, meanline=True, showfliers=False)
    
    # Colorear cajas
    for patch, color in zip(bp['boxes'], colors_boxplot):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(1.5)
    
    ax1.set_ylabel('Número de comentarios', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Nivel de rendimiento (basado en vistas)', fontsize=12, fontweight='bold')
    ax1.set_title('Distribución de Comentarios por Nivel de Rendimiento', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Agregar estadísticas
    stats_comentarios = df.groupby('performance_tier')['comment_count'].agg(['mean', 'median', 'count'])
    
    # Segundo gráfico: scatter plot con tendencia
    ax2.scatter(df['views']/1e6, df['comment_count'], 
               alpha=0.3, s=30, c=df['likes'], cmap='viridis')
    
    # Agregar línea de tendencia
    z = np.polyfit(df['views']/1e6, df['comment_count'], 1)
    p = np.poly1d(z)
    ax2.plot(df['views'].sort_values()/1e6, p(df['views'].sort_values()/1e6), 
            "r--", linewidth=2, alpha=0.8, label='Tendencia lineal')
    
    ax2.set_xlabel('Vistas (Millones)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Número de comentarios', fontsize=12, fontweight='bold')
    ax2.set_title('Relación entre Vistas y Comentarios\n(color = likes)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend()
    
    # Estadísticas en el primer gráfico
    textstr = 'Estadísticas promedio:\n'
    for tier, stats in stats_comentarios.iterrows():
        textstr += f'{tier.capitalize()}: {stats["mean"]:.0f} comentarios\n'
    
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', 
                                             edgecolor='gray', alpha=0.8))
    
    plt.tight_layout()
    filename = os.path.join(CARPETA_GRAFICOS, '08_tendencia_vs_comentarios.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Gráfico guardado en: {MAGENTA}{filename}{ENDC}")
    print("Estadísticas de comentarios por nivel de rendimiento:")
    for tier, stats in stats_comentarios.iterrows():
        print(f"  {tier.upper()}: promedio={VERDE}{stats['mean']:.1f}{ENDC}, "
              f"mediana={VERDE}{stats['median']:.1f}{ENDC}, videos={VERDE}{stats['count']}{ENDC}")
    
    # Análisis de correlación
    correlacion = df['views'].corr(df['comment_count'])
    print(f"Correlación vistas-comentarios: {VERDE}{correlacion:.3f}{ENDC}")

def pre9_prediccion_vistas_likes_dislikes(df):
    """Pregunta 9: Es factible predecir el número de Vistas o Me gusta o No me gusta?"""
    print(f"\n{AZUL}Pregunta 9: Es factible predecir el número de Vistas o Me gusta o No me gusta?{ENDC}")
    
    print(f"{AZUL}Preparando datos para modelado...{ENDC}")
    
    # Preparar features para predicción
    features_numericas = ['likes', 'dislikes', 'comment_count', 'engagement_rate']
    
    # Limpiar datos para ML
    df_ml = df.dropna(subset=features_numericas + ['views'])
    df_ml = df_ml[df_ml['views'] > 0]
    
    X = df_ml[features_numericas]
    y = df_ml['views']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Conjunto de entrenamiento: {VERDE}{len(X_train):,}{ENDC} registros")
    print(f"Conjunto de prueba: {VERDE}{len(X_test):,}{ENDC} registros")
    
    print(f"{AZUL}Entrenando modelos...{ENDC}")
    
    # Modelos
    modelos = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Linear Regression': LinearRegression()
    }
    
    resultados_modelos = {}
    
    for nombre_modelo, modelo in modelos.items():
        print(f"\nEntrenando {nombre_modelo}...")
        
        if nombre_modelo == 'Linear Regression':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            modelo.fit(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)
        else:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
        
        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        resultados_modelos[nombre_modelo] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'modelo': modelo
        }
        
        color_r2 = VERDE if r2 > 0.5 else AZUL if r2 > 0.2 else AMARILLO
        print(f"  RMSE: {AZUL}{rmse:,.0f}{ENDC}")
        print(f"  MAE: {AZUL}{mae:,.0f}{ENDC}")
        print(f"  R² Score: {color_r2}{r2:.3f}{ENDC}")
    
    if resultados_modelos:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Resultados de Modelado Predictivo - Views - {NOMBRES_PAISES[CODIGO_PAIS]}', fontsize=16)
        
        for i, (nombre_modelo, datos) in enumerate(resultados_modelos.items()):
            if i < 2:
                # Obtener predicciones
                if nombre_modelo == 'Linear Regression':
                    y_pred_viz = datos['modelo'].predict(scaler.transform(X_test))
                else:
                    y_pred_viz = datos['modelo'].predict(X_test)
                
                # Scatter plot: Real vs Predicho
                axes[i].scatter(y_test, y_pred_viz, alpha=0.6, color='blue')
                
                # Línea diagonal perfecta
                min_val = min(y_test.min(), y_pred_viz.min())
                max_val = max(y_test.max(), y_pred_viz.max())
                axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                axes[i].set_xlabel('Views Real')
                axes[i].set_ylabel('Views Predicho')
                axes[i].set_title(f'{nombre_modelo}\nR² = {datos["r2"]:.3f}')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(CARPETA_GRAFICOS, '09_prediccion_vistas_modelos.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    print(f"\nGráfico guardado en: {MAGENTA}{filename}{ENDC}")
    print("\nResultados de predicción de vistas:")
    for nombre_modelo, datos in resultados_modelos.items():
        print(f"  {nombre_modelo} - R²: {VERDE}{datos['r2']:.3f}{ENDC}, MAE: {VERDE}{datos['mae']/1e6:.2f}M{ENDC}")
    
    print(f"\n{SEPARADOR}")
    print("RESPUESTA PREGUNTA 9")
    print(f"{SEPARADOR}")
    
    mejor_r2 = max(modelo['r2'] for modelo in resultados_modelos.values())
    
    if mejor_r2 > 0.7:
        factibilidad = "MUY FACTIBLE"
        color = VERDE
    elif mejor_r2 > 0.5:
        factibilidad = "FACTIBLE"
        color = VERDE
    elif mejor_r2 > 0.3:
        factibilidad = "PARCIALMENTE FACTIBLE"
        color = AZUL
    else:
        factibilidad = "POCO FACTIBLE"
        color = AMARILLO
    
    print(f"Predicción de VIEWS: {color}{factibilidad}{ENDC} (R² = {color}{mejor_r2:.3f}{ENDC})")
    
    mejor_modelo_nombre = max(resultados_modelos.items(), key=lambda x: x[1]['r2'])[0]
    print(f"Mejor modelo: {VERDE}{mejor_modelo_nombre}{ENDC}")


def main():
    """Función principal que ejecuta todo el análisis"""
    
    print(f"\n{SEPARADOR}")
    print("ANÁLISIS DE VIDEOS DE YOUTUBE EN TENDENCIA")
    print(f"{SEPARADOR}")

    print(f"""
Proyecto de Ciencia de Datos - Metodología CRISP-DM

Objetivo del Proyecto:
Una consultora internacional, con sede en Lima, solicita desarrollar un proyecto de Ciencia 
de Datos con la finalidad de conocer las tendencias de los videos de YouTube en países importantes. 

El proyecto responde a la necesidad de su cliente, una importante empresa de marketing 
digital, que desea obtener respuestas a varios requerimientos de información sobre:
- Categorías de videos más populares
- Patrones temporales de tendencias
- Análisis geográfico de engagement
- Predicción de métricas de rendimiento

Alcance:
Análisis de datos de YouTube para el país: {VERDE}{NOMBRES_PAISES[CODIGO_PAIS]}{ENDC} ({VERDE}{CODIGO_PAIS}{ENDC})
Aplicando la metodología {AZUL}CRISP-DM{ENDC} para crear conocimiento y valor a partir de los datos.
    """)

    mostrar_configuracion_actual()
    
    archivos = obtener_archivos_pais(CODIGO_PAIS)
    videos_existe = os.path.exists(archivos['archivo_videos'])
    categorias_existe = os.path.exists(archivos['archivo_categorias'])
    
    print(f"\nValidación de archivos:")
    if videos_existe:
        print(f"Videos CSV: {VERDE}Encontrado{ENDC} - {MAGENTA}{os.path.basename(archivos['archivo_videos'])}{ENDC}")
    else:
        print(f"Videos CSV: {ROJO}NO ENCONTRADO{ENDC} - {MAGENTA}{os.path.basename(archivos['archivo_videos'])}{ENDC}")

    if categorias_existe:
        print(f"Categorías JSON: {VERDE}Encontrado{ENDC} - {MAGENTA}{os.path.basename(archivos['archivo_categorias'])}{ENDC}")
    else:
        print(f"Categorías JSON: {ROJO}NO ENCONTRADO{ENDC} - {MAGENTA}{os.path.basename(archivos['archivo_categorias'])}{ENDC}")
    
    if not (videos_existe and categorias_existe):
        print(f"\n{AMARILLO}ADVERTENCIA:{ENDC} Algunos archivos no se encontraron.")
        print(f"Verifique que los archivos estén en: {MAGENTA}{CARPETA_DATOS}{ENDC}")
        print(f"Archivos esperados:")
        print(f"  - {MAGENTA}{PATRON_ARCHIVO_VIDEOS.format(country=CODIGO_PAIS)}{ENDC}")
        print(f"  - {MAGENTA}{PATRON_ARCHIVO_CATEGORIAS.format(country=CODIGO_PAIS)}{ENDC}")
        print(f"{ROJO}Saliendo del programa...{ENDC}")
        exit(1)
    
    df = cargar_y_limpiar_datos()
    
    print(f"\n{SEPARADOR}")
    print("PREGUNTAS DE INVESTIGACIÓN")
    print(f"{SEPARADOR}")

    for i, pregunta in PREGUNTAS_INVESTIGACION.items():
        print(f"{VERDE}{i}.{ENDC} {pregunta}")
    
    preguntas = [
        ("Pregunta 1: Categorías de mayor tendencia", pre1_categorias_tendencia),
        ("Pregunta 2: Categorías que más/menos gustan", pre2_categorias_mayor_menor_gusto),
        ("Pregunta 3: Mejor ratio likes/dislikes", pre3_mejor_ratio_likes_dislikes),
        ("Pregunta 4: Ratio vistas/comentarios", pre4_ratio_vistas_comentarios),
        ("Pregunta 5: Volumen en el tiempo", pre5_volumen_tendencia_tiempo),
        ("Pregunta 6: Canales más/menos frecuentes", pre6_canales_mayor_menor_tendencia),
        ("Pregunta 7: Estados con más interacciones", pre7_estados_vistas_interacciones),
        ("Pregunta 8: Tendencia vs comentarios", pre8_tendencia_vs_comentarios_positivos),
        ("Pregunta 9: Predicción de métricas", pre9_prediccion_vistas_likes_dislikes)
    ]
    
    with tqdm(total=len(preguntas), desc="Procesando preguntas") as pbar:
        for descripcion, funcion in preguntas:
            funcion(df)
            pbar.update(1)
    
    # mapa interactivo único
    if 'state' in df.columns and 'lat' in df.columns and 'lon' in df.columns:
        print(f"\n{SEPARADOR}")
        print("CREANDO MAPA INTERACTIVO")
        print(f"{SEPARADOR}")
        
        try:
            geo_data = crear_mapa_ratio_like_dislike(df)
            print(f"{VERDE}Mapa interactivo creado exitosamente{ENDC}")
        except Exception as e:
            print(f"{ROJO}Error creando mapa interactivo: {e}{ENDC}")
    
    guardar_datos_procesados(df, "dataset_final", "Dataset completo procesado con todas las variables")
    
    print(f"\n{SEPARADOR}")
    print("RESUMEN FINAL DEL PROYECTO")
    print(f"{SEPARADOR}")
    
    print(f"""
{VERDE}PROYECTO CRISP-DM COMPLETADO{ENDC}


ANÁLISIS GEOGRÁFICO:
- Mapa interactivo ratio like/dislike: {VERDE}Creado{ENDC}
- Análisis por estados: {VERDE}Completado{ENDC}

MODELADO PREDICTIVO:
- Algoritmos implementados: Random Forest + Linear Regression
- Métricas evaluadas: R^2, RMSE, MAE
- Factibilidad de predicción: {VERDE}Evaluada{ENDC}

ARCHIVOS GENERADOS:
- Gráficos PNG: {VERDE}Guardados en /data/graficos/{ENDC}
- Datos procesados: {VERDE}Guardados en /data/processed/{ENDC}
- Mapa HTML interactivo: {VERDE}Guardado en /data/outputs/{ENDC}
""")


if __name__ == "__main__":
    main()