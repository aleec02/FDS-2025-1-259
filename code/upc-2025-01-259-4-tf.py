import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import datetime
from tqdm import tqdm

VERDE = '\033[92m'
AZUL = '\033[94m'
ROJO = '\033[91m'
MAGENTA = '\033[95m'
AMARILLO = '\033[93m'
ENDC = '\033[0m'
SEPARADOR = "=" * 70

warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

graficos_dir = os.path.join('data', 'graficos')
geo_dir = os.path.join('data', 'geo')
os.makedirs(graficos_dir, exist_ok=True)
os.makedirs(geo_dir, exist_ok=True)

def cargar_y_limpiar_datos():
    """carga y limpia los datos de youtube india con variables derivadas fundamentales"""
    ruta_videos = os.path.join('data', 'all-data', 'INvideos_cc50_202101.csv')
    ruta_categorias = os.path.join('data', 'all-data', 'IN_category_id.json')
    
    print(f"{SEPARADOR}\n{AZUL}Cargando y limpiando datos de YouTube India{ENDC}")
    
    # cargar datos principales con progress bar
    print(f"{AZUL}Cargando archivo CSV...{ENDC}")
    df = pd.read_csv(ruta_videos, low_memory=False)
    
    # cargar mapeo de categorias
    import json
    print(f"{AZUL}Cargando categorias...{ENDC}")
    with open(ruta_categorias, 'r', encoding='utf-8') as f:
        categorias = json.load(f)
    mapeo_categorias = {int(item['id']): item['snippet']['title'] for item in categorias['items']}
    
    print(f"{AZUL}Limpiando datos...{ENDC}")
    columnas_numericas = ['views', 'likes', 'dislikes', 'comment_count', 'category_id']
    
    with tqdm(total=7, desc="Procesando limpieza") as pbar:
        # mapear categorias
        df['category_id'] = pd.to_numeric(df['category_id'], errors='coerce').fillna(0).astype(int)
        df['categoria_nombre'] = df['category_id'].map(mapeo_categorias)
        pbar.update(1)
        
        # limpiar valores numericos
        for col in columnas_numericas:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        pbar.update(1)
        
        # limpiar valores de texto
        for col in ['title', 'channel_title', 'description', 'tags']:
            df[col] = df[col].fillna('')
        pbar.update(1)
        
        # limpiar valores booleanos
        for col in ['comments_disabled', 'ratings_disabled', 'video_error_or_removed']:
            df[col] = df[col].fillna(False)
        pbar.update(1)
        
        # eliminar duplicados
        df = df.drop_duplicates()
        pbar.update(1)
        
        # tratar outliers
        for col in ['views', 'likes', 'dislikes', 'comment_count']:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            limite_inferior = q1 - 3 * iqr
            limite_superior = q3 + 3 * iqr
            df[col] = np.clip(df[col], limite_inferior, limite_superior)
        pbar.update(1)
        
        # crear variables derivadas
        print(f"\n{AZUL}Creando variables derivadas fundamentales para el analisis...{ENDC}")
        
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
    print(f"{VERDE}Variables derivadas creadas para analisis avanzado de engagement y performance{ENDC}")
    
    return df

def pre1_categorias_tendencia(df):
    """Pregunta 1: Que categorias de videos son las de mayor tendencia?"""
    print(f"\n{AZUL}Pregunta 1: Que categorias de videos son las de mayor tendencia?{ENDC}")
    
    # calcular frecuencia de categorias en trending
    tendencia_categorias = df['categoria_nombre'].value_counts().head(10)
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(tendencia_categorias)))
    
    bars = plt.barh(tendencia_categorias.index[::-1], tendencia_categorias.values[::-1], 
                   color=colors, edgecolor='navy', linewidth=1.2, alpha=0.8)
    
    plt.xlabel('Numero de videos en tendencia', fontsize=13, fontweight='bold')
    plt.title('Top 10 Categorias con Mayor Frecuencia en Tendencias', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # agregar valores en las barras
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                f'{int(width):,}', va='center', ha='left', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    filename = os.path.join(graficos_dir, '01_categorias_mayor_tendencia.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Grafico guardado en: {MAGENTA}{filename}{ENDC}")
    print(f"La categoria con mas videos en tendencia es: {VERDE}{tendencia_categorias.index[0]}{ENDC}")
    print(f"Total de videos: {VERDE}{tendencia_categorias.iloc[0]:,}{ENDC}")
    print(f"Las top 3 categorias representan {VERDE}{(tendencia_categorias.head(3).sum()/len(df)*100):.1f}%{ENDC} del total")

def pre2_categorias_mayor_menor_gusto(df):
    """Pregunta 2: Que categorias de videos son los que mas gustan? Y las que menos gustan?"""
    print(f"\n{AZUL}Pregunta 2: Que categorias de videos son los que mas gustan? Y las que menos gustan?{ENDC}")
    
    # calcular likes promedio por categoria
    likes_por_categoria = df.groupby('categoria_nombre')['likes'].agg(['mean', 'count']).reset_index()
    likes_por_categoria = likes_por_categoria[likes_por_categoria['count'] >= 10]
    likes_por_categoria = likes_por_categoria.sort_values('mean', ascending=False)
    
    top_5_mas_gustan = likes_por_categoria.head(5)
    top_5_menos_gustan = likes_por_categoria.tail(5)
    
    # crear figura con mejor diseño
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # colores degradados
    colors_mas = plt.cm.Greens(np.linspace(0.5, 0.9, 5))
    colors_menos = plt.cm.Reds(np.linspace(0.5, 0.9, 5))
    
    # grafico de categorias que mas gustan
    bars1 = ax1.barh(top_5_mas_gustan['categoria_nombre'][::-1], 
                     top_5_mas_gustan['mean'][::-1], 
                     color=colors_mas, edgecolor='darkgreen', linewidth=1.2, alpha=0.8)
    ax1.set_xlabel('Likes promedio', fontsize=12, fontweight='bold')
    ax1.set_title('Top 5 Categorias que MAS Gustan', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, bar in enumerate(bars1):
        ax1.text(bar.get_width() + bar.get_width()*0.02, bar.get_y() + bar.get_height()/2,
                f'{int(bar.get_width()):,}', va='center', ha='left', fontsize=10, fontweight='bold')
    
    # grafico de categorias que menos gustan  
    bars2 = ax2.barh(top_5_menos_gustan['categoria_nombre'][::-1], 
                     top_5_menos_gustan['mean'][::-1], 
                     color=colors_menos, edgecolor='darkred', linewidth=1.2, alpha=0.8)
    ax2.set_xlabel('Likes promedio', fontsize=12, fontweight='bold')
    ax2.set_title('Top 5 Categorias que MENOS Gustan', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, bar in enumerate(bars2):
        ax2.text(bar.get_width() + bar.get_width()*0.02, bar.get_y() + bar.get_height()/2,
                f'{int(bar.get_width()):,}', va='center', ha='left', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filename = os.path.join(graficos_dir, '02_categorias_mas_menos_gustan.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Grafico guardado en: {MAGENTA}{filename}{ENDC}")
    print(f"Categoria que MAS gusta: {VERDE}{top_5_mas_gustan.iloc[0]['categoria_nombre']}{ENDC}")
    print(f"Promedio de likes: {VERDE}{top_5_mas_gustan.iloc[0]['mean']:,.0f}{ENDC}")
    print(f"Categoria que MENOS gusta: {AMARILLO}{top_5_menos_gustan.iloc[-1]['categoria_nombre']}{ENDC}")
    print(f"Promedio de likes: {AMARILLO}{top_5_menos_gustan.iloc[-1]['mean']:,.0f}{ENDC}")

def pre3_mejor_ratio_likes_dislikes(df):
    """Pregunta 3: Que categorias de videos tienen la mejor proporcion (ratio) de Me gusta / No me gusta?"""
    print(f"\n{AZUL}Pregunta 3: Que categorias de videos tienen la mejor proporcion (ratio) de Me gusta / No me gusta?{ENDC}")
    
    # calcular ratio promedio por categoria
    ratio_por_categoria = df[df['total_interactions'] > 0].groupby('categoria_nombre').agg({
        'like_dislike_ratio': 'mean',
        'video_id': 'count'
    }).reset_index()
    
    ratio_por_categoria = ratio_por_categoria[ratio_por_categoria['video_id'] >= 15]
    ratio_por_categoria = ratio_por_categoria.sort_values('like_dislike_ratio', ascending=False)
    
    top_10_ratio = ratio_por_categoria.head(10)
    
    # crear grafico de barras horizontales con estilo mejorado
    plt.figure(figsize=(12, 8))
    
    # usar colormap degradado
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_10_ratio)))
    
    bars = plt.barh(top_10_ratio['categoria_nombre'][::-1], 
                   top_10_ratio['like_dislike_ratio'][::-1],
                   color=colors, edgecolor='darkblue', linewidth=1.2, alpha=0.85)
    
    plt.xlabel('Ratio Me gusta / Total interacciones', fontsize=13, fontweight='bold')
    plt.title('Top 10 Categorias con Mejor Ratio Me gusta / No me gusta', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # agregar valores en las barras con formato mejorado
    for i, bar in enumerate(bars):
        ratio_val = bar.get_width()
        percentage = ratio_val * 100
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{ratio_val:.3f} ({percentage:.1f}%)', 
                va='center', ha='left', fontsize=10, fontweight='bold')
    
    plt.xlim(0, 1.05)
    plt.tight_layout()
    filename = os.path.join(graficos_dir, '03_mejor_ratio_likes_dislikes.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Grafico guardado en: {MAGENTA}{filename}{ENDC}")
    print(f"Mejor categoria en ratio likes/dislikes: {VERDE}{top_10_ratio.iloc[0]['categoria_nombre']}{ENDC}")
    print(f"Ratio: {VERDE}{top_10_ratio.iloc[0]['like_dislike_ratio']:.3f}{ENDC}")
    print(f"Esto significa que {VERDE}{top_10_ratio.iloc[0]['like_dislike_ratio']*100:.1f}%{ENDC} de las interacciones son likes")

def pre4_ratio_vistas_comentarios(df):
    """Pregunta 4: Que categorias de videos tienen la mejor proporcion (ratio) de Vistas / Comentarios?"""
    print(f"\n{AZUL}Pregunta 4: Que categorias de videos tienen la mejor proporcion (ratio) de Vistas / Comentarios?{ENDC}")
    
    # calcular metricas por categoria
    df_con_comentarios = df[df['comment_count'] > 0]
    ratio_vistas_comentarios = df_con_comentarios.groupby('categoria_nombre').agg({
        'views_comments_ratio': 'mean',
        'comments_per_view': 'mean', 
        'video_id': 'count'
    }).reset_index()
    
    ratio_vistas_comentarios = ratio_vistas_comentarios[ratio_vistas_comentarios['video_id'] >= 10]
    ratio_vistas_comentarios = ratio_vistas_comentarios.sort_values('comments_per_view', ascending=False)
    
    top_10_comentarios = ratio_vistas_comentarios.head(10)
    
    # crear scatterplot mejorado
    plt.figure(figsize=(14, 9))
    
    # normalizar tamaños para mejor visualizacion
    sizes = (top_10_comentarios['video_id'] - top_10_comentarios['video_id'].min() + 10) * 5
    
    scatter = plt.scatter(top_10_comentarios['views_comments_ratio'], 
                         top_10_comentarios['comments_per_view'] * 100,  # convertir a porcentaje
                         s=sizes,
                         c=range(len(top_10_comentarios)), 
                         cmap='plasma', alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # agregar etiquetas mejoradas
    for i, row in top_10_comentarios.iterrows():
        plt.annotate(row['categoria_nombre'], 
                    (row['views_comments_ratio'], row['comments_per_view'] * 100),
                    xytext=(5, 5), textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                             edgecolor='gray', alpha=0.8))
    
    plt.xlabel('Ratio Vistas / Comentarios (menor = mas engagement)', fontsize=13, fontweight='bold')
    plt.ylabel('Comentarios por Vista (%)', fontsize=13, fontweight='bold') 
    plt.title('Categorias con Mejor Engagement en Comentarios\n(tamaño = numero de videos)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # agregar colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Ranking', fontsize=11)
    
    plt.tight_layout()
    filename = os.path.join(graficos_dir, '04_ratio_vistas_comentarios.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Grafico guardado en: {MAGENTA}{filename}{ENDC}")
    print(f"Mejor categoria en engagement de comentarios: {VERDE}{top_10_comentarios.iloc[0]['categoria_nombre']}{ENDC}")
    print(f"Comentarios por vista: {VERDE}{top_10_comentarios.iloc[0]['comments_per_view']*100:.2f}%{ENDC}")
    print(f"Ratio vistas/comentarios: {VERDE}{top_10_comentarios.iloc[0]['views_comments_ratio']:.1f}{ENDC}")

def pre5_volumen_tendencia_tiempo(df):
    """Pregunta 5: Como ha cambiado el volumen de los videos en tendencia a lo largo del tiempo?"""
    print(f"\n{AZUL}Pregunta 5: Como ha cambiado el volumen de los videos en tendencia a lo largo del tiempo?{ENDC}")
    
    if 'trending_date' in df.columns:
        # convertir trending_date a datetime
        df['fecha_trending'] = pd.to_datetime(df['trending_date'], errors='coerce')
        df_fechas = df.dropna(subset=['fecha_trending'])
        
        # agrupar por semana
        df_fechas['semana'] = df_fechas['fecha_trending'].dt.to_period('W')
        volumen_semanal = df_fechas.groupby('semana').size().reset_index(name='num_videos')
        
        # crear grafico de linea mejorado
        plt.figure(figsize=(15, 8))
        
        # convertir periodos a fechas
        fechas = [p.start_time for p in volumen_semanal['semana']]
        
        # grafico principal
        plt.plot(fechas, volumen_semanal['num_videos'], 
                linewidth=3, color='steelblue', marker='o', markersize=6,
                markeredgecolor='darkblue', markeredgewidth=1.5, alpha=0.9)
        
        # area sombreada
        plt.fill_between(fechas, volumen_semanal['num_videos'], 
                        alpha=0.25, color='lightblue')
        
        # linea de promedio
        promedio = volumen_semanal['num_videos'].mean()
        plt.axhline(y=promedio, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Promedio: {promedio:.0f} videos')
        
        plt.xlabel('Fecha (semanas)', fontsize=13, fontweight='bold')
        plt.ylabel('Numero de videos en tendencia', fontsize=13, fontweight='bold')
        plt.title('Evolucion Temporal del Volumen de Videos en Tendencia\n(Agregacion Semanal)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45)
        plt.legend(fontsize=12)
        
        # resaltar pico y minimo
        max_idx = volumen_semanal['num_videos'].idxmax()
        min_idx = volumen_semanal['num_videos'].idxmin()
        
        plt.annotate(f'Pico: {volumen_semanal.iloc[max_idx]["num_videos"]} videos',
                    xy=(fechas[max_idx], volumen_semanal.iloc[max_idx]['num_videos']),
                    xytext=(10, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
        
        plt.annotate(f'Minimo: {volumen_semanal.iloc[min_idx]["num_videos"]} videos',
                    xy=(fechas[min_idx], volumen_semanal.iloc[min_idx]['num_videos']),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3'))
        
        plt.tight_layout()
        filename = os.path.join(graficos_dir, '05_volumen_tendencia_tiempo.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Grafico guardado en: {MAGENTA}{filename}{ENDC}")
        print(f"Semana con mayor volumen: {VERDE}{volumen_semanal.iloc[max_idx]['semana']}{ENDC}")
        print(f"Numero de videos: {VERDE}{volumen_semanal.iloc[max_idx]['num_videos']}{ENDC}")
        print(f"Promedio semanal: {VERDE}{volumen_semanal['num_videos'].mean():.1f}{ENDC} videos")
        
    else:
        print(f"{ROJO}No se encontro la columna trending_date en los datos{ENDC}")



def pre6_canales_mayor_menor_tendencia(df):
    """analiza los canales con más apariciones en tendencias"""
    print(f"\n{AZUL}pregunta 6: qué canales de youtube son tendencia más frecuentemente? y cuáles con menos frecuencia?{ENDC}")
    
    # limpieza de nombres de canal
    df['channel_title'] = df['channel_title'].str.strip()
    df = df[df['channel_title'].astype(bool)]
    
    # análisis de frecuencia
    canales_trending = df['channel_title'].value_counts()
    top_canales = canales_trending.head(15)
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
    
    plt.xlabel('número de videos en tendencia', fontsize=12, fontweight='bold')
    plt.ylabel('canales', fontsize=12, fontweight='bold')
    plt.title('top 15 canales con más videos en tendencia', 
             fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    filename = os.path.join(graficos_dir, '06_canales_tendencia_distribucion.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n{VERDE}resultados clave:{ENDC}")
    print(f"canal con más videos en tendencia: {VERDE}{top_channel}{ENDC}")
    print(f"número de videos en tendencia: {VERDE}{top_count}{ENDC}")
    print(f"\n{VERDE}distribución de canales:{ENDC}")
    print(f"- canales con 1 aparición: {AMARILLO}{len(canales_unicos):,}{ENDC} ({len(canales_unicos)/len(canales_trending)*100:.1f}%)")
    print(f"- canales con 2-5 apariciones: {AMARILLO}{len(canales_trending[(canales_trending > 1) & (canales_trending <= 5)]):,}{ENDC}")
    print(f"- canales con 6-10 apariciones: {AMARILLO}{len(canales_trending[(canales_trending > 5) & (canales_trending <= 10)]):,}{ENDC}")
    print(f"- canales con más de 10 apariciones: {AMARILLO}{len(canales_trending[canales_trending > 10]):,}{ENDC}")
    print(f"\n{VERDE}total de canales analizados: {AMARILLO}{len(canales_trending):,}{ENDC}")
    print(f"\ngráfico guardado en: {MAGENTA}{filename}{ENDC}")



def pre7_estados_vistas_interacciones(df):
    """Pregunta 7: En que Estados se presenta el mayor numero de Vistas, Me gusta y No me gusta?"""
    print(f"\n{AZUL}Pregunta 7: En que Estados se presenta el mayor numero de Vistas, Me gusta y No me gusta?{ENDC}")
    
    if 'state' in df.columns:
        # agregar metricas por estado
        estados_stats = df.groupby('state').agg({
            'views': 'sum',
            'likes': 'sum', 
            'dislikes': 'sum',
            'video_id': 'count'
        }).reset_index()
        
        estados_stats = estados_stats.sort_values('views', ascending=False).head(12)
        
        # crear stacked bar chart mejorado
        fig, ax = plt.subplots(figsize=(15, 9))
        
        width = 0.75
        x = np.arange(len(estados_stats))
        
        # normalizar para mejor visualizacion (en millones)
        views_m = estados_stats['views'] / 1e6
        likes_m = estados_stats['likes'] / 1e6  
        dislikes_m = estados_stats['dislikes'] / 1e6
        
        # crear barras apiladas
        p1 = ax.bar(x, views_m, width, label='Vistas (M)', 
                   color='steelblue', alpha=0.8, edgecolor='darkblue', linewidth=1.2)
        p2 = ax.bar(x, likes_m, width, bottom=views_m, label='Likes (M)', 
                   color='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=1.2)
        p3 = ax.bar(x, dislikes_m, width, bottom=views_m + likes_m, label='Dislikes (M)', 
                   color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=1.2)
        
        # personalizar ejes
        ax.set_xlabel('Estados', fontsize=13, fontweight='bold')
        ax.set_ylabel('Cantidad (Millones)', fontsize=13, fontweight='bold')
        ax.set_title('Top 12 Estados por Vistas, Likes y Dislikes\n(valores en millones)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(estados_stats['state'], rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # agregar anotaciones en barras principales
        for i, (idx, row) in enumerate(estados_stats.iterrows()):
            total = views_m.iloc[i] + likes_m.iloc[i] + dislikes_m.iloc[i]
            ax.text(i, total + total*0.01, f'{total:.1f}M', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        filename = os.path.join(graficos_dir, '07_estados_vistas_interacciones.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # guardar datos geograficos
        geo_filename = os.path.join(geo_dir, '07_estados_metricas.csv')
        estados_stats.to_csv(geo_filename, index=False)
        
        print(f"Grafico guardado en: {MAGENTA}{filename}{ENDC}")
        print(f"Datos geograficos guardados en: {MAGENTA}{geo_filename}{ENDC}")
        print(f"Estado con mas vistas: {VERDE}{estados_stats.iloc[0]['state']}{ENDC}")
        print(f"Total de vistas: {VERDE}{estados_stats.iloc[0]['views']:,}{ENDC}")
        print(f"Total de videos: {VERDE}{estados_stats.iloc[0]['video_id']:,}{ENDC}")
        
    else:
        print(f"{ROJO}No se encontro la columna state en los datos{ENDC}")

def pre8_tendencia_vs_comentarios_positivos(df):
    """Pregunta 8: Los videos en tendencia son los que mayor cantidad de comentarios positivos reciben?"""
    print(f"\n{AZUL}Pregunta 8: Los videos en tendencia son los que mayor cantidad de comentarios positivos reciben?{ENDC}")
    
    # analizar relacion entre performance tier y comentarios
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # preparar datos para boxplot
    data_boxplot = []
    labels_boxplot = []
    colors_boxplot = ['#ffcccc', '#ffffcc', '#ccffcc']
    
    for tier, color in zip(['bajo', 'medio', 'alto'], colors_boxplot):
        data = df[df['performance_tier'] == tier]['comment_count'].values
        if len(data) > 0:
            data_boxplot.append(data)
            labels_boxplot.append(f'{tier.capitalize()}\nrendimiento')
    
    # crear boxplot mejorado
    bp = ax1.boxplot(data_boxplot, labels=labels_boxplot, patch_artist=True, 
                     showmeans=True, meanline=True, showfliers=False)
    
    # colorear cajas
    for patch, color in zip(bp['boxes'], colors_boxplot):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(1.5)
    
    ax1.set_ylabel('Numero de comentarios', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Nivel de rendimiento (basado en vistas)', fontsize=12, fontweight='bold')
    ax1.set_title('Distribucion de Comentarios por Nivel de Rendimiento', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # agregar estadisticas
    stats_comentarios = df.groupby('performance_tier')['comment_count'].agg(['mean', 'median', 'count'])
    
    # crear segundo grafico: scatter plot con tendencia
    ax2.scatter(df['views']/1e6, df['comment_count'], 
               alpha=0.3, s=30, c=df['likes'], cmap='viridis')
    
    # agregar linea de tendencia
    z = np.polyfit(df['views']/1e6, df['comment_count'], 1)
    p = np.poly1d(z)
    ax2.plot(df['views'].sort_values()/1e6, p(df['views'].sort_values()/1e6), 
            "r--", linewidth=2, alpha=0.8, label='Tendencia lineal')
    
    ax2.set_xlabel('Vistas (Millones)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Numero de comentarios', fontsize=12, fontweight='bold')
    ax2.set_title('Relacion entre Vistas y Comentarios\n(color = likes)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend()
    
    # agregar texto con estadisticas en el primer grafico
    textstr = 'Estadisticas promedio:\n'
    for tier, stats in stats_comentarios.iterrows():
        textstr += f'{tier.capitalize()}: {stats["mean"]:.0f} comentarios\n'
    
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', 
                                             edgecolor='gray', alpha=0.8))
    
    plt.tight_layout()
    filename = os.path.join(graficos_dir, '08_tendencia_vs_comentarios.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Grafico guardado en: {MAGENTA}{filename}{ENDC}")
    print("Estadisticas de comentarios por nivel de rendimiento:")
    for tier, stats in stats_comentarios.iterrows():
        print(f"  {tier.upper()}: promedio={VERDE}{stats['mean']:.1f}{ENDC}, "
              f"mediana={VERDE}{stats['median']:.1f}{ENDC}, videos={VERDE}{stats['count']}{ENDC}")
    
    # analisis de correlacion
    correlacion = df['views'].corr(df['comment_count'])
    print(f"Correlacion vistas-comentarios: {VERDE}{correlacion:.3f}{ENDC}")

def pre9_prediccion_vistas_likes_dislikes(df):
    """Pregunta 9: Es factible predecir el numero de Vistas o Me gusta o No me gusta?"""
    print(f"\n{AZUL}Pregunta 9: Es factible predecir el numero de Vistas o Me gusta o No me gusta?{ENDC}")
    
    # importar librerias para machine learning
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    
    print(f"{AZUL}Preparando datos para modelado...{ENDC}")
    
    # preparar features para prediccion
    features_numericas = ['likes', 'dislikes', 'comment_count', 'engagement_rate']
    
    # limpiar datos para ML
    df_ml = df.dropna(subset=features_numericas + ['views'])
    df_ml = df_ml[df_ml['views'] > 0]
    
    X = df_ml[features_numericas]
    y = df_ml['views']
    
    # split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"{AZUL}Entrenando modelos...{ENDC}")
    
    # progress bar para entrenamiento
    with tqdm(total=2, desc="Entrenando modelos") as pbar:
        # random forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        pbar.update(1)
        
        # linear regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        pbar.update(1)
    
    # predicciones
    y_pred_rf = rf_model.predict(X_test)
    y_pred_lr = lr_model.predict(X_test_scaled)
    
    # metricas
    rf_r2 = r2_score(y_test, y_pred_rf)
    lr_r2 = r2_score(y_test, y_pred_lr)
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    lr_mae = mean_absolute_error(y_test, y_pred_lr)
    
    # crear figura con multiples subplots
    fig = plt.figure(figsize=(18, 12))
    
    # subplot 1: random forest
    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(y_test/1e6, y_pred_rf/1e6, alpha=0.5, s=20, 
               c='forestgreen', edgecolors='darkgreen', linewidth=0.5)
    ax1.plot([y_test.min()/1e6, y_test.max()/1e6], 
            [y_test.min()/1e6, y_test.max()/1e6], 
            'r--', linewidth=2, label='Prediccion perfecta')
    ax1.set_xlabel('Vistas reales (Millones)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Vistas predichas (Millones)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Random Forest\nR² = {rf_r2:.3f}, MAE = {rf_mae/1e6:.2f}M', 
                 fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # subplot 2: linear regression
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(y_test/1e6, y_pred_lr/1e6, alpha=0.5, s=20, 
               c='royalblue', edgecolors='darkblue', linewidth=0.5)
    ax2.plot([y_test.min()/1e6, y_test.max()/1e6], 
            [y_test.min()/1e6, y_test.max()/1e6], 
            'r--', linewidth=2, label='Prediccion perfecta')
    ax2.set_xlabel('Vistas reales (Millones)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Vistas predichas (Millones)', fontsize=11, fontweight='bold') 
    ax2.set_title(f'Regresion Lineal\nR² = {lr_r2:.3f}, MAE = {lr_mae/1e6:.2f}M', 
                 fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # subplot 3: feature importance
    ax3 = plt.subplot(2, 2, 3)
    feature_importance = pd.DataFrame({
        'feature': features_numericas,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    bars = ax3.barh(feature_importance['feature'], feature_importance['importance'],
                    color=plt.cm.Oranges(np.linspace(0.4, 0.8, len(feature_importance))))
    ax3.set_xlabel('Importancia', fontsize=11, fontweight='bold')
    ax3.set_title('Importancia de Features (Random Forest)', fontsize=13, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, bar in enumerate(bars):
        ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.3f}', va='center', ha='left', fontsize=10)
    
    # subplot 4: residuales
    ax4 = plt.subplot(2, 2, 4)
    residuales_rf = y_test - y_pred_rf
    ax4.scatter(y_pred_rf/1e6, residuales_rf/1e6, alpha=0.5, s=20, 
               c='purple', edgecolors='lightcoral', linewidth=0.5)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Prediccion (Millones)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Residuales (Millones)', fontsize=11, fontweight='bold')
    ax4.set_title('Analisis de Residuales - Random Forest', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    filename = os.path.join(graficos_dir, '09_prediccion_vistas_modelos.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Grafico guardado en: {MAGENTA}{filename}{ENDC}")
    print("\nResultados de prediccion de vistas:")
    print(f"  Random Forest - R²: {VERDE}{rf_r2:.3f}{ENDC}, MAE: {VERDE}{rf_mae/1e6:.2f}M{ENDC}")
    print(f"  Regresion Lineal - R²: {VERDE}{lr_r2:.3f}{ENDC}, MAE: {VERDE}{lr_mae/1e6:.2f}M{ENDC}")
    print("\nImportancia de features (Random Forest):")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {VERDE}{row['importance']:.3f}{ENDC}")
    
    mejor_modelo = "Random Forest" if rf_r2 > lr_r2 else "Regresion Lineal"
    print(f"\nConclusion: {VERDE}SI es factible predecir vistas{ENDC}")
    print(f"Mejor modelo: {VERDE}{mejor_modelo}{ENDC}")
    print(f"El modelo explica el {VERDE}{max(rf_r2, lr_r2)*100:.1f}%{ENDC} de la variabilidad en las vistas")

def main():
    
    df = cargar_y_limpiar_datos()
    
    preguntas = [
        ("Pregunta 1: Categorias de mayor tendencia", pre1_categorias_tendencia),
        ("Pregunta 2: Categorias que mas/menos gustan", pre2_categorias_mayor_menor_gusto),
        ("Pregunta 3: Mejor ratio likes/dislikes", pre3_mejor_ratio_likes_dislikes),
        ("Pregunta 4: Ratio vistas/comentarios", pre4_ratio_vistas_comentarios),
        ("Pregunta 5: Volumen en el tiempo", pre5_volumen_tendencia_tiempo),
        ("Pregunta 6: Canales mas/menos frecuentes", pre6_canales_mayor_menor_tendencia),
        ("Pregunta 7: Estados con mas interacciones", pre7_estados_vistas_interacciones),
        ("Pregunta 8: Tendencia vs comentarios", pre8_tendencia_vs_comentarios_positivos),
        ("Pregunta 9: Prediccion de metricas", pre9_prediccion_vistas_likes_dislikes)
    ]
    
    with tqdm(total=len(preguntas), desc="Procesando preguntas") as pbar:
        for descripcion, funcion in preguntas:
            funcion(df)
            pbar.update(1)
    
    print(f"{VERDE}Analisis completado exitosamente{ENDC}")
    print(f"Graficos guardados en: {MAGENTA}{graficos_dir}{ENDC}")
    print(f"Datos geograficos en: {MAGENTA}{geo_dir}{ENDC}")
    print(f"{SEPARADOR}")

if __name__ == "__main__":
    main()