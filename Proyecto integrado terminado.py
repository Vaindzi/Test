#!/usr/bin/env python
# coding: utf-8

# Hola Iván!
# 
# Soy **Patricio Requena** 👋. Es un placer ser el revisor de tu proyecto el día de hoy!
# 
# Revisaré tu proyecto detenidamente con el objetivo de ayudarte a mejorar y perfeccionar tus habilidades. Durante mi revisión, identificaré áreas donde puedas hacer mejoras en tu código, señalando específicamente qué y cómo podrías ajustar para optimizar el rendimiento y la claridad de tu proyecto. Además, es importante para mí destacar los aspectos que has manejado excepcionalmente bien. Reconocer tus fortalezas te ayudará a entender qué técnicas y métodos están funcionando a tu favor y cómo puedes aplicarlos en futuras tareas. 
# 
# _**Recuerda que al final de este notebook encontrarás un comentario general de mi parte**_, empecemos!
# 
# Encontrarás mis comentarios dentro de cajas verdes, amarillas o rojas, ⚠️ **por favor, no muevas, modifiques o borres mis comentarios** ⚠️:
# 
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si todo está perfecto.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si tu código está bien pero se puede mejorar o hay algún detalle que le hace falta.
# </div>
# 
# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si de pronto hace falta algo o existe algún problema con tu código o conclusiones.
# </div>
# 
# Puedes responderme de esta forma:
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class=“tocSkip”></a>
# </div>

# # Proyecto Integrado Sprint 6

# # Paso 1. Abre el archivo de datos y estudia la información general 

# In[1]:


# importar las librerías
import pandas as pd
import numpy as np
from math import factorial
from scipy import stats as st
import math as mt
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind


# In[2]:


# importar el dataset
df = pd.read_csv('/datasets/games.csv')


# In[3]:


# info actual de las primeras 10 filas del df
df.info()


# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Buen trabajo con la carga de datos y la importación de las librerías necesarias, pero te recomiendo hacer la importación de librerías en una celda distinta a la de carga de tus datos
# </div>

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (2da Iteracion)</b> <a class=“tocSkip”></a>
# 
# Buen trabajo con los cambios
# </div>

# In[4]:


# info actual de las primeras 10 filas del df
df.describe()


# **Comentarios acerca de los datos:**
# - Los videojuegos en el conjunto de datos fueron lanzados entre 1980 y 2016, con la mayoría lanzados alrededor de 2006.
# 
# - Las ventas son más altas en América del Norte que en Estados Unidos y Japón, lo que sugiere posibles diferencias en preferencias de juego o estrategias de mercado.
# 
# - Las calificaciones críticas promedio rondan los 69 sobre 100, pero hay una variabilidad significativa, indicando opiniones diversas sobre la calidad de los juegos.

# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
#     
# `.head()` es un muy método para visualizar tus datos en tu exploración inicial, te recomiendo también usar `.describe()` y `.info()` para complementarlo.
# </div>

# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class=“tocSkip”></a>
#     
# Listo! Gracias por la recomendación.
# </div>

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (2da Iteracion)</b> <a class=“tocSkip”></a>
# 
# De nada Iván! Buen trabajo agregando esos dos pasos, de esta forma al inicio de tu proyecto puedes conocer más de tu dataset.
# </div>

# # **Paso 2:** Preparación de los datos

# 2.1 Renombrar las columnas a snake case.

# In[5]:


# Un diccionario vacío para almacenar los nuevos nombres de columnas
new_columns = {}

# Bucle for para cambiar los nombres a minúsculas
for column in df.columns:
    new_columns[column] = column.lower()
    
# aplicar los nuevos nombres al df
df = df.rename(columns=new_columns)
df.columns


# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Hiciste un buen trabajo! Pero escribir los nombres de las columnas de manera manual puede causar algún error más adelante, por lo que te recomiendo ponerlo dentro de un bucle y aplicar `.lower()` a los nombres
# </div>

# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class=“tocSkip”></a>
#     
# Definitivamente quedó mejor con lower y el bucle, menos código y menos rango de error que la lista de nombres de columna manual + una función. Muchas gracias!
# </div>
# 
# 

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (2da Iteracion)</b> <a class=“tocSkip”></a>
# 
# Excelente! Buen trabajo con estos cambios
# </div>

# 2.2 Convertir las columnas a los tipos de datos correctos y sustituir los valores ausentes necesarios:

# In[6]:


# Convertir columna critic_score a float usando np.float32
df['critic_score'] = df['critic_score'].astype(np.float32)

# Sustituir 'tbd' por valores ausentes en user_score
df['user_score'] = df['user_score'].replace('tbd', np.nan)

# Convertir user_score a float usando np.float32
df['user_score'] = df['user_score'].astype(np.float32)

# Mostrar nuevamente el df para ver los cambios en los tipos
df.info()


# 2.3 Exploración y tratamiento de los valores ausentes.

# In[7]:


# Conteo de los valores ausentes en el df
df.isna().sum()


# In[8]:


# Eliminar valores ausentes solo en la columna 'year_of_release'
df = df.dropna(subset=['year_of_release'])

# Eliminar valores ausentes solo en la columna 'critic_score'
df = df.dropna(subset=['critic_score'])

# Eliminar valores ausentes solo en la columna 'user_score'
df = df.dropna(subset=['user_score'])

# Eliminar valores ausentes solo en la columna 'rating'
df = df.dropna(subset=['rating'])

# Mostrar el nuevo conteo de valores ausentes
df.isna().sum()


# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor (2da Iteracion)</b> <a class=“tocSkip”></a>
# 
# Buen trabajo Iván! Para este proyecto el eliminar todas las filas que tienen datos ausentes no afectó tu análisis pero habrán casos donde no puedas eliminarlos por que son tus columnas de interés. 
#     
# Para esos casos dependerá en que tanto tienes de conocimiento de la fuente de datos puesto que muchas veces sabrás la razón del NaN y podrás llenar estos con algún valor en específico, en otros puedes recurrir a llenarlos con la media, mediana, o moda, y en otros cómo aquí sería eliminarlos directamente.
#     
# Cómo te menciono esto dependerá en que tanto sabes de la fuente de esos datos para poder procesarlos, pero eliminarlos nos puede hacer perder mucha información.
# </div>

# In[9]:


df.info()


# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Bien hecho! También podrías utilizar la librería NumPy para esto, por ejemplo, para cambiar el tipo de dato en lugar de float puedes usar `np.float32` y en lugar de None puedes usar `np.nan`
# </div>

# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class=“tocSkip”></a>
#     
# Listo en el punto 2.2! aunque tengo la duda de cuál sería la ventaja de hacerlo con numpy.
# </div>
# 

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (2da Iteracion)</b> <a class=“tocSkip”></a>
# 
# Bien hecho Iván. Te explico la diferencia, la ventaja de usar NumPy para establecer tus tipos de datos es que estos utilizan menos memoria y cuando trabajamos con grandes volúmenes de datos esto ayudará a que el proceso que realices en ellos sea más rápido. Por ejemplo, si tenemos una variable que almacena el valor `0.32` en un float de Python este tendrá un tamaño de 24 bytes pero si lo cambiamos a np.float32 este se reduce a 4 bytes, ¿Ves la diferencia? Entonces cuado hacemos esto en nuestros DataFrames y tenemos una gran cantidad de datos spbre los cuales vamos a ejecutar varias operaciones esto será mucho más rápido.
# </div>

# 2.4 Crear una columna con las ventas totales y cambiar el orden de las columnas para mejorar la legibilidad.

# In[10]:


# sumar las ventas de todas las regiones
df['total_sales'] = df['na_sales'] + df['eu_sales'] + df['jp_sales'] + df['other_sales']

# cambiar el orden de las columnas y comprobar cambios
new_column_order = ['name', 'platform', 'year_of_release', 'genre', 'na_sales', 'eu_sales', 'jp_sales', 'other_sales', 'total_sales', 'critic_score', 'user_score', 'rating']
df = df.reindex(columns=new_column_order)
df.head()


# # **Resumen de la preparación de los datos:**
# 
# **Nombre de columnas y tipos de datos:**
# - Se cambió el nombre de las columnas a snake_case.
# - Se cambió critic_score de object a float (ya que las otras columnas con puntuaciones o número de ventas son float también).
# - Se agregó una columna con las ventas totales de todas las regiones y se cambió el orden de las columnas para facilitar la lectura del df.
# 
# **Valores ausentes:**
# - Se decidió dejar los valores ausentes por las siguiente razones:
#     - La cantidad de valores ausentes es significativa en las columnas critic_score y user_score.
#     - Se reemplazaron los valores con tbd por NaN en la columna user_score
#     - Sustituir estos valores por 0 podría alterar la interpretación de los datos. Al igual que ponerlos como "desconocidos" o "tbd".
#     - Mantener los valores ausentes permite que el análisis refleje mejor la información disponible y ayuda a no generar suposiciones inapropiadas sobre los datos.
#     
# **Suposiciones sobre datos ausentes:**
# - Los juegos que no tienen un critic_score ni un user_score no fueron sometidos a ninguna evaluación.
# - Pudo haber un error en la recopilación de datos de estos juegos.
# - Cuando hay critic_score y "tbd" en user_score, probablemente quiere decir que se cuenta con el dato pero no ha sido cargado.
# 
# 

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Buen resumen, el dejar claro el proceso realizada ayuda a la comprensión de tu trabajo
# </div>

# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Te recomiendo realizar una exploración por valores ausentes y duplicados y que trates de aplicar alguna solución en estos si es posible
# </div>

# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class=“tocSkip”></a>
#     
# Listo en el punto 2.3! Gracias!
# </div>

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (2da Iteracion)</b> <a class=“tocSkip”></a>
# 
# Buen trabajo con los cambios!
# </div>

# # **Paso 3: Análisis de los datos**

# 3. Sacar el total de lanzamientos de videojuegos por años para ver la distribución

# In[11]:


#sacar el total de lanzamientos de videojuegos por año
games_per_year = df.groupby('year_of_release')['name'].count()

# Crear la gráfica de barras para mostrar la distribución
plt.figure(figsize=(8, 4))
games_per_year.plot(kind='bar', color='skyblue')
plt.title('Total de Lanzamientos de videojuegos por año')
plt.xlabel('Año de Lanzamiento')
plt.ylabel('Cantidad de Lanzamientos')
plt.xticks(rotation=65)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# **Comentarios sobre lanzamientos de videojuegos calificados por los usuarios y la crítica por año:**
# - La industria tuvo un crecimiento notorio del año 2000 en adelante
# - A partir de 2009 hubo un nuevo descenso en la tendencia de lanzamiento de juegos

# 3.1 Sacar los años promedio que las plataformas siguen generando ventas (años en que se pierde su popularidad)

# In[12]:


# Sacar el total de años que cada plataforma aparece en el datafrme
platform_years = df.groupby('platform')['year_of_release'].nunique().reset_index()


# In[13]:


# sacar la media de años para calcular cuando una se considera "vieja"
platform_years_mean = platform_years['year_of_release'].mean()
print(f"La media de años que cada plataforma está vigente es: {platform_years_mean:.1f}")


# 3.2 Sacar los años promedio que transcurren para que salga una nueva plataforma.

# In[14]:


# sacar el año de lanzamiento de cada plataforma
platform_launch_year = df.groupby('platform')['year_of_release'].min()
# ordenar datos de los años de menor a mayor
platform_launch_year_sorted = platform_launch_year.sort_values()
# calcular el intervalo de años entre una nueva plataforma y la anterior inmediata
interval_of_launch_years = platform_launch_year_sorted.diff()
# calcular la media de años que transucurren entre lanzamientos
interval_of_launch_years_mean = interval_of_launch_years.mean()
print(f"La media de años que transcurren para nuevos lanzamientos de plataformas es: {interval_of_launch_years_mean}")





# **Nota:**
# En adelante, considero importante sólo tomar en cuenta los datos de 2009 en adelante considerando que las consolas tienen una vida media de vida de 8 años.

# 3.3 Obtener las plataformas más populares en los años más relevantes (últimos  8 años considerando que es el promedio de vigencia de cada plataforma)

# In[15]:


# sumar las ventas totales de cada plataforma
sales_by_platform = df.groupby(['platform', 'year_of_release'])['total_sales'].sum()
sales_by_platform = sales_by_platform.reset_index()
# sacar las 5 plataformas con más ventas en años relevantes
relevant_years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
top_sales_platforms = sales_by_platform[sales_by_platform['year_of_release'].isin(relevant_years)]
top_sales_platforms = top_sales_platforms.groupby('platform')['total_sales'].sum()
top_sales_platforms = top_sales_platforms.sort_values(ascending=False).reset_index().head(5)

# Crear la gráfica de barras
plt.figure(figsize=(4, 3))
plt.bar(top_sales_platforms['platform'], top_sales_platforms['total_sales'], color='skyblue')
plt.title('Top 5 Plataformas con Más Ventas Totales')
plt.xlabel('Plataforma')
plt.ylabel('Ventas Totales')
plt.show()


# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Vas por buen camino! Realizaste los cálculos necesarios pero te recomiendo también agregar gráficas y conclusiones de lo que vas obteniendo en cada paso
# </div>

# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class=“tocSkip”></a>
#     
# Agregué una gráfica para los lanzamientos de videojuegos por año y otra para las 5 plataformas con más ventas, espero sean suficientes :)
# </div>

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (2da Iteracion)</b> <a class=“tocSkip”></a>
# 
# Buen trabajo! las gráficas son de gran ayuda cuando queremos comunicar nuestros resultados
# </div>

# 3.4 Mostrar la distribución de ventas de las plataformas más populares en los últimos 8 años

# In[16]:


#hacer un lista de las plataformas con más ventas
top_platforms = ['PS4', 'PS3', 'X360', 'Wii', '3DS']
#filtrar el df con las plataformas top
df_filtrado = df[df['platform'].isin(top_platforms)]
df_filtrado = df_filtrado[df_filtrado['year_of_release'].isin(relevant_years)]
#hacer un dataframe ordenado de menores a mayores ventas
top_platforms_by_sales = df_filtrado.groupby(['platform', 'year_of_release'])['total_sales'].sum().reset_index()
top_platforms_by_sales = top_platforms_by_sales.sort_values(by='total_sales')


# Crear el gráfico de barras agrupado y configurar tamaño
plt.figure(figsize=(10, 5))
sns.barplot(data=top_platforms_by_sales, x='platform', y='total_sales', hue='year_of_release', palette='viridis')

# Agregar título y etiquetas de los ejes
plt.title('Distribución de Ventas por Plataforma y Año')
plt.xlabel('Plataforma')
plt.ylabel('Ventas Totales (Millonres de dólares EU)')

# Rotar las etiquetas del eje x para mejorar la legibilidad
plt.xticks(rotation=45)

# Mostrar el gráfico
plt.legend(title='Año', bbox_to_anchor=(1, 1))
plt.show()


# **Conclusiones del gráfico de ventas por año y plataforma:**
# - Las plataformas con más ventas fueron el PS3 y el X360
# - El X360, Wii y PS3 son las plataformas con más ventas, aunque el PS3 se mantuvo más estable en cuanto a las mismas más años.
# - Las ventas del Wii fueron las que más decayeron

# 3.5 Sacar las plataformas que eran populares los 3 primeros años del periodo tomado en cuenta.

# In[17]:


# filrar df por años
initial_period_df = df[df['year_of_release'].between(2009, 2013)]
# agrupar por ventas totales
initial_sales_by_platform = initial_period_df.groupby('platform')['total_sales'].sum().sort_values(ascending=False)
# convertir a df y mostrar resultado de las 5 más populares (con más ventas)
initial_sales_df = initial_sales_by_platform.to_frame().reset_index().head(5)
initial_sales_df


# 3.6 Sacar las plataformas que eran populares los últimos 3 años del periodo tomado en cuenta.

# In[18]:


# filrar df por años
later_period_df = df[df['year_of_release'].between(2014, 2016)]
# agrupar por ventas totales
later_sales_by_platform = later_period_df.groupby('platform')['total_sales'].sum().sort_values(ascending=False)
# agrupar por ventas totales
later_sales_df = later_sales_by_platform.to_frame().reset_index().head(5)
later_sales_df


# **Menciona las plataformas que perdieron popularidad y las conclusiones generales hasta ahora:**
# - Las plataformas que perdieron popularidad fueron el X360, Wii y el PS3, que eran las que tuvieron más ventas.
# - La únicas que siguieron siendo populares fueron 3DS y PC.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Buen trabajo con los cálculos y con la conclusión
# </div>

# 3.7 Crea un diagrama de caja para las ventas globales de todos los juegos, desglosados por plataforma.

# In[19]:


#trazar un diagama de cajas para visualizar la distribución de ventas totales por plataforma
sns.boxplot(x = "total_sales", y = "platform", data = top_platforms_by_sales, palette= 'pastel',  saturation = 0.9, linewidth = 1,
           fliersize =8)

plt.xlabel('Ventas totales (Millonres de dólares EU)')
plt.ylabel('Plataforma')
plt.title('Distribución de ventas totales de videojuegos por plataforma de 2009 a 2016')
plt.grid(True)
# mostrar el diagrama de caja
plt.tight_layout()
plt.show()


# **Menciona tus hallazgos sobre la distribución ventas promedio de cada una de las plataformas seleccionadas:**
# - El X360, el Wii y el PS3 tienen las mayores ventas, pero la mediana de ventas del PS3 y del X360 son mayores a las del Wii, lo que implica que se mantuvieron más estables.
# - El X360 tuvo ventas atípicas un año en específico.
# - El 3DS y el PS3 son los que tuvieron menos ventas mínimas en el periodo

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Genial! Presentaste el gráfico correcto y obtuviste una conclusión acertada
# </div>

# 3.9 Gráfica de dispersión para calcular relación entre reseñas de usuarios y ventas.

# In[20]:


df_x360 = df[df['platform'] == 'X360']

df_x360_filtered = df_x360[['user_score', 'total_sales']].dropna()

correlation = df_x360_filtered['user_score'].corr(df_x360_filtered['total_sales'])
print(f'la correlación entre las reseñas de usuario y las ventas totales de la pltaforma X360 es de: {correlation:.2f}.')


# In[21]:


plt.figure(figsize=(7, 4))
plt.scatter(df_x360_filtered['user_score'], df_x360_filtered['total_sales'], alpha=0.5)

# Agregar etiquetas y título
plt.title('Relación entre reseñas de usuario y ventas para la plataforma X360')
plt.xlabel('Puntuación de usuario')
plt.ylabel('Ventas totales (Millonres de dólares EU)')

# Mostrar el gráfico
plt.tight_layout()
plt.show()


# **Conclusión sobre la relación entre reseñas de usuario y ventas totales de X360:**
# No existe una relación fuerte entre las ventas y las reseñas.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Vas por buen camino! Mostraste el gráfico de dispersión entre puntuación de usuario y ventas de manera correcta

# 3.10 Distribución de ventas de videojuegos por género:

# In[22]:


# Sacar ventas totales de videojuegois ordenadas por género
sales_by_genre = df.groupby('genre')['total_sales'].sum().reset_index()
# Ordenar de mayor a menor
sales_by_genre = sales_by_genre.sort_values(by='total_sales', ascending=False)
# Resetear el indice
sales_by_genre.reset_index(drop=True, inplace=True)
sales_by_genre


# 3.11 Crear gráfico de ventas de vidoejuegos por género

# In[23]:


# Crear gráfico de distribución de ventas de videojuegos por género
sns.barplot(data=sales_by_genre, x='genre', y='total_sales', palette='viridis')

# Agregar título y etiquetas de los ejes
plt.title('Distribución de ventas de videojuegos por género (1980-2016)')
plt.xlabel('Plataforma')
plt.ylabel('Ventas Totales (millonres de dólares EU)')

# Rotar las etiquetas del eje x para mejorar la legibilidad
plt.xticks(rotation=65)

# Mostrar el gráfico
plt.show()


# **Conclusiones sobre las ventas de videojuegos por género:**
# - Los juegos de acción encabezan la industria
# - Los juegos de deportes y shooters son el segundo lugar y tercer lugar respectivamente, pero sólo representan el ~70% de los juegos de acción
# - Los juegos de rol, racing y misc tienen ventas similares
# - Los juegos "puzzle", aventura y estrategia son los que menos ventas tienen.
# 
# Tomando en cuenta esta información, sí es fácil generalizar sobre los géneros que encabezan la industria.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Buen trabajo mostrando tus resultados por género
# </div>

# 3.12 Ventas de juegos multiplataforma en las plataformas principales:

# In[24]:


# Sacar ventas totales por género en cada plataforma
sales_by_genre_and_platform = df.groupby(['genre', 'platform'])['total_sales'].sum().reset_index()
# Sacar ventas totales en las mejores plataformas
sales_by_genre_and_platform = sales_by_genre_and_platform[sales_by_genre_and_platform['platform'].isin(top_platforms)]
# Ordenar datos del género con más ventas al menor
sales_by_genre_and_platform = sales_by_genre_and_platform.sort_values(by='total_sales', ascending=False)

# Crear gráfico de distribución de ventas de videojuegos por género y plataforma
plt.figure(figsize=(14,8))
sns.barplot(data=sales_by_genre_and_platform, x='genre', y='total_sales', hue='platform', palette='tab10')
# Agregar título y etiquetas de los ejes
plt.title('Ventas por género y plataforma en las mejores consolas')
plt.xlabel('Género')
plt.ylabel('Ventas Totales (millonres de dólares EU)')
# Rotar las etiquetas del eje x para mejorar la legibilidad y mostrar el gráfico
plt.xticks(rotation=45)
plt.legend(title='Plataforma')
plt.show()





# **Conclusiones principales de los géneros de videojuegos en las diferentes plataformas:**
# - Los juegos de acción, que es el género con más ventas se vende más en PS3
# - Los shooters se venden más en X360
# - En juegos de deportes y misc y plataforma, Wii es la plataforma que más vende
# - Los shooters se venden más en X360
# 

# # Paso 4. Crea un perfil de usuario para cada región

# Para cada región (NA, EU, JP) determina:
# 
# - Las cinco plataformas principales. 
# - Describe las variaciones en sus cuotas de mercado de una región a otra.
# - Los cinco géneros principales. Explica la diferencia.
# - Si las clasificaciones de ESRB afectan a las ventas en regiones individuales.

# 4.1 Sacar las 5 consolas principales en Norte América:

# In[25]:


# sacar ventas en las plataformas principales en Norte América
top_platforms_na = df.groupby('platform')['na_sales'].sum().reset_index()
# Agrupar de menor a menor
top_platforms_na = top_platforms_na.sort_values(by='na_sales', ascending=False)
top_platforms_na.reset_index(drop=True, inplace=True)
# Resetear el indice y hacer que aparezca desde 1
top_platforms_na.index += 1
# Imprimir ls 5 plataformas principales
top_platforms_na.head(5)


# 4.2 Sacar las 5 consolas principales en EU:

# In[26]:


# sacar ventas en las plataformas principales en EU:
top_platforms_eu = df.groupby('platform')['eu_sales'].sum().reset_index()
# Agrupar de menor a menor
top_platforms_eu = top_platforms_eu.sort_values(by='eu_sales', ascending=False)
top_platforms_eu.reset_index(drop=True, inplace=True)
# Resetear el indice y hacer que aparezca desde 1
top_platforms_eu.index += 1
# Imprimir ls 5 plataformas principales
top_platforms_eu.head(5)


# 4.3 Sacar las 5 consolas principales en Japón:

# In[27]:


# sacar ventas en las plataformas principales en Japón:
top_platforms_jp = df.groupby('platform')['jp_sales'].sum().reset_index()
# Agrupar de menor a menor
top_platforms_jp = top_platforms_jp.sort_values(by='jp_sales', ascending=False)
top_platforms_jp.reset_index(drop=True, inplace=True)
# Resetear el indice y hacer que aparezca desde 1
top_platforms_jp.index += 1
# Imprimir ls 5 plataformas principales
top_platforms_jp.head(5)


# 4.4 Unir los dataframes de NA,EU y JP.
# 
# 

# In[28]:


# Hacer una lista con las plataformas que aparecen en el top 5 de cada región
platforms_in_dfs = ['X360', 'PS2', 'Wii', 'PS3', 'DS', 'PS', 'SNES', '3DS']
# Unir NA con EU
merged_df = top_platforms_na.merge(top_platforms_eu, on=['platform'], how='outer')
# Unir NA y EU con JP
merged_df_2 = merged_df.merge(top_platforms_jp, on=['platform'], how='outer')
# Filtrar el dataframe para que sólo se consideren el top 5 de plataformas de cada región
merged_df_filtered = merged_df_2[merged_df_2['platform'].isin(platforms_in_dfs)]
merged_df_filtered

                      


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Excelente! Lo hiciste muy bien, y el DataFrame resultante es el correcto ya que muestras todo en una sola tabla
# </div>

# 4.5 Hacer gráfico para ver las diferencias en ventas de las plataformas en cada región.

# In[29]:


# Hacer un gráfico del top 5 plataformas y sus ventas por region para ver diferencia
plt.bar(merged_df_filtered['platform'], merged_df_filtered['na_sales'], color='blue', label='North America')
plt.bar(merged_df_filtered['platform'], merged_df_filtered['eu_sales'], color='green', label='USA', alpha=0.7)
plt.bar(merged_df_filtered['platform'], merged_df_filtered['jp_sales'], color='red', label='Japan', alpha=0.5)

# Configurar etiquetas y leyenda
plt.xlabel('Platforma')
plt.ylabel('Ventas totales (millonres de dólares USA)')
plt.title('Ventas de videojuegos del top 5 de plataformas en Norte América, USA y Japón')
plt.xticks(rotation=45)
plt.legend()

# Mostrar el gráfico
plt.tight_layout()
plt.show()


# 4.6 Hacer gráfico de las ventas totales por región

# In[30]:


# Configurar el gráfico
plt.figure(figsize=(6, 4))

# Obtener las ventas totales por región
sales_df = df[['year_of_release', 'na_sales', 'eu_sales', 'jp_sales']]
# Obtener las ventas totales por región en años relevantes
total_sales_in_relevant_years = sales_df[sales_df['year_of_release'].isin(relevant_years)][['na_sales', 'eu_sales', 'jp_sales']].sum()

# Crear gráfico de distribución de ventas de videojuegos por región
plt.bar(x=['Norte América', 'EU', 'Japón'], height=total_sales_in_relevant_years, color=['green', 'red', 'blue'])

# Configurar etiquetas y título
plt.xlabel('Region')
plt.ylabel('Ventas totales (millonres de dólares EU)')
plt.title('Distribución de ventas totales de videojuegos por región de 2010 a 2016')

# Mostrar el gráfico
plt.tight_layout()
plt.show()


# **Conlusiones sobre las ventas de plataformas y ventas totales por región:**
# 
# **Popularidad en plataformas**
# - Norté América y USA tienen básicamente las mismas plataformas como las más populares.
# - 3DS y DS son populares en Japón.
# 
# **Ventas totales:**
# - Norte América vende más videouegos en la mayoría de plataformas y es la región con más ventas en general.
# 

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Tus visualizaciones son muy claras y las conclusiones ayudan a enter lo que estás mostrando, bien hecho!
# </div>

# 4.7 Sacar los géneros principales por región.

# In[31]:


# sacar ventas en las plataformas principales en Norte América
top_genres_na = df.groupby('genre')['na_sales'].sum().reset_index()
# Agrupar de menor a menor
top_genres_na = top_genres_na.sort_values(by='na_sales', ascending=False)
top_genres_na.reset_index(drop=True, inplace=True)
# Resetear el indice y hacer que aparezca desde 1
top_genres_na.index += 1
# Imprimir ls 5 plataformas principales
top_genres_na.head(5)


# In[32]:


# sacar ventas en las plataformas principales en Norte América
top_genres_eu = df.groupby('genre')['eu_sales'].sum().reset_index()
# Agrupar de menor a menor
top_genres_eu = top_genres_eu.sort_values(by='eu_sales', ascending=False)
top_genres_eu.reset_index(drop=True, inplace=True)
# Resetear el indice y hacer que aparezca desde 1
top_genres_eu.index += 1
# Imprimir ls 5 plataformas principales
top_genres_eu.head(5)


# In[33]:


# sacar ventas en las plataformas principales en Norte América
top_genres_jp = df.groupby('genre')['jp_sales'].sum().reset_index()
# Agrupar de menor a menor
top_genres_jp = top_genres_jp.sort_values(by='jp_sales', ascending=False)
top_genres_jp.reset_index(drop=True, inplace=True)
# Resetear el indice y hacer que aparezca desde 1
top_genres_jp.index += 1
# Imprimir ls 5 plataformas principales
top_genres_jp.head(5)


# 4.8 Unir los dataframes de ventas por género de NA, EU y JP.

# In[34]:


# Hacer una lista con los géneros que aparecen en el top 5 de cada región
genres_in_dfs = ['Action', 'Sports', 'Shooter', 'Platform', 'Misc', 'Racing', 'Role-Playing']
# Unir NA con EU
merged_df_genres = top_genres_na.merge(top_genres_eu, on=['genre'], how='outer')
# Unir NA y EU con JP
merged_df_genres_2 = merged_df_genres.merge(top_genres_jp, on=['genre'], how='outer')
# Filtrar el dataframe para que sólo se consideren el top 5 de plataformas de cada región
merged_genres_filtered = merged_df_genres_2[merged_df_genres_2['genre'].isin(genres_in_dfs)]
merged_genres_filtered


# 4.9 Hacer gráfico para ver las diferencias en ventas por género por región.

# In[35]:


# Hacer un gráfico del top 5 plataformas y sus ventas por region para ver diferencia
plt.bar(merged_genres_filtered['genre'], merged_genres_filtered['na_sales'], color='blue', label='North America')
plt.bar(merged_genres_filtered['genre'], merged_genres_filtered['eu_sales'], color='green', label='EU Sales', alpha=0.7)
plt.bar(merged_genres_filtered['genre'], merged_genres_filtered['jp_sales'], color='red', label='Japan Sales', alpha=0.5)

# Configurar etiquetas y leyenda
plt.xlabel('Género')
plt.ylabel('Ventas totales (millonres de dólares EU)')
plt.title('Top Géneros de Videojuegos: Norte América, EU y Japón')
plt.xticks(rotation=45)
plt.legend()

# Mostrar el gráfico
plt.tight_layout()
plt.show()


# 4.10 Si las clasificaciones de ESRB afectan a las ventas en regiones individuales.

# In[36]:


# 1. Filtrar los datos relevantes
relevant_data = df[['critic_score', 'na_sales', 'eu_sales', 'jp_sales']]

# 2. Limpiar los datos
clean_data = relevant_data.dropna()

# 3. Análisis de correlación
correlation_na = clean_data['critic_score'].corr(clean_data['na_sales'])
correlation_eu = clean_data['critic_score'].corr(clean_data['eu_sales'])
correlation_jp = clean_data['critic_score'].corr(clean_data['jp_sales'])

print(f"Correlación con NA Sales: {correlation_na:.2f}")
print(f"Correlación con EU Sales: {correlation_eu:.2f}")
print(f"Correlación con JP Sales: {correlation_jp:.2f}")


# In[37]:


# Crear un gráfico de dispersión que muestre la correlación entre critic_score y ventas totales por región
plt.scatter(clean_data['critic_score'], clean_data['na_sales'], label='NA Sales', alpha=0.5)
plt.scatter(clean_data['critic_score'], clean_data['eu_sales'], label='USA Sales', alpha=0.5)
plt.scatter(clean_data['critic_score'], clean_data['jp_sales'], label='JP Sales', alpha=0.5)
plt.xlabel('Puntación de los críticos')
plt.ylabel('Ventas totales (millones de dólares EU)')
plt.title('Correlación entre la puntuación de la crítica y ventas totales en NA, USA y JP')
plt.legend()
plt.tight_layout()
plt.show()


# **Conlusiones sobre los géneros más populares y la relación crítica - ventas totales en NA, EU y JP:**
# 
# **Géneros:**
# - Los géneros de acción, deportes y disparos son los más populares en NA y USA. Por el contrario, el género menos popular son los de juegos de rol, sin embargo, los juegos de rol son el género más popular en Japón.
# 
# **Relación crítica y ventas totales:**
# - La correlación entre la puntuación de los críticos y las ventas es baja en las regiones deNorte América, Estados Unidos y Japón. Esto sugiere que, aunque existe alguna relación entre la calidad percibida de un juego y sus ventas, otros factores pueden tener un impacto más significativo en las decisiones de compra de los consumidores en estos mercados.
# - Norte América es donde existe más correlación entre crítica y ventas totales.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Bien hecho Iván! Con esto complementas muy bien tu análisis
# </div>

# # Paso 5. Prueba las siguientes hipótesis:
# 
# — Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son las mismas.
# 
# — Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes.
# 
# Establece tu mismo el valor de umbral alfa.

# 5.1 Calcular las medias de puntuaciones de usuario de X0ne y PC y su distribución:

# In[40]:


# Calcular la media de user_score para X0ne
ratings_xone = df[df['platform'] == 'XOne']['user_score'].reset_index()
ratings_xone = ratings_xone.dropna(subset=['user_score'])

# Calcular la media de user_score para PC
ratings_pc = df[df['platform'] == 'PC']['user_score'].reset_index()
ratings_pc = ratings_pc.dropna(subset=['user_score'])

media_xone = ratings_xone.mean()

# Calcular la media de user_score para la plataforma PC
media_pc = ratings_pc.mean()
            
print("La media de puntuaciones de usuario para XOne es: {:.2f}".format(media_xone[1]))
print("La media de puntuaciones de usuario para PC es: {:.2f}".format(media_pc[1]))


# In[41]:


# Histograma para comparar visualmente la distribución de las puntuaciones de usuario en las plataformas X0ne y PC
plt.figure(figsize=(6, 4))

sns.kdeplot(data=ratings_xone['user_score'], label='XOne', shade=True)
sns.kdeplot(data=ratings_pc['user_score'], label='PC', shade=True)

# Agregar etiquetas y título
plt.xlabel('Puntuación de usuario')
plt.ylabel('Densidad')
plt.title('Histograma de puntuaciones de usuario en X0ne y PC')
plt.legend()

# Mostrar el histograma
plt.show()


# 5.2 Calcular las varianzas en puntuaciones de usuario de X0ne y PC.

# In[42]:


# Calcular la varianza para XOne
media_xone_var = np.var(ratings_xone)
print("La varianza de puntuaciones promedio de usuario de X0ne es: {:.2f}".format(media_xone_var[1]))


# Calcular la varianza para PC
media_pc_var = np.var(ratings_pc)
print('La varianza de puntuaciones promedio de usuario de PC es: {:.2f}'.format(media_pc_var[1]))


# 5.3 Aplicar prueba estadística y tomar una decisión.

# In[48]:


# Realizar la prueba t de Student para muestras independientes
t_statistic, p_value = stats.ttest_ind(ratings_xone['user_score'], ratings_pc['user_score'], equal_var=False)

# Imprimir los resultados
print("Valor estadístico t:", t_statistic)
print("Valor p:", p_value)

# Comparar el valor p con el valor alfa y tomar una decisión
alpha = 0.05
if p_value < alpha:
    print("Rechazar la hipótesis nula. Las calificaciones promedio de usuarios de XOne son diferentes a las de PC")
else:
    print("No se puede rechazar la hipótesis nula. No hay suficiente evidencia para afirmar que las calificaciones promedio de las plataformas XOne y PC son diferentes.")


# **— ¿Cómo se formuló las hipótesis nula y alternativa?**
# - La hipótesis nula prueba que las calificaciones entre plataformas son iguales.
# - La hipótesis alternativa probaría una desaigualdad.
# 
# Siendo la hipótesis: — Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son las mismas...La hipótesis nula no se puede rechazar.
# 
# **— Qué criterio utilizaste para probar las hipótesis y por qué?**
# 
# **R:** t de student
# - Estamos comparando las medias de dos muestras independientes: en este caso, las calificaciones promedio de los géneros de acción y deportes provienen de dos muestras diferentes y no están relacionadas entre sí.
# - Las calificaciones promedio son datos continuos: las calificaciones de usuario, al ser una escala numérica del 1 al 10, y esto cumple con el requisito de la prueba t de student.

# 5.4 Calcular las puntuaciones de usuario promedio y su distribución de los géneros de acción y deportes.

# In[45]:


# Calcular la media de user_score para X0ne
ratings_action = df[df['genre'] == 'Action']['user_score'].reset_index()
ratings_action = ratings_action.dropna(subset=['user_score'])

# Calcular la media de user_score para PC
ratings_sports = df[df['genre'] == 'Sports']['user_score'].reset_index()
ratings_sports = ratings_sports.dropna(subset=['user_score'])

media_action = ratings_action.mean()

# Calcular la media de user_score para la plataforma PC
media_sports = ratings_sports.mean()
            
print("La media de puntuaciones de usuario para el género de Acción es: {:.2f}".format(media_action[1]))
print("La media de puntuaciones de usuario para el género de Deportes es: {:.2f}".format(media_sports[1]))


# In[53]:


# Histograma para comparar visualmente la distribución de las puntuaciones de usuario en los géneros acción y deportes
plt.figure(figsize=(6, 4))

sns.kdeplot(data=ratings_action['user_score'], label='Acción', shade=True)
sns.kdeplot(data=ratings_sports['user_score'], label='Deportes', shade=True)

# Agregar etiquetas y título
plt.xlabel('Puntuación de usuario')
plt.ylabel('Densidad de probabilidad')
plt.title('Histograma de puntuaciones de usuario en videojuegos de acción y deportes')
plt.legend()

# Mostrar el histograma
plt.show()


# In[51]:


# Calcular la varianza para XOne
media_action_var = np.var(ratings_action)
print("La varianza de puntuaciones promedio de usuario para el género de Acción es: {:.2f}".format(media_action_var[1]))


# Calcular la varianza para PC
media_sports_var = np.var(ratings_sports)
print('La varianza de puntuaciones promedio de usuario para el género de Deportes es: {:.2f}'.format(media_sports_var[1]))


# In[54]:


# Realizar la prueba t de Student para muestras independientes
t_statistic, p_value = stats.ttest_ind(ratings_action['user_score'], ratings_sports['user_score'])

# Imprimir los resultados
print("Valor estadístico t:", t_statistic)
print("Valor p:", p_value)

# Comparar el valor p con el valor alfa y tomar una decisión
alpha = 0.05
if p_value < alpha:
    print("Rechazar la hipótesis nula. Las calificaciones promedio de usuarios en el género de acción son diferentes a las de deportes")
else:
    print("No se puede rechazar la hipótesis nula. No hay suficiente evidencia para afirmar que las calificaciones promedio en los géneros de acción y deportes son diferentes")


# **Conclusión:**
#  - Siendo la hipótesis: — Las calificaciones promedio de los usuarios para los géneros de acción y deportes son las mismas...La hipótesis nula no se puede rechazar.
#  
# **— Qué criterio utilizaste para probar las hipótesis y por qué?**
# 
# **R:** t de student
# - Estamos comparando las medias de dos muestras independientes: en este caso, las calificaciones promedio de los géneros de acción y deportes provienen de dos muestras diferentes y no están relacionadas entre sí.
# - Las calificaciones promedio son datos continuos: las calificaciones de usuario, al ser una escala numérica del 1 al 10, y esto cumple con el requisito de la prueba t de student.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Hiciste una buena aplicación de las pruebas de hipótesis y respondiste las preguntas en base a esto de manera adecuado, buen trabajo!
# </div>

# # Paso 6. Escribe una conclusión general

# **Conclusiones finales del proyecto:**
# - La industria de los videojuegos experimentó un crecimiento notorio de 2000 a 2008, seguido de un declive a partir de 2009. Se observa que nace una nueva plataforma cada 1.7 años y que cada plataforma dura en promedio 8 años teniendo ventas.
# - Las plataformas líderes en ventas son X360 y PS3.
# - Los géneros de acción, deportes y isparos son los más populares en América del Norte y Europa, mientras que los juegos de rol lideran en Japón.
# - Los juegos de acción se venden más en PS3. Los shoorters se venden más en X360 y los juegos de deportes se venden más en Wii
# - Los juegos de acertijos, aventura y estrategia son los que menos ventas tienen.En juegos de deportes y misc y plataforma, Wii es la plataforma que más vende
# - La correlación entre la puntuación de los críticos y las ventas es baja en general, sugiriendo que otros factores pueden influir más en las decisiones de compra. Sin embargo, en América del Norte existe una correlación más significativa entre crítica y ventas totales, aunque sigue sin ser un factor crítico para las ventas.
# - No podemos estar seguros de que las calificaciones promedio de los usuarios para las plataformas Xbox One y PC sean exactamente iguales.
# - No hay suficiente evidencia para decir que las calificaciones promedio de los juegos de acción y deportes son diferentes.

# <div class="alert alert-block alert-info">
# <b>Comentario general (1ra Iteracion)</b> <a class=“tocSkip”></a>
#     
# 
# Iván se nota tu manejo de las diferentes librerías para llevar a cabo tu proyecto y debo destacar lo ordenado que has presentado tu proyecto, te felicito!
# <br>
# <br>
# Generaste visualizaciones muy buenas y a partir de estas sacaste buenas conclusiones que complementaron tu análisis a la perfección.
# <br>
# <br>
# Son pequeñas cosas que hay que corregir por lo que estoy seguro para tu siguiente iteración habrás completado tu proyecto.
# <br>
# <br>
# Te he dejado mis comentarios a lo largo del proyecto y espero que te sirvan de ayuda para la próxima iteración.
# Un saludo! 🦾
# </div>

# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class=“tocSkip”></a>
#     
# Gracias por los comments, sí cambiaron un poquito algunos datos tras procesar los valores ausentes desde el inicio, tuve que hacer algunas modificaciones tras ello en los siguientes pasos, pero cambiaron muy poco las conclusiones generales!
#     
# Sólo te dejo 2 dudas finales:
# 1. Vi que al tratar los valores ausentes se recortó mucho el DF ¿no estaríamos perdiendo data relevante que podría ayudar a responder otras dudas si los dejamos? Lo pregunto porque aunque en este caso no cambió mucho la conclusión, creo que al tratar los valores ausentes los datos se basaron más en ventas de videojuegos calificados, y no ventas de videojuegos en general.
# 
# 2. ¿Qué diferencia hay entre hacer el cambio de tipo de datos a float, etc con Numpy a con Python?
#     
# Muchas gracias por revisar mi proyecto y por los comentarios. Un saludo :)!
# </div>

# <div class="alert alert-block alert-info">
# <b>Comentario general (2da Iteracion)</b> <a class=“tocSkip”></a>
#     
# Muy buen trabajo Iván! Adaptaste tu código muy bien luego de mis comentarios.
# <br>
# <br>
# En respuesta a tus dudas:
# - Sobre tu duda en Numpy te he dejado una explicación justo debajo de tu corrección que espero te sirva en próximos proyectos.
# - Respecto al tratamiento de los valores ausentes igual te dejé un comentario en ese punto, pero si, tienes razón en cuanto a que se redujo mucho tu dataset pero aún así no afectó tus análisis. Cuando nos encontramos con este tipo de casos podemos quitar todas las filas con NaN, o reemplazarlo si conocemos la razón de esos NaN por ejemplo si tenemos una columna relacionada a contar algo y tenemos NaN eso se debe a que el contéo es 0 y podemos llenar esas filas con 0 usando `fillna()`, también podemos aplicar técnicas de Data Imputation para llenar estos datos con la media, moda, o mediana pero esto dependerá de que tan bien conozcas la fuente de los datos.
# <br>
# <br>
# Saludos! 🦾
# </div>
