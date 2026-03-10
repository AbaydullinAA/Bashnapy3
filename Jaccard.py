# Метрики сходства
# Метод Жаккарда – только текстовые признаки
# Евклидовы расстояния – для числовых метрик
# Расстояния должны быть минимальны, показывает схожесть фильмов
# L=sqrt((x1-x2)^2+(y1-y2)^2)

import pandas as pd
import math
import numpy as np
import os

df_clean = pd.read_csv('./watches_processed.csv')

min_price = df_clean['price'].min()
max_price = df_clean['price'].max()
df_clean['price_norm'] = (df_clean['price'] - min_price) / (max_price - min_price)

min_rew = df_clean['number_of_reviews'].min()
max_rew = df_clean['number_of_reviews'].max()
df_clean['rew_norm'] = (df_clean['number_of_reviews'] - min_rew) / (max_rew - min_rew)

min_rat = df_clean['rating'].min()
max_rat = df_clean['rating'].max()
df_clean['rating_norm'] = (df_clean['rating'] - min_rat) / (max_rat - min_rat)

print("Статистика после нормализации:")
print(df_clean[['rating_norm', 'rew_norm', 'price_norm']].describe())
print("\n")

def jaccard_similarity(list1, list2):
    """Возвращает долю общих элементов в двух списках."""
    set1 = set(list1) if isinstance(list1, list) else set()
    set2 = set(list2) if isinstance(list2, list) else set()
    if len(set1) == 0 and len(set2) == 0:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

def euclidean_distance(row1, row2):
    """Расстояние между двумя фильмами по нормализованным рейтингу и длительности."""
    return math.sqrt(
        (row1['rating_norm'] - row2['rating_norm'])**2 +
        (row1['rew_norm'] - row2['rew_norm'])**2 +
        (row1['price_norm'] - row2['price_norm']) ** 2
    )

MAX_DIST = math.sqrt(3)
def numeric_similarity(row1, row2):
    """Преобразует евклидово расстояние в сходство (1 - нормированное расстояние)."""
    dist = euclidean_distance(row1, row2)
    return 1 - (dist / MAX_DIST)

def combined_similarity(row1, row2, w_numeric=0.6, w_brand_name=0.3, w_watch_name=0.1):
    """
    Взвешенная сумма сходств:
    - числовое (нормализованные рейтинг и длительность)
    - название бренда (Жаккард)
    - название часов (Жаккард)
    """
    sim_num = numeric_similarity(row1, row2)
    sim_brand = jaccard_similarity(row1['brand_name'], row2['brand_name'])
    sim_watch = jaccard_similarity(row1['watch_name'], row2['watch_name'])
    return w_numeric * sim_num + w_brand_name * sim_brand + w_watch_name * sim_watch

# Выберем несколько часов для сравнения (первые 5)
indices = [0, 1, 2, 3, 4]
titles = df_clean.loc[indices, 'watch_name'].tolist()
print("Часы для сравнения:")
for i, idx in enumerate(indices):
    row = df_clean.loc[idx]
    print(f"{i+1}. {row['watch_name']} ( Рейтинг={row['rating']}, Цена = {row['price']}, Количество отзывов = {row['number_of_reviews']}, "
          f"Бренд - производитель={row['brand_name']} )")
print("\n")

# Сравним первые часы с остальными
ref_idx = indices[0]
ref_film = df_clean.loc[ref_idx]
print(f"Референсные часы: {ref_film['watch_name']}")
print("Сходство с другими часами (w_num = 0.6, w_brand_name = 0.3, w_watch_name = 0.1):")
for i in indices[1:]:
    other = df_clean.loc[i]
    sim = combined_similarity(ref_film, other)
    print(f"{other['watch_name']}: {sim:.3f}")