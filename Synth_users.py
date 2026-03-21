import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import datetime

df = pd.read_csv(r"./watches_processed.csv")
print(f"Загружено {len(df)} часов")

print("Столбцы в файле:", df.columns.tolist())

user_types = [
    {
        'name': 'fan_of_rich_brands',
        'like_brands': ['Swiss Military Hanowa', 'BRETLEY', 'Versace', 'TISSOT', 'Seiko'],
        'dislike_brands': ['Matrix', 'GENERIC', 'SELLORIA', 'Acnos', 'Talgo'],
        'bias': -0.2,
        'noise': 0.3,
        'count': 75
    },
    {
        'name': 'fan_of_middle_brands',
        'like_brands': ['Casio', 'Tommy Hilfiger', 'Fastrack', 'Titan', 'Fossil'],
        'dislike_brands': ['TISSOT', 'Seiko', 'Matrix', 'SELLORIA', 'Talgo'],
        'bias': 0.2,
        'noise': 0.2,
        'count': 100
    },
    {
        'name': 'fan_of_cheap_brands',
        'like_brands': ['Matrix', 'GENERIC', 'SELLORIA', 'Acnos', 'Talgo'],
        'dislike_brands': ['Swiss Military Hanowa', 'BRETLEY', 'Versace', 'TISSOT', 'Seiko'],
        'bias': 0.3,
        'noise': 0.3,
        'count': 150
    },
    {
        'name': 'usual_buyer',
        'like_brands': [],
        'dislike_brands': [],
        'bias': 0.0,
        'noise': 0.8,
        'count': 200
    },
    {
        'name': 'critic',
        'like_brands': [],
        'dislike_brands': [],
        'bias': -0.5,
        'noise': 0.2,
        'count': 65
    },
]


def calculate_rating(brand_name, user_profile):
    """
    brand_name: название бренда (строка)
    user_profile: профиль пользователя
    возвращает: оценку от 1 до 5
    """
    if not brand_name:
        return round(random.uniform(2.5, 4.5), 1)

    if isinstance(brand_name, str):
        brand_list = [brand_name]
    else:
        brand_list = brand_name

    like_overlap = 0
    if user_profile['like_brands']:
        like_overlap = len(set(brand_list).intersection(set(user_profile['like_brands'])))

    dislike_overlap = 0
    if user_profile['dislike_brands']:
        dislike_overlap = len(set(brand_list).intersection(set(user_profile['dislike_brands'])))

    base_rating = 2.5
    like_boost = like_overlap * 0.8
    dislike_penalty = dislike_overlap * 0.7

    rating = base_rating + like_boost - dislike_penalty + user_profile['bias']

    noise = random.uniform(-user_profile['noise'], user_profile['noise'])
    rating += noise

    rating = min(5.0, rating)
    rating = max(1.0, rating)
    return round(rating, 1)


ratings_data = []
user_id_counter = 1

print("Генерация пользователей и оценок...")
for user_type in user_types:
    print(f"  Создаём {user_type['count']} пользователей типа '{user_type['name']}'...")
    for _ in tqdm(range(user_type['count'])):
        num_ratings = random.randint(30, 150)
        sampled_watches = df.sample(n=min(num_ratings, len(df)))
        for idx, watch in sampled_watches.iterrows():
            brand_name = watch['brand_name']
            rating = calculate_rating(brand_name, user_type)
            td = random.random() * datetime.timedelta(days=1)
            td = str(td)
            ratings_data.append({
                'user_id': user_id_counter,
                'watch_name': watch['watch_name'],
                'rating': rating,
                'timestamp': td,
                'user_type': user_type['name']
            })
        user_id_counter += 1

print(f"\nСгенерировано {user_id_counter - 1} пользователей")
print(f"Всего оценок: {len(ratings_data)}")

ratings_df = pd.DataFrame(ratings_data)
ratings_df.to_csv('synthetic_ratings.csv', index=False)

print("\nСохранено в synthetic_ratings.csv")

print("\n=== СТАТИСТИКА ===")
print(f"Уникальных пользователей: {ratings_df['user_id'].nunique()}")
print(f"Уникальных часов: {ratings_df['watch_name'].nunique()}")
print(f"Всего оценок: {len(ratings_df)}")
print(f"Средняя оценка: {ratings_df['rating'].mean():.2f}")

print("\nРаспределение по типам пользователей:")
print(ratings_df.groupby('user_type')['user_id'].nunique())

print("\nПервые 20 записей:")
print(ratings_df.head(20))
print("\nПоследние 20 записей:")
print(ratings_df.tail(20))