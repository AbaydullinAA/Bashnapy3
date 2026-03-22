import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.metrics.pairwise import cosine_similarity

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

avg_price = df_clean['price'].mean()
print(f"Средняя цена: {avg_price:.2f}")
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
    """Расстояние между двумя часами по нормализованным рейтингу и длительности."""
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

def combined_similarity(row1, row2, w_numeric=0.6, w_brand_name=0.25, w_watch_name=0.15):
    """
    Взвешенная сумма сходств:
    - числовое (нормализованные рейтинг и длительность)
    - название бренда (Жаккард)
    - название часов (Жаккард)
    """
    sim_num = numeric_similarity(row1, row2)
    sim_brand = jaccard_similarity([row1['brand_name']], [row2['brand_name']])
    sim_watch = jaccard_similarity(row1['watch_name'], row2['watch_name'])
    return w_numeric * sim_num + w_brand_name * sim_brand + w_watch_name * sim_watch

if not os.path.exists('watches_heatmap.png'):
    print("Строим тепловую карту для первых 50 часов...")
    sample_df = df_clean.sample(50, random_state=42).copy()
    n = len(sample_df)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        sim_matrix[i][i] = 1
        for j in range(i + 1, n):
            sim_matrix[i][j] = sim_matrix[j][i] = combined_similarity(sample_df.iloc[i], sample_df.iloc[j])
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_matrix, annot=False, xticklabels=sample_df['brand_name'], yticklabels=sample_df['brand_name'], cmap='coolwarm')
    plt.title('Матрица сходства первых 50 часов (Жаккард + Евклид)')
    plt.tight_layout()
    plt.savefig('watches_heatmap.png', dpi=150)
    plt.close()
    print("Тепловая карта сохранена как watches_heatmap.png\n")
else:
    print("Тепловая карта часов уже есть.\n")

ratings = pd.read_csv('synthetic_ratings.csv')
print(f"Загружено {len(ratings)} оценок от {ratings['user_id'].nunique()} пользователей")

user_item_matrix = ratings.pivot_table(
    index='user_id',
    columns='watch_name',
    values='rating'
).fillna(0)
all_watches = user_item_matrix.columns.tolist()

user_means = user_item_matrix.mean(axis=1)
centered_matrix = user_item_matrix.sub(user_means, axis=0)
user_type_map = ratings.groupby('user_id')['user_type'].first().to_dict()
user_types_list = [user_type_map.get(uid, 'unknown') for uid in user_item_matrix.index]

def get_collab_recommendations(user_ratings, top_k=10, top_n=5):
    user_vector = pd.Series(index=all_watches, dtype=float).fillna(0)
    for title, rating in user_ratings.items():
        if title in user_vector.index:
            user_vector[title] = rating
        else:
            print(f"Предупреждение: Часы '{title}' не найдены в датасете")

    user_mean = user_vector[user_vector > 0].mean()
    if np.isnan(user_mean):
        user_mean = 0
    user_centered = user_vector - user_mean

    # Сходство со всеми синтетиками
    sim_scores = cosine_similarity([user_centered], centered_matrix)[0]

    # Поиск топ-K похожих
    similar_users_idx = np.argsort(sim_scores)[::-1][:top_k]
    sim_values = sim_scores[similar_users_idx]

    # Сбор кандидатов от похожих пользователей
    candidates = {}
    for idx, sim in zip(similar_users_idx, sim_values):
        user = user_item_matrix.index[idx]
        user_ratings_raw = user_item_matrix.loc[user]
        for watch, rating in user_ratings_raw[user_ratings_raw >= 4].items():
            if watch not in user_ratings and watch not in candidates:
                candidates[watch] = 0
            if watch not in user_ratings:
                candidates[watch] += rating * sim

    if not candidates:
        return [], sim_scores
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return sorted_candidates[:top_n], sim_scores


def get_content_recommendations(movie_title, top_n=5):
    matches = df_clean[df_clean['title'].str.lower().str.contains(movie_title.lower(), na=False)]
    if len(matches) == 0:
        return None, f"Фильм '{movie_title}' не найден."

    if len(matches) > 1:
        print("\nНайдено несколько вариантов:")
        for i, row in enumerate(matches.itertuples(), 1):
            print(f"  {i}. {row.title}")
        print("0 — ввести название заново")

        while True:
            choice = input("Введите номер или точное название: ").strip()
            if choice == '0':
                return None, "Попробуйте ввести название заново."
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(matches):
                    target = matches.iloc[idx]
                    break
                else:
                    print("Неверный номер. Попробуйте ещё раз.")
                    continue
            else:
                exact = matches[matches['title'].str.lower() == choice.lower()]
                if len(exact) == 1:
                    target = exact.iloc[0]
                    break
                else:
                    print("Название не найдено среди вариантов. Попробуйте ещё раз или введите 0.")
                    continue
    else:
        target = matches.iloc[0]

    similarities = []
    for _, row in df_clean.iterrows():
        if row['title'] == target['title']:
            continue
        sim = combined_similarity(target, row, w_numeric=0.4, w_genre=0.3, w_actors=0.3)
        similarities.append((row['title'], sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return target['title'], similarities[:top_n]



def plot_user_type_similarity(sim_scores, user_types):
    df_sim = pd.DataFrame({'user_type': user_types, 'similarity': sim_scores})
    grouped = df_sim.groupby('user_type', as_index=False)['similarity'].mean()
    grouped = grouped.sort_values('similarity', ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=grouped, x='user_type', y='similarity', hue='user_type', legend=False)
    plt.title('Сходство ваших предпочтений с профилями синтетических пользователей')
    plt.xlabel('Тип пользователя')
    plt.ylabel('Среднее косинусное сходство')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('user_similarity_to_types.png', dpi=150)
    plt.show()
    print("График сходства сохранён как user_similarity_to_types.png")



while True:
    print("\n" + "=" * 50)
    print("Выберите действие:")
    print("1 — Коллаборативная фильтрация (Оцените несколько часов)")
    print("2 — Контентная фильтрация (Похожие часы по свойствам)")
    print("0 — Выход")
    print("=" * 50)
    mode = input("Ваш выбор: ").strip()

    if mode == '1':
        print("\nОцените несколько часов от 1 до 5 (или введите Стоп для завершения).")
        user_ratings = {}
        while True:
            title = input("\nНазвание часов: ").strip()
            if title.lower() == 'стоп':
                break

            # Поиск совпадений
            # 1. Точное вхождение подстроки
            exact_matches = [m for m in all_watches if title.lower() in m.lower()]
            if exact_matches:
                matching = exact_matches
            else:
                # 2. Поиск по расстоянию Левенштейна
                threshold = 6
                candidates = []
                for watch in all_watches:
                    dist = levenshtein(title, watch)
                    if dist <= threshold:
                        candidates.append((watch, dist))
                if candidates:
                    candidates.sort(key=lambda x: x[1])
                    matching = [c[0] for c in candidates[:10]]
                else:
                    matching = []

            if not matching:
                print("Часы не найдены. Попробуйте ещё раз.")
                continue

            # Если нашлось несколько вариантов
            if len(matching) > 1:
                print("Найдено несколько вариантов:")
                for i, m in enumerate(matching[:5]):  # показываем первые 5
                    print(f"  {i + 1}. {m}")
                choice = input("Введите номер (или 0, чтобы ввести название заново): ").strip()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(matching):
                        title = matching[idx]
                    else:
                        continue
                else:
                    continue
            else:
                title = matching[0]

            try:
                rating = float(input("Оценка (1-5): "))
                if rating < 1 or rating > 5:
                    print("Оценка должна быть от 1 до 5.")
                    continue
                user_ratings[title] = rating
                print(f"Добавлено: {title} — {rating}")
            except ValueError:
                print("Введите число.")

        if len(user_ratings) == 0:
            print("Не введено ни одной оценки. Завершаем.")
            exit()

        print("\nВаши оценки:")
        for title, rating in user_ratings.items():
            print(f"  {title}: {rating}")

        print("\nИщем похожих пользователей...")
        recommendations, sim_scores = get_collab_recommendations(user_ratings, top_k=10, top_n=5)

        plot_user_type_similarity(sim_scores, user_types_list)

        if not recommendations:
            print("Не удалось найти рекомендации. Попробуйте оценить другие часы.")
        else:
            print("\nРекомендуемые часы (коллаборативная фильтрация):")
            for i, (title, score) in enumerate(recommendations, 1):
                print(f"{i}. {title} — прогноз: {score:.2f}")


    elif mode == '2':
        title_input = input("\nВведите название часов (для поиска похожих): ").strip()
        target, recs = get_content_recommendations(title_input, top_n=5)
        if target is None:
            print(recs)
        else:
            print(f"\nЧасы, похожие на '{target}' (по свойствам):")
            for i, (title, sim) in enumerate(recs, 1):
                print(f"{i}. {title} (сходство: {sim:.3f})")

    else:
        print("Выход.")
        sys.exit()
