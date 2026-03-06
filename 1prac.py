import pandas as pd
import os

df = pd.read_csv('./Amazon_Men_Wrist_Watches.csv')

print(df.head())
print(df.info())

print("=== КОЛИЧЕСТВО ПРОПУСКОВ В КАЖДОМ СТОЛБЦЕ ===")
print(df.isnull().sum())
print("\n")


def parse_reviews(val):
    if pd.isna(val):
        return float('nan')
    val = str(val).strip().replace(',', '').upper()
    if 'K' in val:
        return float(val.replace('K', '')) * 1000
    elif 'M' in val:
        return float(val.replace('M', '')) * 1000000
    else:
        try:
            return float(val)
        except:
            return float('nan')

def parse_price(val):
    if pd.isna(val):
        return float('nan')
    val = str(val).strip().replace(',', '').replace(' ', '')
    try:
        return float(val)
    except:
        return float('nan')

df['number_of_reviews'] = df['number_of_reviews'].apply(parse_reviews)

df['rating'] = df['rating'].fillna(df['rating'].mean().round(2))
df['number_of_reviews'] = df['number_of_reviews'].fillna(df['number_of_reviews'].mean().round())
df['price'] = df['price'].apply(parse_price)
df['price'] = df['price'].fillna(df['price'].mean().round(2))

print("=== ТИПЫ ДАННЫХ ПОСЛЕ ПРЕОБРАЗОВАНИЙ ===")
print(df.dtypes)
print("\n")

df.to_csv('watches_processed.csv', index=False)
print("Обработанные данные сохранены в watches_processed.csv")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print("\n=== ИТОГОВАЯ ТАБЛИЦА (ПЕРВЫЕ 3 СТРОКИ) ===")
print(df[['brand_name', 'watch_name', 'rating', 'number_of_reviews', 'price']].head(5))
