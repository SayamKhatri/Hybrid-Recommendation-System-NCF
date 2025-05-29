import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from utils.config import DATA_PATH, MOVIE_PATH

df_movies = pd.read_parquet(DATA_PATH)
df_ratings = pd.read_parquet(MOVIE_PATH)

# Creating year from title 

df_movies['year'] = df_movies['title'].str.extract(r'\((\d{4})\)')
df_movies['list_of_genres'] = df_movies['genres'].str.split('|')
df_movies= df_movies[['movie_id', 'title', 'year', 'list_of_genres']]

unique_genres = set()

for movie_genre in df_movies['list_of_genres'].values:
    movie_genre_set = set(movie_genre)
    unique_genres = unique_genres | movie_genre_set


for genre in unique_genres:
    df_movies[genre] = 0

idx = 0
for movie_id in df_movies['movie_id'].unique():
    for columns in df_movies[df_movies['movie_id'] == movie_id]['list_of_genres'].values[0]:
        df_movies.loc[idx, columns] = 1
    idx+=1

df_m = df_movies.drop(columns=['list_of_genres'])
df_m = df_m.drop(index=7903).reset_index(drop = True)

scaler = MinMaxScaler()

df_m['year_scaled'] = scaler.fit_transform(df_m[['year']])

columns_to_include = [
    'Mystery', 'Film-Noir', 'Children', 'IMAX', 'Adventure', 'Western',
    'Comedy', 'Documentary', 'Fantasy', 'Action', 'Musical', 'Crime',
    'Horror', 'Sci-Fi', 'Animation', 'Thriller', 'War', 'Romance', 'Drama',
    'year_scaled'
]

df_m['movie_vector'] = df_m[columns_to_include].values.astype(int).tolist()
positive_movies = df_ratings[df_ratings['implicit_feedback'] == 1]
result = positive_movies.merge(df_m[['movie_id', 'movie_vector']], on='movie_id', how='inner')

user_id_mapping = df_ratings.set_index('user_id')['user_id_encoded'].to_dict()

# KEY - VALUE
# ORIGINAL USER ID - ENCODED USER ID
movie_id_mapping = df_ratings.set_index('movie_id')['movie_id_encoded'].to_dict()


user_profiles = {}

for user_id, group in result.groupby('user_id_encoded'):
    v = np.array(group['movie_vector'].tolist())
    profile = v.mean(axis = 0)
    user_profiles[user_id] = profile.tolist()




def recommend_content_for_user(user_id, user_profiles, df_m, top_n=10):
    user_vector = user_profiles[user_id]
    scores = []

    for idx, row in df_m.iterrows():
        movie_vector = np.array(row['movie_vector'])
        score = np.dot(user_vector, movie_vector)
        scores.append(score)

    df_m = df_m.copy()
    df_m['cb_score'] = scores

    recommended = df_m.sort_values('cb_score', ascending=False).head(top_n)
    return recommended[['movie_id', 'title', 'cb_score']]

user_id = 0 
top_cb_recs = recommend_content_for_user(user_id, user_profiles, df_m, top_n=10)
print(top_cb_recs)
