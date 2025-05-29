import pandas as pd
from src.utils.config import DATA_PATH

def load_and_preprocess():
    df_ratings = pd.read_parquet(DATA_PATH)

    def mapping(x):
        if x >= 4.0:
            return 1
        else:
            return 0

    df_ratings['implicit_feedback'] = df_ratings['rating'].apply(mapping)

    df_ratings = df_ratings[df_ratings['implicit_feedback'] == 1].reset_index(drop=True)

    # Filtering users who only have more than 20 positive interaction

    positive_interaction_per_user = df_ratings[df_ratings['implicit_feedback'] == 1] \
                                    .groupby('user_id')['movie_id'].count()

    rare_users = positive_interaction_per_user[positive_interaction_per_user < 20] \
                                        .index.tolist()

    df_ratings = df_ratings[~df_ratings['user_id'].isin(rare_users)]

    # Filtering out rare movies

    movie_interaction_per_user = df_ratings.groupby('movie_id')['user_id'].count()

    rare_movies = movie_interaction_per_user[movie_interaction_per_user < 4] \
                                        .index.tolist()

    df_ratings = df_ratings[~df_ratings['movie_id'].isin(rare_movies)]

    # Encoding user and movie id's to a continous scale as expectd by NCF
    from sklearn.preprocessing import LabelEncoder

    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    df_ratings['user_id_encoded'] = user_encoder.fit_transform(df_ratings['user_id'])
    df_ratings['movie_id_encoded'] = movie_encoder.fit_transform(df_ratings['movie_id'])


    # no of unique users in our data

    print('No of unique users:', df_ratings['user_id_encoded'].nunique())

    # no of unique movies in our data

    print('No of unique movies:', df_ratings['movie_id_encoded'].nunique())


    return df_ratings