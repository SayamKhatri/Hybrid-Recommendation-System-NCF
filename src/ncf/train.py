import pandas as pd
import tensorflow as tf 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from ncf.model import ncf_model
import numpy as np
from tqdm import tqdm
import sys 
import os

def data_prep(df_ratings):
    val_df = df_ratings.groupby('user_id_encoded', group_keys=False).sample(1, random_state=42)

    val_indices = val_df.index

    train_df = df_ratings.drop(val_indices).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Add Negative Sampling for users for movies they have not interacted with in train data 

    user_rated_movies = train_df.groupby('user_id_encoded')['movie_id_encoded'].apply(set).to_dict()
    all_movies = set(train_df['movie_id_encoded'].unique())

    NEGATIVE_RATIO = 3
    neg_samples = []

    for user in tqdm(train_df['user_id_encoded'].unique(), desc="Users Sampling"):
        user_watched_movies = user_rated_movies.get(user, set())
        n_positives = len(user_watched_movies)
        n_negatives = n_positives * NEGATIVE_RATIO
        candidates = np.array(list(all_movies - user_watched_movies))
        if len(candidates) == 0 or n_negatives == 0:
            continue
        n_samples = min(n_negatives, len(candidates))
        neg_movies = np.random.choice(candidates, size=n_samples, replace=False)
        neg_samples.append(
            pd.DataFrame({
                'user_id_encoded': [user]*n_samples,
                'movie_id_encoded': neg_movies,
                'implicit_feedback': [0]*n_samples
            })
        )

    df_negatives = pd.concat(neg_samples, ignore_index=True)

    # Data with positive + negative samples

    train_data_final = pd.concat([
        train_df[['user_id_encoded', 'movie_id_encoded' ,'implicit_feedback']],
        df_negatives
    ], ignore_index=True)


    NEGATIVE_RATIO = 2

    # Create validation dataset with negatives

    val_neg_samples = []
    for user in tqdm(val_df['user_id_encoded'].unique(), desc='creating val users'):
        user_watched_movies = user_rated_movies.get(user, set()) | set(val_df[val_df['user_id_encoded'] == user]['movie_id_encoded'])
        candidates = list(all_movies - user_watched_movies)
        if not candidates:
            continue
        n_negatives = NEGATIVE_RATIO  # 2 negatives per positive
        n_samples = min(n_negatives, len(candidates))
        neg_movies = np.random.choice(candidates, size=n_samples, replace=False)
        val_neg_samples.append(
            pd.DataFrame({
                'user_id_encoded': [user] * n_samples,
                'movie_id_encoded': neg_movies,
                'implicit_feedback': [0] * n_samples
            })
        )

    df_val_negatives = pd.concat(val_neg_samples, ignore_index=True) if val_neg_samples else pd.DataFrame()
    val_df_with_neg = pd.concat([
        val_df[['user_id_encoded', 'movie_id_encoded', 'implicit_feedback']],
        df_val_negatives
    ], ignore_index=True)


    # Preparing validation dataset for loss

    val_users = np.array(val_df_with_neg['user_id_encoded'], dtype=np.int32)
    val_movies = np.array(val_df_with_neg['movie_id_encoded'], dtype=np.int32)
    val_labels = np.array(val_df_with_neg['implicit_feedback'], dtype=np.float32)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        ((val_users, val_movies), val_labels)
    )
    val_dataset = val_dataset.batch(batch_size=256).prefetch(tf.data.AUTOTUNE)

    # Verify validation data
    print("Validation samples with negatives:", len(val_df_with_neg))
    print("Validation data balance:\n", val_df_with_neg['implicit_feedback'].value_counts())
    val_movies_missing = set(val_df['movie_id_encoded'].unique()) - all_movies
    print("Validation movies missing in training:", len(val_movies_missing))

    return train_data_final, val_df_with_neg, train_df, val_df


def train_ncf(df_ratings):
    train_data_final, val_df_with_neg , train_df, val_df= data_prep(df_ratings)

    # Prepare training dataset
    train_users = np.array(train_data_final['user_id_encoded'])
    train_movies = np.array(train_data_final['movie_id_encoded'])
    train_labels = np.array(train_data_final['implicit_feedback'])

    train_dataset = tf.data.Dataset.from_tensor_slices(
        ((train_users, train_movies), train_labels)
    )
    train_dataset = train_dataset.shuffle(buffer_size=100000, seed=42)
    train_dataset = train_dataset.batch(batch_size=256).prefetch(tf.data.AUTOTUNE)


    # Training model
    train_users = np.array(train_data_final['user_id_encoded'], dtype=np.int32)
    train_movies = np.array(train_data_final['movie_id_encoded'], dtype=np.int32)
    train_labels = np.array(train_data_final['implicit_feedback'], dtype=np.float32)

    val_users = np.array(val_df_with_neg['user_id_encoded'], dtype=np.int32)
    val_movies = np.array(val_df_with_neg['movie_id_encoded'], dtype=np.int32)
    val_labels = np.array(val_df_with_neg['implicit_feedback'], dtype=np.float32)


    checkpoint = ModelCheckpoint('new_best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)


    model = ncf_model(train_data_final)

    history = model.fit(
        [train_users, train_movies], train_labels,
        validation_data=([val_users, val_movies], val_labels),
        epochs=6, batch_size=256, verbose=1,
        callbacks=[checkpoint, early_stop]
    )

    sys.path.append(os.path.abspath(os.path.join('..')))
    model.save('/models/ncf_model.h5')

    return model, train_data_final, train_df, val_df