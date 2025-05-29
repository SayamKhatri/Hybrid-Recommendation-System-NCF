import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model

def evaluation(model, train_data_final, train_df, val_df):

    val_movies = set(val_df['movie_id_encoded'].unique())
    all_movies = set(train_data_final['movie_id_encoded'].unique())
    missing_movies = val_movies - all_movies

    user_rated_movies = train_df.groupby('user_id_encoded')['movie_id_encoded'].apply(set).to_dict()
    

    def evaluate_model(model, val_df, user_rated_movies, all_movies, num_negatives=50, k=10, max_users=None):
        val_df = val_df[val_df['movie_id_encoded'].isin(all_movies)]


        hr, ndcg = [], []
        users_to_eval = val_df['user_id_encoded'].unique()
        if max_users:
            users_to_eval = users_to_eval[:max_users]
        batch_size = 1000
        skipped_users = 0

        for start in range(0, len(users_to_eval), batch_size):
            batch_users = users_to_eval[start:start + batch_size]
            batch_users_list, batch_items_list = [], []
            batch_pos_items, batch_neg_items = [], []

            for user in batch_users:
                pos_item = val_df[val_df['user_id_encoded'] == user]['movie_id_encoded'].values[0]
                non_rated = list(all_movies - user_rated_movies.get(user, set()))

                if not non_rated:
                    skipped_users += 1
                    continue

                neg_items = np.random.choice(non_rated, size=min(num_negatives, len(non_rated)), replace=False)
                items = np.array([pos_item] + list(neg_items))
                users = np.array([user] * len(items))

                batch_users_list.extend(users)
                batch_items_list.extend(items)
                batch_pos_items.append(pos_item)
                batch_neg_items.append(neg_items)


            if not batch_users_list:
                continue

            scores = model.predict(
                [np.array(batch_users_list, dtype=np.int32), np.array(batch_items_list, dtype=np.int32)],
                verbose=0,
                batch_size=256
            ).flatten()
            idx = 0

            for i, pos_item in enumerate(batch_pos_items):
                num_items = num_negatives + 1
                user_scores = scores[idx:idx + num_items]
                top_k_indices = np.argsort(user_scores)[::-1][:k]
                items = np.array([pos_item] + list(batch_neg_items[i]))

                top_k_items = items[top_k_indices]
                hr.append(int(pos_item in top_k_items))

                if pos_item in top_k_items:
                    rank = np.where(top_k_items == pos_item)[0][0]
                    ndcg.append(1.0 / np.log2(rank + 2))
                else:
                    ndcg.append(0.0)
                idx += num_items

        if skipped_users:
            print(f"Skipped {skipped_users} users with no non-rated items.")
        return np.mean(hr) if hr else 0.0, np.mean(ndcg) if ndcg else 0.0

    user_rated_movies = train_df.groupby('user_id_encoded')['movie_id_encoded'].apply(set).to_dict()
    all_movies = set(train_data_final['movie_id_encoded'].unique())

    hr, ndcg = evaluate_model(model, val_df, user_rated_movies, all_movies, max_users=None, num_negatives=100)
    print(f"Test HR@10: {hr:.4f}, NDCG@10: {ndcg:.4f}")