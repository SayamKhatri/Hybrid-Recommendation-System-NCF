import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, regularizers, optimizers

def ncf_model(train_data_final):
    num_users = train_data_final['user_id_encoded'].nunique()  
    num_items = train_data_final['movie_id_encoded'].nunique()  


    embedding_size = 32
    l2_reg = 1e-4

    user_input = layers.Input(shape=(), name='user_id_encoded', dtype=tf.int32)
    movie_input = layers.Input(shape=(), name='movie_id_encoded', dtype=tf.int32)

    user_embedding = layers.Embedding(num_users, embedding_size, name='user_embeddings',
                                    embeddings_regularizer=regularizers.l2(l2_reg))(user_input)
    movie_embedding = layers.Embedding(num_items, embedding_size, name='movie_embeddings',
                                    embeddings_regularizer=regularizers.l2(l2_reg))(movie_input)

    interaction = layers.Concatenate()([user_embedding, movie_embedding])

    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(interaction)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=[user_input, movie_input], outputs=output)


    optimizer = optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()