import tensorflow as tf
from tensorflow.keras import layers, regularizers


class NCF(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_size=32, **kwargs):
        super(NCF, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size

        # Define layers with regularization
        self.user_embedding = layers.Embedding(num_users, embedding_size)
        self.item_embedding = layers.Embedding(num_items, embedding_size)
        self.dense1 = layers.Dense(128, activation='relu',
                                   kernel_regularizer=regularizers.l2(0.03))
        self.dropout1 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(64, activation='relu',
                                   kernel_regularizer=regularizers.l2(0.03))
        self.dropout2 = layers.Dropout(0.3)

        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        item_vector = self.item_embedding(inputs[:, 1])
        interaction = tf.concat([user_vector, item_vector], axis=1)

        # Apply dense layers with dropout
        x = self.dense1(interaction)
        x = self.dropout1(x)  # Apply dropout after the first dense layer
        x = self.dense2(x)
        x = self.dropout2(x)  # Apply dropout after the second dense layer
        return self.output_layer(x)

    def get_config(self):
        config = super(NCF, self).get_config()
        config.update({
            'num_users': self.num_users,
            'num_items': self.num_items,
            'embedding_size': self.embedding_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        num_users = config.get('num_users', 1000)
        num_items = config.get('num_items', 1000)
        embedding_size = config.get('embedding_size', 32)
        return cls(num_users, num_items, embedding_size)
