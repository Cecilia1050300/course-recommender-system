import tensorflow as tf
from tensorflow.keras import layers

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        w_init = tf.random_uniform_initializer()    # 初始化碼本向量
        self.embeddings = tf.Variable(  # 碼本向量矩陣，shape = (embedding_dim, num_embeddings)
            initial_value=w_init(shape=(self.embedding_dim, self.num_embeddings), dtype="float32"),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        input_shape = tf.shape(x)   # (Batch, H', W', embedding_dim)
        flattened = tf.reshape(x, [-1, self.embedding_dim]) # (Batch*H'*W', embedding_dim)

        distances = (   # 計算輸入向量與碼本向量的距離，shape = (Batch*H'*W', num_embeddings)
            tf.reduce_sum(flattened ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * tf.matmul(flattened, self.embeddings)
        )

        encoding_indices = tf.argmin(distances, axis=1) # (Batch*H'*W',) 每個輸入向量對應的最近碼本索引
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)   # (Batch*H'*W', num_embeddings) one-hot 編碼

        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True) # (Batch*H'*W', embedding_dim) 量化後的向量
        quantized = tf.reshape(quantized, input_shape)  # (Batch, H', W', embedding_dim) 恢復原始形狀

        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)  # 碼本向量更新
        q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)  # 編碼器更新
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss  # 總量化損失 = 編碼器更新損失 + 量化損失權重 * 碼本向量更新損失

        self.add_loss(vq_loss)  # 加入量化損失到模型的總損失中
        self.add_metric(vq_loss, name="vq_loss")    # 監控量化損失

        quantized = x + tf.stop_gradient(quantized - x) # Straight-Through Estimator，讓梯度直接傳回編碼器
        return quantized

    def get_indices(self, x):
        input_shape = tf.shape(x)   # (Batch, H', W', embedding_dim)
        flattened = tf.reshape(x, [-1, self.embedding_dim]) # (Batch*H'*W', embedding_dim)

        distances = (tf.reduce_sum(flattened ** 2, axis=1, keepdims=True)
                    + tf.reduce_sum(self.embeddings ** 2, axis=0)
                    - 2 * tf.matmul(flattened, self.embeddings))

        encoding_indices = tf.argmin(distances, axis=1) # (Batch*H'*W',) 每個輸入向量對應的最近碼本索引
        return tf.reshape(encoding_indices, [input_shape[0], input_shape[1], input_shape[2]])   # 恢復 (Batch, H', W')

    def get_config(self):
        config = super().get_config()   # 返回層的配置字典，包含初始化參數
        config.update({ # 添加自定義參數到配置字典中，以便在保存和加載模型時使用
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "commitment_cost": self.commitment_cost,
        })
        return config