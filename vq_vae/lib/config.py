import numpy as np
from pathlib import Path

base_dir = Path(__file__).resolve().parent

data_dir = base_dir.parent.parent / "1. spectrogram" / "temp"

latent_dim = 128 # 每個 token 的向量長度
num_embeddings = [32, 64, 128, 256, 512, 1024, 2048]    # 碼本大小 vocab size，對應 token 的種類數量
commitment_cost = 0.25   # 量化損失的權重

n_dict = {
    n: (base_dir.parent / f"{n}/vq_vae_{n}.weights.h5",
        base_dir.parent / f"{n}/vq_vae_history_{n}.csv",
        base_dir.parent / f"{n}/vq_vae_history_{n}.png",
        base_dir.parent / f"{n}/vq_vae_matrix_{n}.csv",
        base_dir.parent / f"{n}/vq_vae_tokens_{n}.csv",)
    for n in num_embeddings
}

seed = 42

batch_size, epochs = 128, 100

input_size, channel = 128, 1
output_size = 4

base_filter = 64

conv_layers = int(np.log2(input_size // output_size))   # 卷積層數量
input_shape = (input_size, input_size, channel) # spectrogram 的大小和通道數
token_shape_2D = (output_size, output_size) # spectrogram 對應 token 的 2D 形狀
token_length = output_size * output_size    # spectrogram 對應的 token 長度