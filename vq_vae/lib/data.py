import tensorflow as tf
import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt
from pathlib import Path

from lib import config as cfg

def python_load_image(path):
    path_str = path.numpy().decode("utf-8")
    with Image.open(path_str).convert("L") as img:
        img = img.resize(cfg.input_shape[:2])
        img_array = np.array(img).astype(np.float32) / 255.0
        return np.expand_dims(img_array, axis=-1)

def load_and_preprocess(path):
    img = tf.py_function(python_load_image, [path], tf.float32)
    img.set_shape(cfg.input_shape)
    return img

def get_data():
    file_paths = sorted(list(cfg.data_dir.rglob("*.png")))
    file_paths = [str(p) for p in file_paths]
    path_dataset = tf.data.Dataset.from_tensor_slices(file_paths)

    img_dataset = path_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = img_dataset.shuffle(
        buffer_size=len(file_paths)//2,
        seed=cfg.seed,
        reshuffle_each_iteration=True
    ).batch(cfg.batch_size).map(lambda x: (x, x)).prefetch(tf.data.AUTOTUNE)
    inference_dataset = img_dataset.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    return file_paths, train_dataset, inference_dataset

def write_history(history, num_embedding):
    data = history.history if hasattr(history, "history") else history

    df = pd.DataFrame(data)
    df.reset_index(inplace=True)

    df.rename(columns={"index": "epoch"}, inplace=True)
    df["epoch"] = df["epoch"] + 1

    df.to_csv(cfg.n_dict[num_embedding][1], sep=",", index=False, float_format="%.6f")

def draw_history(history, num_embedding):
    plt.figure(figsize=(12, 7))

    metrics_to_plot = [
        ("loss", "Total Loss (MSE + VQ)", "royalblue"),
        ("vq_loss", "VQ", "orange"),
        ("mse", "MSE", "seagreen"),
        ("mae", "MAE", "crimson"),
        ("rmse", "RMSE", "darkorchid")
    ]

    for m, label, color in metrics_to_plot:
        if m in history.history:
            lw = 3 if m == "loss" else 2
            ls = "--" if m == "vq_loss" else "-"
            plt.plot(
                history.history[m], label=label,
                color=color, linewidth=lw,
                linestyle=ls, alpha=0.8,
                marker="o", markersize=3
            )

    plt.title("VQ-VAE Training Curves", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Value", fontsize=12)

    all_vals = [v for k in history.history for v in history.history[k] if k in [m[0] for m in metrics_to_plot]]
    if all_vals:
        plt.ylim(0, max(all_vals) * 1.1)

    plt.grid(True, which="major", linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, fontsize=12, shadow=True)

    plt.tight_layout()
    plt.savefig(cfg.n_dict[num_embedding][2])

def save_matrix(vq_layer, num_embedding):
    codebook_weights = vq_layer.embeddings.numpy().T
    df_codebook = pd.DataFrame(codebook_weights)
    df_codebook.to_csv(cfg.n_dict[num_embedding][3], index=True, header=False)

def save_tokens(file_paths, inference_dataset, encoder, vq_layer, num_embedding):
    results, file_idx = [], 0

    for batch_imgs in inference_dataset:
        encoded_outputs = encoder(batch_imgs, training=False)
        batch_indices = vq_layer.get_indices(encoded_outputs)
        flat_tokens = tf.reshape(batch_indices, [tf.shape(batch_indices)[0], -1]).numpy()

        for i in range(len(flat_tokens)):
            tokens_list = flat_tokens[i].tolist()
            results.append([Path(file_paths[file_idx]).stem] + tokens_list)
            file_idx += 1

    column_names = ["img"] + [f"token{i}" for i in range(cfg.token_length)]
    df_tokens = pd.DataFrame(results, columns=column_names)
    df_tokens.to_csv(cfg.n_dict[num_embedding][4], index=False, header=False)