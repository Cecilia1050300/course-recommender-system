from lib import config as cfg
from lib import data
from lib import model

def main():
    for num_embedding in cfg.num_embeddings:
        (cfg.base_dir.parent / f"{num_embedding}").mkdir(exist_ok=True, parents=True)

        file_paths, train_dataset, inference_dataset = data.get_data()
        vq_vae, encoder, decoder, vq_layer = model.build_vq_vae(num_embedding)
        history, vq_vae = model.compile_and_train_vq_vae(vq_vae, train_dataset, num_embedding)

        data.write_history(history, num_embedding)
        data.draw_history(history, num_embedding)
        data.save_matrix(vq_layer, num_embedding)
        data.save_tokens(file_paths, inference_dataset, encoder, vq_layer, num_embedding)

if __name__ == "__main__":
    main()