# import torch
# import clip
# from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram"]).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", logits_per_image)  # prints: [[0.9927937  0.00421068 0.00299572]]

import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse
def get_pc_points_and_normals_from_files(points_path, normals_path):
    with open(points_path, 'rb') as f:
        pc_points = np.load(f)
    with open(normals_path, 'rb') as f:
        pc_normals = np.load(f)
    return pc_points, pc_normals
if __name__ == '__main__':
    learning_times_stage_1, learning_times_stage_2 = 20, 10
    epoch_number_per_cycle = learning_times_stage_1 + learning_times_stage_2
    cycle_number = 2
    cycles_to_output_images = [0, 1, -1]
    for epoch in range(epoch_number_per_cycle * cycle_number):
        output_image = (epoch % epoch_number_per_cycle - learning_times_stage_1) in [i % learning_times_stage_2 for i in cycles_to_output_images]
        print(f'epoch: {epoch, output_image}')