import clip
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from data import PointsDataset
from model import DeepSDF
from render_sdf import SDFRenderer
from tools import generate_description, merge_path, get_pc_points_and_normals, \
    save_image

import argparse
import logging


def sdf_loss(pred, gt):
    pred = torch.clamp(pred, -1, 1)
    gt = torch.clamp(gt, -1, 1)
    return F.mse_loss(pred, gt)


def clip_loss(input_image, clip_model, clip_preprocess, text_sentences: list, device, func=torch.mean):
    transform = T.ToPILImage()
    image = transform(input_image)
    clip_image = clip_preprocess(image).unsqueeze(0).to(device)
    clip_text = clip.tokenize(text_sentences).to(device)

    image_feats = clip_model.encode_image(clip_image)
    text_feats = clip_model.encode_text(clip_text)

    # Calculate cosine similarity
    norm = torch.norm(image_feats, dim=1) * torch.norm(text_feats, dim=1)
    similarity = torch.sum(image_feats * text_feats, dim=1) / norm
    loss = 1 - similarity
    return loss.mean()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    points_path, normals_path = './output/point_cloud.npy', './output/normals.npy'

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--use_dropout", type=bool, default=False)

    parser.add_argument("--categories", type=str, nargs='+', default=['plane', 'airplane'])
    parser.add_argument("--input_directory", type=str, default='./bproc_render_output')
    parser.add_argument("--output_directory", type=str, default='./results')
    parser.add_argument("--point_cloud_filename", type=str, default='point_cloud.npy')
    parser.add_argument("--normals_filename", type=str, default='normals.npy')

    parser.add_argument("--sample_rate", type=float, default=50,
                        help='number of sample points over number of origin points')
    parser.add_argument("--sample_points", type=int, default=50000, help='number of sample points')
    parser.add_argument("--use_rate", type=bool, default=True, help='use rate(True) or points(False)')

    parser.add_argument("--render_res", type=bool, default=64, help='Rendering resolution')
    parser.add_argument("--log_interval", type=bool, default=10, help='Log every n intervals')

    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    use_dropout = args.use_dropout

    categories = args.categories
    input_directory = args.input_directory
    output_directory = args.output_directory
    point_cloud_filename = args.point_cloud_filename
    normals_filename = args.normals_filename

    sample_rate = args.sample_rate
    sample_points = args.sample_points
    use_rate = args.use_rate
    render_res = args.render_res
    log_interval = args.log_interval

    descriptions = generate_description(categories)
    print(f'descriptions: {descriptions}')

    mesh_path = 'models/bunny.obj'
    train_data = PointsDataset(sample_points, points_path, normals_path, mesh_path=mesh_path, source='mesh')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    sdf_model = DeepSDF(use_dropout=use_dropout)
    optim = torch.optim.Adam(sdf_model.parameters(), lr=learning_rate)

    running_loss = 0.0
    running_s_loss = 0.0
    running_c_loss = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    sdf_renderer = SDFRenderer(render_res)

    sdf_model.to(device)
    sdf_model.train()
    alpha = 0.9  # Scalar weight to control sdf loss vs. clip loss

    # Train just the SDF for 1 epoch, then both the SDF and the renderer
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            pts = data['pts']
            gt_sdf = data['sdf']
            optim.zero_grad()

            pred_sdf = sdf_model(pts)
            pred_sdf = pred_sdf.flatten()
            s_loss = sdf_loss(pred_sdf, gt_sdf)

            if epoch > 0:
                image = sdf_renderer(sdf_model)
                save_image('out.png', image)
                c_loss = clip_loss(image, clip_model, clip_preprocess, descriptions, device)

            if epoch == 0:
                loss = s_loss
                running_loss += loss.item()
                running_s_loss += s_loss.item()
            else:
                loss = alpha * s_loss + (1 - alpha) * c_loss
                running_loss += loss.item()
                running_c_loss += c_loss.item()

            loss.backward()
            optim.step()

            if i % log_interval == log_interval - 1:
                if epoch == 0:
                    print(f'[{epoch + 1}, {i + 1:5d}] SDF loss: {running_s_loss / log_interval:.3f}, '
                          f'Total loss: {running_loss / log_interval:.3f}')
                else:
                    print(f'[{epoch + 1}, {i + 1:5d}] SDF loss: {running_s_loss / log_interval:.3f}, '
                          f'CLIP loss: {running_c_loss / log_interval:.3f}, Total loss: {running_loss / log_interval:.3f}')

                running_loss = 0.0
                running_s_loss = 0.0
                running_c_loss = 0.0

    torch.save(sdf_model.state_dict(), 'model.pth')
