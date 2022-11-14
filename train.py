import clip
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

import config
from data import PointsDataset
from model import DeepSDF
from render_sdf import SDFRenderer


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
    text_sentences = ['a bunny', 'a photo of a bunny']
    points_path, normals_path = './output/point_cloud.npy', './output/normals.npy'

    mesh_path = 'models/bunny.obj'
    train_data = PointsDataset(config.num_samples, points_path, normals_path, mesh_path=mesh_path, source='mesh')
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

    sdf_model = DeepSDF(use_dropout=config.use_dropout)
    optim = torch.optim.Adam(sdf_model.parameters(), lr=config.lr)
    running_loss = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    sdf_renderer = SDFRenderer()

    # every 200 epoches, the first half don't use clip, the second half use
    # if_this_epoch_use_clip = lambda epoch: ((epoch % 102) > 2)
    if_this_epoch_use_clip = lambda epoch: epoch > 0
    sdf_model.to(device)
    sdf_model.train()

    for epoch in range(config.epochs):
        use_clip = if_this_epoch_use_clip(epoch)
        for i, data in enumerate(train_loader):
            pts = data['pts']
            gt_sdf = data['sdf']

            optim.zero_grad()
            if not use_clip:
                pred_sdf = sdf_model(pts)
                pred_sdf = pred_sdf.flatten()
                loss = sdf_loss(pred_sdf, gt_sdf)
                loss.backward()
                optim.step()
            else:
                image = sdf_renderer(sdf_model)
                torchvision.utils.save_image(image, f'out.png')
                loss = clip_loss(image, clip_model, clip_preprocess, text_sentences, device)
                loss.backward()
                optim.step()

            running_loss += loss.item()
            if i % config.log_interval == config.log_interval - 1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / config.log_interval:.3f}')
                running_loss = 0.0

    torch.save(sdf_model.state_dict(), 'model.pth')
