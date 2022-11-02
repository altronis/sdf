import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import clip
from PIL import Image
import numpy as np

import config
from data import PointsDataset
from model import DeepSDF

def sdf_loss(pred, gt):
    # pred = torch.clamp(pred, -1, 1)
    # gt = torch.clamp(gt, -1, 1)
    return F.mse_loss(pred, gt)

def clip_loss(input_image, clip_model, clip_preprocess, text_sentences: list, device, func=np.mean):
    clip_image = clip_preprocess(Image(input_image)).unsqueeze(0).to(device)
    clip_text = clip.tokenize(text_sentences).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = clip_model(clip_image, clip_text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return func(probs)


if __name__ == '__main__':
    text_sentences = ['an airplane', 'a photo of an airplane']

    mesh_path = 'bunny.obj'
    train_data = PointsDataset(mesh_path, config.num_samples)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

    model = DeepSDF(use_dropout=config.use_dropout)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    running_loss = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    # every 200 epoches, the first half don't use clip, the second half use
    if_this_epoch_use_clip = lambda epoch: ((epoch % 200) > 100)

    model.to(device)
    model.train()

    for epoch in range(config.epochs):
        use_clip = if_this_epoch_use_clip(epoch)
        for i, data in enumerate(train_loader):
            pts = data['pts']
            gt_sdf = data['sdf']

            optim.zero_grad()
            if not use_clip:
                pred_sdf = model(pts, use_clip)
                pred_sdf = pred_sdf.flatten()
                loss = sdf_loss(pred_sdf, gt_sdf)
                loss.backward()
            else:
                image = model(pts, use_clip)
                loss = clip_loss(image, clip_model, clip_preprocess, text_sentences, device)
                loss.backward()

            optim.step()

            running_loss += loss.item()
            if i % config.log_interval == config.log_interval - 1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / config.log_interval:.3f}')
                running_loss = 0.0

    torch.save(model.state_dict(), 'model.pth')
