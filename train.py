import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import config
from data import PointsDataset
from model import DeepSDF


def sdf_loss(pred, gt):
    # pred = torch.clamp(pred, -1, 1)
    # gt = torch.clamp(gt, -1, 1)
    return F.mse_loss(pred, gt)


if __name__ == '__main__':
    mesh_path = 'bunny.obj'
    train_data = PointsDataset(mesh_path, config.num_samples)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

    model = DeepSDF(use_dropout=config.use_dropout)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    running_loss = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(config.epochs):
        for i, data in enumerate(train_loader):
            pts = data['pts']
            gt_sdf = data['sdf']

            optim.zero_grad()
            pred_sdf = model(pts)
            pred_sdf = pred_sdf.flatten()

            loss = sdf_loss(pred_sdf, gt_sdf)
            loss.backward()
            optim.step()

            running_loss += loss.item()
            if i % config.log_interval == config.log_interval - 1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / config.log_interval:.3f}')
                running_loss = 0.0

    torch.save(model.state_dict(), 'model.pth')
