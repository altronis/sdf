import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
import clip
import numpy as np
import argparse
import logging

from data import PointsDataset
from model import DeepSDF
from render import render
from tools import generate_description, merge_path, get_pc_points_and_normals, save_image, generate_intrinsics

def sdf_loss(pred, gt):
    # pred = torch.clamp(pred, -1, 1)
    # gt = torch.clamp(gt, -1, 1)
    return F.mse_loss(pred, gt)

def clip_loss(input_image, clip_model, clip_preprocess, text_sentences: list, device, func=(lambda x: torch.mean(-x))):
    transform = T.ToPILImage()
    image = transform(input_image)
    clip_image = clip_preprocess(image).unsqueeze(0).to(device)
    clip_text = clip.tokenize(text_sentences).to(device)

    logits_per_image, logits_per_text = clip_model(clip_image, clip_text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return func(logits_per_image)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--learning_times_stage_1", type=int, default=20)
    parser.add_argument("--learning_times_stage_2", type=int, default=20)
    parser.add_argument("--output_loss_interval", type=int, default=100)
    parser.add_argument("--cycle_number", type=int, default=200)
    parser.add_argument("--cycles_to_output_images", type=int, nargs='+', default=[0, -1])
    parser.add_argument("--use_dropout", type=bool, default=False)
    parser.add_argument("--categories", type=str, nargs='+', default=['plane', 'airplane'])
    parser.add_argument("--input_directory", type=str, default='./bproc_render_output')
    parser.add_argument("--output_directory", type=str, default='./results')
    parser.add_argument("--point_cloud_filename", type=str, default='point_cloud.npy')
    parser.add_argument("--normals_filename", type=str, default='normals.npy')
    parser.add_argument("--sample_rate", type=float, default=50, help='number of sample points over number of origin points')
    parser.add_argument("--sample_points", type=int, default=50000, help='number of sample points')
    parser.add_argument("--use_rate", type=bool, default=True, help='use rate(True) or points(False)')

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    learning_times_stage_1 = args.learning_times_stage_1
    learning_times_stage_2 = args.learning_times_stage_2
    output_loss_interval = args.output_loss_interval
    cycle_number = args.cycle_number
    cycles_to_output_images = args.cycles_to_output_images
    use_dropout = args.use_dropout
    categories = args.categories
    input_directory = args.input_directory
    output_directory = args.output_directory
    point_cloud_filename = args.point_cloud_filename
    normals_filename = args.normals_filename
    sample_rate = args.sample_rate
    sample_points = args.sample_points
    use_rate = args.use_rate

    descriptions = generate_description(categories)

    print(f'descriptions: {descriptions}')
    pc_points, pc_normals = get_pc_points_and_normals(points_path=merge_path(input_directory, point_cloud_filename),
                                                      normals_path=merge_path(input_directory, normals_filename),
                                                      source='file',
                                                      normalize=True
                                                     )
    num_samples = int(sample_rate * len(pc_points)) if use_rate else sample_points
    print(f'num_samples: {num_samples}')
    print(f'pc_points.shape: {pc_points.shape}, pc_normals.shape: {pc_normals.shape}')
    train_data = PointsDataset(num_samples, pc_points, pc_normals)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = DeepSDF(use_dropout=use_dropout)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    # every 200 epoches, the first half don't use clip, the second half use
    # if_this_epoch_use_clip = lambda epoch: ((epoch % 20) >= 10)
    model.to(device)
    model.train()

    epoch_number_per_cycle = learning_times_stage_1 + learning_times_stage_2
    for epoch in range(epoch_number_per_cycle * cycle_number):
        use_clip = ((epoch % epoch_number_per_cycle) >= learning_times_stage_1)
        output_image = (epoch % epoch_number_per_cycle - learning_times_stage_1) in [i % learning_times_stage_2 for i in cycles_to_output_images]

        if not use_clip:
            for i, data in enumerate(train_loader):
                pts = data['pts']
                gt_sdf = data['sdf']
                optim.zero_grad()
                pred_sdf = model(pts)
                pred_sdf = pred_sdf.flatten()
                loss = sdf_loss(pred_sdf, gt_sdf)
                loss.backward()
                optim.step()
                if i % output_loss_interval == 0:
                    logging.info(f'epoch: {epoch}, i: {i}, loss: {loss.item():.5f}')
        else:
            intrinsics = generate_intrinsics(focal_x=256, focal_y=256, width=256, height=256)
            image = render(model, intrinsics, distance=2.5, azimuth=np.pi/6, elevation=np.pi/8, device=device)
            optim.zero_grad()
            if output_image:
                save_image(output_directory, f'epoch_{epoch}.png', image)
            loss = clip_loss(image, clip_model, clip_preprocess, descriptions, device)
            loss.backward()
            optim.step()
            logging.info(f'epoch: {epoch}, loss: {loss.item():.5f}')

    torch.save(model.state_dict(), 'model.pth')
