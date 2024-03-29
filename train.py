import argparse
import torch
from torch.optim import Adam
from torch.autograd import grad
from torch.utils.data import DataLoader
from dataset import get_dataset
from models import Generator, Discriminator
from utils import resize_image, concat_image, save_image


class GradientPenalty:
    def __init__(self, batch_size, gp_lambda, device):
        self.batch_size = batch_size
        self.gp_lambda = gp_lambda
        self.device = device

    def __call__(self, discriminator, real_data, fake_data, progress):
        alpha = torch.rand(self.batch_size, 1, 1, 1, requires_grad=True, device=self.device)
        interpolates = (1 - alpha) * real_data + alpha * fake_data
        d_interpolates = discriminator(interpolates, progress.alpha, progress.stage)

        gradients = grad(outputs=d_interpolates,
                         inputs=interpolates,
                         grad_outputs=torch.ones(d_interpolates.size(), device=self.device),
                         create_graph=True,
                         retain_graph=True,
                         only_inputs=True)[0]
        gradients = gradients.view(self.batch_size, -1)
        gradient_penalty = self.gp_lambda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty


class Progress:
    def __init__(self, max_stage, max_epoch, max_step):
        self.alpha = 0
        self.stage = 0
        self.max_stage = max_stage
        self.max_epoch = max_epoch
        self.max_step = max_step

    def progress(self, current_stage, current_epoch, current_step):
        self.stage = current_stage
        p = (current_epoch * self.max_step + current_step) / (self.max_epoch * self.max_step)
        self.alpha = p if 0 < current_stage < self.max_stage else 1


def train(generator, discriminator, opt):
    g_optimizer = Adam(generator.parameters(), lr=1e-3, betas=(0, 0.99))
    d_optimizer = Adam(discriminator.parameters(), lr=1e-3, betas=(0, 0.99))

    for stage in range(opt.num_stages + 1):
        train_dataset = get_dataset(data_name=opt.dataset, data_root=opt.data_root, stage=stage, max_stage=opt.num_stages, train=True)
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size[stage], shuffle=True)

        gp = GradientPenalty(opt.batch_size[stage], 10, opt.device)
        progress = Progress(opt.num_stages, opt.num_epochs, len(train_loader))

        generator.train()
        discriminator.train()
        for epoch in range(opt.num_epochs):
            for i, (real_images,  label) in enumerate(train_loader):
                real_images = real_images.to(opt.device)
                progress.progress(stage, epoch, i)

                # discriminator
                generator.zero_grad()
                discriminator.zero_grad()

                latent_dim = min(opt.base_channels * 2 ** opt.num_stages, 512)
                z = torch.randn(opt.batch_size[stage], latent_dim, 1, 1, device=opt.device)

                with torch.no_grad():
                    fake_images = generator(z, progress.alpha, progress.stage)

                d_real = discriminator(real_images, progress.alpha, progress.stage).mean()
                d_fake = discriminator(fake_images, progress.alpha, progress.stage).mean()

                gradient_penalty = gp(discriminator, real_images.data, fake_images.data, progress)

                epsilon_penalty = (d_real ** 2).mean() * 0.001

                d_loss = d_fake - d_real
                d_loss_gp = d_loss + gradient_penalty + epsilon_penalty

                d_loss_gp.backward()
                d_optimizer.step()

                # generator
                generator.zero_grad()
                discriminator.zero_grad()

                latent_dim = min(opt.base_channels * 2 ** opt.num_stages, 512)
                z = torch.randn(opt.batch_size[stage], latent_dim, 1, 1, device=opt.device)

                fake_images = generator(z, progress.alpha, progress.stage)
                g_fake = discriminator(fake_images, progress.alpha, progress.stage).mean()
                g_loss = -g_fake

                g_loss.backward()
                g_optimizer.step()

            print("Stage:{:>2} | Epoch :{:>3} | Dis Loss:{:>10.5f} | Gen Loss:{:>10.5f}".format(stage, epoch, d_loss, g_loss))
            fake_images = fake_images.permute(0, 2, 3, 1).cpu().detach().numpy()
            fake_images = concat_image(fake_images)
            fake_images = resize_image(fake_images, 200)
            save_image("./outputs/train/{}stage_{}epoch.jpg".format(stage, epoch), fake_images)


def main():
    parser = argparse.ArgumentParser(description="PGGAN")
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--batch_size", type=list, default=[32, 32, 32, 32, 32, 16, 8, 4, 2])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--image_channels", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    opt = parser.parse_args()

    generator = Generator(max_stage=opt.num_stages, base_channels=opt.base_channels, image_channels=opt.image_channels).to(opt.device)
    discriminator = Discriminator(max_stage=opt.num_stages, base_channels=opt.base_channels, image_channels=opt.image_channels).to(opt.device)

    train(generator, discriminator, opt)
    torch.save(generator.state_dict(), "./weights/generator.pth")
    torch.save(discriminator.state_dict(), "./weights/discriminator.pth")


if __name__ == "__main__":
    main()
