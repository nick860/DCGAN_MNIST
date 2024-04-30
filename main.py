import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from GAN_FACE import *
import matplotlib.pyplot as plt
from tqdm import tqdm
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 128
    IMAGE_SIZE = 64
    CHANNELS_IMG = 1
    Z_DIM = 100
    NUM_EPOCHS = 5
    FEATURES_DISC = 64
    FEATURES_GEN = 64

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE), # resize image to 64x64
            transforms.ToTensor(), # convert image to tensor
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)] # normalize image to [-1, 1]
            ),
        ]
    )
    
    dataset = datasets.MNIST(root="dataset/", train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
    initialize_weights(gen) # initialize weights of generator
    initialize_weights(disc) # initialize weights of discriminator

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)) # betas are the exponential decay rates for the first and second moment estimates
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss() # Binary Cross Entropy Loss

    fixied_noise = torch.randn(32, Z_DIM, 1, 1).to(device) # 32 random noise vectors, its 32 because we want to show 32 images in a grid
    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")
    step = 0

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, _) in enumerate(tqdm(loader)):
            real = real.to(device)
            noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
            fake = gen(noise)

            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1) # detach fake tensor from the computational graph
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixied_noise)
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                step += 1
    torch.save(gen.state_dict(), "gen.pth")
    torch.save(disc.state_dict(), "disc.pth")
    writer_real.close()
    writer_fake.close()

def genrate_image():
    Z_DIM = 100
    CHANNELS_IMG = 1
    FEATURES_GEN = 64
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN)
    gen.load_state_dict(torch.load("gen.pth", map_location=torch.device('cpu')))
    gen.eval()
    fixed_noise = torch.randn(1, Z_DIM, 1, 1)
    fake = gen(fixed_noise)
    img_fake = torchvision.utils.make_grid(fake, normalize=True)
    plt.imshow(img_fake.permute(1, 2, 0))
    plt.show()

if __name__ == "__main__":
    #train()
    while True:
        genrate_image()