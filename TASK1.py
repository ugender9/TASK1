# Step-1 : Import necessary libraries

import torch

import torch.nn as nn

import torch.optim as optim

import torchvision

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import numpy as np

 

# Set device

device = torch.device(‘cuda’ if torch.cuda.is_available() else ‘cpu’)

 

# Step-2: Define data transformations

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])

 

# Step-3: Load and preprocess the dataset

train_dataset = datasets.CIFAR10(root=’./data’, train=True, download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

 

# Step-4: Set up model parameters

latent_dims = 100

learning_rate = 0.0002

beta_1 = 0.5

beta_2 = 0.999

num_epochs = 5

 

# Step-5: Create Generator class

class Generator(nn.Module):

    def _init_(self, latent_dims):

        super(Generator, self)._init_()

 

        self.model = nn.Sequential(

            nn.Linear(latent_dims, 128 * 8 * 8),

            nn.ReLU(),

            nn.Unflatten(1, (128, 8, 8)),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),

            nn.BatchNorm2d(128, momentum=0.78),

            nn.ReLU(),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),

            nn.BatchNorm2d(64, momentum=0.78),

            nn.ReLU(),

            nn.Conv2d(64, 3, kernel_size=3, padding=1),

            nn.Tanh()

        )

 

    def forward(self, z):

        img = self.model(z)

        return img

 

# Step-6: Create Discriminator class

class Discriminator(nn.Module):

    def _init_(self):

        super(Discriminator, self)._init_()

 

        self.model = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),

            nn.LeakyReLU(0.2),

            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),

            nn.ZeroPad2d((0, 1, 0, 1)),

            nn.BatchNorm2d(64, momentum=0.82),

            nn.LeakyReLU(0.25),

            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),

            nn.BatchNorm2d(128, momentum=0.82),

            nn.LeakyReLU(0.2),

            nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(256, momentum=0.8),

            nn.LeakyReLU(0.25),

            nn.Dropout(0.25),

            nn.Flatten(),

            nn.Linear(256 * 5 * 5, 1),

            nn.Sigmoid()

        )

 

    def forward(self, img):

        validity = self.model(img)

        return validity

 

# Step-7: Build the Generative Adversarial Network architecture

generator = Generator(latent_dims).to(device)

discriminator = Discriminator().to(device)

 

# Loss function

adversarial_loss = nn.BCELoss()

 

# Optimizers

optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta_1, beta_2))

optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta_1, beta_2))

 

# Step-8: Train the GAN model

# Training loop

for epoch in range(num_epochs):

    for i, batch in enumerate(dataloader):

        # Convert list to tensor

        real_images = batch[0].to(device)

        # Adversarial ground truths

        valid = torch.ones(real_images.size(0), 1, device=device)

        fake = torch.zeros(real_images.size(0), 1, device=device)

        # Configure input

        real_images = real_images.to(device)

 

        # ———————

        # Train Discriminator

        # ———————

        optimizer_D.zero_grad()

        # Sample noise as generator input

        z = torch.randn(real_images.size(0), latent_dims, device=device)

        # Generate a batch of images

        fake_images = generator(z)

 

        # Measure discriminator’s ability

        real_loss = adversarial_loss(discriminator(real_images), valid)

        fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake)

        d_loss = (real_loss + fake_loss) / 2

        # Backward pass and optimize

        d_loss.backward()

        optimizer_D.step()

 

        # —————–

        # Train Generator

        # —————–

        optimizer_G.zero_grad()

        # Generate a batch of images

        gen_images = generator(z)

        # Adversarial loss

        g_loss = adversarial_loss(discriminator(gen_images), valid)

        # Backward pass and optimize

        g_loss.backward()

        optimizer_G.step()

        # ———————

        # Progress Monitoring

        # ———————

        if (i + 1) % 100 == 0:

            print(

                f”Epoch [{epoch+1}/{num_epochs}] Batch {i+1}/{len(dataloader)} “

                f”Discriminator Loss: {d_loss.item():.4f} “

                f”Generator Loss: {g_loss.item():.4f}”

            )

    # Save

    if (epoch + 1) % 5 == 0:

        with torch.no_grad():

            z = torch.randn(16, latent_dims, device=device)

            generated = generator(z).detach().cpu()

            grid = torchvision.utils.make_grid(generated, nrow=4, normalize=True)

            plt.imshow(np.transpose(grid, (1, 2, 0)))

            plt.axis(“off”)

            plt.show()
