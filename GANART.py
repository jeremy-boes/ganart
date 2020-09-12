from IPython import display

import copy
import random
import torch
import time
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

from ganart.utils import Logger

# Parameters

# device = torch.device('cuda:0')
device = torch.device('cpu')

data_path = '/path/to/data'
error_file_path = '/path/to/csv/to/save/loss/and/accuracy/of/functions'

batch_size = 32
resize_value = 120

num_test_samples = 4
num_epochs = 400
num_rolls = 5

# Create the discriminator
kernel_s = (4, 4)
stride_s = (2, 2)

# Code to create discriminator and generator. This actually should be later in the work flow, but since it's a common
# element to manipulate, it was placed here to reduce scrolling/searching.
class DiscriminatorNet(torch.nn.Module):

    def __init__(self):
        super(DiscriminatorNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=kernel_s,
                stride=stride_s, padding=(1, 1), bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=kernel_s,
                stride=stride_s, padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=kernel_s,
                stride=stride_s, padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=kernel_s,
                stride=stride_s, padding=(1, 1), bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(1024 * 7 * 10, 1),
            # nn.Linear(49152, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten and apply sigmoid
        x = x.view(-1, 1024 * 7 * 10)
        # x = x.view(-1, 49152)
        x = self.out(x)
        return x


# Create generator
class GeneratorNet(torch.nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()

        self.linear = torch.nn.Linear(100, 1024 * 7 * 10)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=(6, 4),
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=3, kernel_size=(8, 4),
                stride=2, padding=1, bias=False
            )
        )
        self.out = torch.nn.Tanh()

    def forward(self, x):
        # Project and reshape
        x = self.linear(x)
        x = x.view(x.shape[0], 1024, 7, 10)
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Apply Tanh
        return self.out(x)


# Create data set
def createDataSet():
    compose = transforms.Compose(
        [transforms.Resize(resize_value),
         transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))
         ])
    train_dataset = datasets.ImageFolder(root=data_path, transform=compose)
    return train_dataset
    # return datasets.MNIST(root='./dataset', train=True, transform=compose, download=True)


# Create noise
def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n


# Create target data sets that are for a thing...
def ones_target(size):
    data = Variable(torch.ones(size, 1))
    data = data.new_full(data.size(), random.uniform(0.9, 1))
    return data.to(device)


def zeros_target(size):
    data = Variable(torch.zeros(size, 1))
    return data.to(device)


# Train the discriminator and generator
def train_discriminator(optimizer, discriminator, real_data, fake_data):
    N = real_data.size(0)
    real_data.to(device)
    fake_data.to(device)
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    # prediction_real = discriminator(real_data.cuda())
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, discriminator, fake_data):
    N = fake_data.size(0)
    fake_data.to(device)
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error


# Load data
data = createDataSet()
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
num_batches = len(data_loader)
print("num batches: " + str(num_batches))

discriminator = DiscriminatorNet()
discriminator.to(device)

generator = GeneratorNet()
generator.to(device)

# Create optimiziers
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

loss = nn.BCELoss()

# Set up noise and logger
test_noise = noise(num_test_samples)

logger = Logger(model_name='DConv-GConv-GANART', data_name='Lands')

with open(error_file_path, 'w') as error_file:
    error_file.write('Epoch,D_error,G_error,D_pred_real,D_pred_fake\n')
    error_file.flush()
    # Do the run of the code
    start = time.time()
    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch))
        print("Elapsed Time: " + str(time.time() - start))
        start = time.time()
        for n_batch, (real_batch, _) in enumerate(data_loader):

            # 1. Train Discriminator
            real_data = Variable(real_batch)
            fake_data = generator(noise(real_data.size(0))).detach()
            d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, discriminator, real_data, fake_data)

            # 2. Train Generator
            # Unroll D
            discriminator_copy = copy.deepcopy(discriminator)
            data_loader_copy = copy.deepcopy(data_loader)
            iterator = iter(data_loader_copy)
            for k in range(num_rolls - 1):
                if (n_batch + k < num_batches):
                    real_data = Variable(next(iterator)[0])
                    fake_data = generator(noise(real_data.size(0))).detach()
                    train_discriminator(d_optimizer, discriminator_copy, real_data, fake_data)
            # Train G
            fake_data = generator(noise(real_batch.size(0)))
            g_error = train_generator(g_optimizer, discriminator_copy, fake_data)

            # 3. Log
            logger.log(d_error, g_error, epoch, n_batch, num_batches)

            # Display Progress
            if n_batch % 50 == 0:
                display.clear_output(True)
                # Display Images
                test_images = generator(test_noise).data.cpu()
                logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
                # Display status Logs
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
                )
                error_file.write('{},{},{},{:.4f},{:.4f}\n'.format(str(epoch), str(d_error.data.cpu().numpy()),
                                                                str(g_error.data.cpu().numpy()), d_pred_real.data.mean(),
                                                                d_pred_fake.data.mean()))
                error_file.flush()
            if epoch % 50 == 0:
                # Model Checkpoints
                logger.save_models(generator, discriminator, epoch)
logger.save_models(generator, discriminator, epoch)
logger.close()
