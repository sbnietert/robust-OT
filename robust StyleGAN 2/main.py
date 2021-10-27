"""
---
title: StyleGAN 2 Model Training
summary: >
 An annotated PyTorch implementation of StyleGAN2 model training code.
---

# [StyleGAN 2](index.html) Model Training

This is the training code for [StyleGAN 2](index.html) model.

![Generated Images](generated_64.png)

*<small>These are $64 \times 64$ images generated after training for about 80K steps.</small>*

*Our implementation is a minimalistic StyleGAN 2 model training code.
Only single GPU training is supported to keep the implementation simple.
We managed to shrink it to keep it at less than 500 lines of code, including the training loop.*

*Without DDP (distributed data parallel) and multi-gpu training it will not be possible to train the model
for large resolutions (128+).
If you want training code with fp16 and DDP take a look at
[lucidrains/stylegan2-pytorch](https://github.com/lucidrains/stylegan2-pytorch).*

We trained this on [CelebA-HQ dataset](https://github.com/tkarras/progressive_growing_of_gans).
You can find the download instruction in this
[discussion on fast.ai](https://forums.fast.ai/t/download-celeba-hq-dataset/45873/3).
Save the images inside [`data/stylegan` folder](#dataset_path).
"""

import math
from pathlib import Path
from typing import Iterator, Tuple
import datetime

import torch
import torch.utils.data
import torchvision
from PIL import Image

from labml_nn.gan.stylegan import Discriminator, Generator, MappingNetwork, GradientPenalty, PathLengthPenalty
from labml_nn.gan.wasserstein import DiscriminatorLoss, GeneratorLoss
from labml_nn.utils import cycle_dataloader

from torch.utils.tensorboard import SummaryWriter

class Dataset(torch.utils.data.Dataset):
    """
    ## Dataset

    This loads the training dataset and resize it to the give image size.
    """

    def __init__(self, path: str, image_size: int):
        """
        * `path` path to the folder containing the images
        * `image_size` size of the image
        """
        super().__init__()

        # Get the paths of all `jpg` and `png` files
        self.paths = [p for p in Path(path).glob(f'**/*.jpg')] + [p for p in Path(path).glob(f'**/*.png')]

        # Transformation
        self.transform = torchvision.transforms.Compose([
            # Resize the image
            torchvision.transforms.Resize(image_size),
            # Convert to PyTorch tensor
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        """Number of images"""
        return len(self.paths)

    def __getitem__(self, index):
        """Get the the `index`-th image"""
        path = self.paths[index]
        img = Image.open(path).convert('RGB') # make sure even monochrome images have 3 channels
        return self.transform(img)


class Experiment:
    """
    ## Configurations
    """

    # Device to train the model on.
    device: torch.device = torch.device('cuda')

    # [StyleGAN2 Discriminator](index.html#discriminator)
    discriminator: Discriminator
    # [StyleGAN2 Generator](index.html#generator)
    generator: Generator
    # [Mapping network](index.html#mapping_network)
    mapping_network: MappingNetwork

    # Discriminator and generator loss functions.
    # We use [Wasserstein loss](../wasserstein/index.html)
    discriminator_loss: DiscriminatorLoss
    generator_loss: GeneratorLoss
    discriminator_relu: bool

    # Optimizers
    generator_optimizer: torch.optim.Adam
    discriminator_optimizer: torch.optim.Adam
    mapping_network_optimizer: torch.optim.Adam

    # [Gradient Penalty Regularization Loss](index.html#gradient_penalty)
    gradient_penalty = GradientPenalty()
    # Gradient penalty coefficient $\gamma$
    gradient_penalty_coefficient: float = 10.

    # [Path length penalty](index.html#path_length_penalty)
    path_length_penalty: PathLengthPenalty

    # Data loader
    loader: Iterator

    # Batch size
    batch_size: int = 32
    # Dimensionality of $z$ and $w$
    d_latent: int = 512
    # Height/width of the image
    image_size: int = 32
    # Number of layers in the mapping network
    mapping_network_layers: int = 8
    # Generator & Discriminator learning rate
    learning_rate: float = 1e-3
    # Mapping network learning rate ($100 \times$ lower than the others)
    mapping_network_learning_rate: float = 1e-5
    # Number of steps to accumulate gradients on. Use this to increase the effective batch size.
    gradient_accumulate_steps: int = 1
    # $\beta_1$ and $\beta_2$ for Adam optimizer
    adam_betas: Tuple[float, float] = (0.0, 0.99)
    # Probability of mixing styles
    style_mixing_prob: float = 0.9

    # Total number of training steps
    training_steps: int = 150_000

    # Number of blocks in the generator (calculated based on image resolution)
    n_gen_blocks: int

    # ### Lazy regularization
    # Instead of calculating the regularization losses, the paper proposes lazy regularization
    # where the regularization terms are calculated once in a while.
    # This improves the training efficiency a lot.

    # The interval at which to compute gradient penalty
    lazy_gradient_penalty_interval: int = 4
    # Path length penalty calculation interval
    lazy_path_penalty_interval: int = 32
    # Skip calculating path length penalty during the initial phase of training
    lazy_path_penalty_after: int = 5_000

    # How often to log generated images
    log_generated_interval: int = 500
    # How often to save model checkpoints
    save_checkpoint_interval: int = 2_000

    # <a id="dataset_path"></a>
    # We trained this on [CelebA-HQ dataset](https://github.com/tkarras/progressive_growing_of_gans).
    # You can find the download instruction in this
    # [discussion on fast.ai](https://forums.fast.ai/t/download-celeba-hq-dataset/45873/3).
    # Save the images inside `data/stylegan` folder.
    clean_dataset_path: str = str('./data/CelebA-HQ')
    corrupt_dataset_path: str = str('./data/MNIST')
    corruption_fraction: float
    range_penalty_delta: float

    tb: SummaryWriter
    name: str = 'run'

    def __init__(self, custom_configs):
        self.__dict__.update(custom_configs)

        # Start TensorBoard
        self.name += '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb = SummaryWriter(log_dir='/data/robust-stylegan2-v2/runs/' + self.name)
        self.tb.add_text('custom.configs', str(custom_configs))

    def init(self):
        """
        ### Initialize
        """
        # Create dataset
        clean_dataset = Dataset(self.clean_dataset_path, self.image_size)
        corrupt_dataset = Dataset(self.corrupt_dataset_path, self.image_size)
        num_corrupted_images = int(len(clean_dataset) * self.corruption_fraction / (1 - self.corruption_fraction))
        assert(num_corrupted_images <= len(corrupt_dataset))
        corrupt_subset = torch.randperm(len(corrupt_dataset))[:num_corrupted_images]
        corrupt_subset_dataset = torch.utils.data.Subset(corrupt_dataset, corrupt_subset)
        combined_dataset = torch.utils.data.ConcatDataset([clean_dataset, corrupt_subset_dataset])

        # Create data loader
        dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=self.batch_size, num_workers=32,
                                                 shuffle=True, drop_last=True, pin_memory=True)
        # Continuous [cyclic loader](../../utils.html#cycle_dataloader)
        self.loader = cycle_dataloader(dataloader)

        # $\log_2$ of image resolution
        log_resolution = int(math.log2(self.image_size))

        # Create discriminator and generator
        self.discriminator = Discriminator(log_resolution).to(self.device)
        self.generator = Generator(log_resolution, self.d_latent).to(self.device)
        # Get number of generator blocks for creating style and noise inputs
        self.n_gen_blocks = self.generator.n_blocks
        # Create mapping network
        self.mapping_network = MappingNetwork(self.d_latent, self.mapping_network_layers).to(self.device)
        # Create path length penalty loss
        self.path_length_penalty = PathLengthPenalty(0.99).to(self.device)

        # Discriminator and generator losses
        self.discriminator_loss = DiscriminatorLoss().to(self.device)
        self.generator_loss = GeneratorLoss().to(self.device)

        self.discriminator_relu = False

        # Create optimizers
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate, betas=self.adam_betas
        )
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate, betas=self.adam_betas
        )
        self.mapping_network_optimizer = torch.optim.Adam(
            self.mapping_network.parameters(),
            lr=self.mapping_network_learning_rate, betas=self.adam_betas
        )

    def get_w(self, batch_size: int):
        """
        ### Sample $w$

        This samples $z$ randomly and get $w$ from the mapping network.

        We also apply style mixing sometimes where we generate two latent variables
        $z_1$ and $z_2$ and get corresponding $w_1$ and $w_2$.
        Then we randomly sample a cross-over point and apply $w_1$ to
        the generator blocks before the cross-over point and
        $w_2$ to the blocks after.
        """

        # Mix styles
        if torch.rand(()).item() < self.style_mixing_prob:
            # Random cross-over point
            cross_over_point = int(torch.rand(()).item() * self.n_gen_blocks)
            # Sample $z_1$ and $z_2$
            z2 = torch.randn(batch_size, self.d_latent).to(self.device)
            z1 = torch.randn(batch_size, self.d_latent).to(self.device)
            # Get $w_1$ and $w_2$
            w1 = self.mapping_network(z1)
            w2 = self.mapping_network(z2)
            # Expand $w_1$ and $w_2$ for the generator blocks and concatenate
            w1 = w1[None, :, :].expand(cross_over_point, -1, -1)
            w2 = w2[None, :, :].expand(self.n_gen_blocks - cross_over_point, -1, -1)
            return torch.cat((w1, w2), dim=0)
        # Without mixing
        else:
            # Sample $z$ and $z$
            z = torch.randn(batch_size, self.d_latent).to(self.device)
            # Get $w$ and $w$
            w = self.mapping_network(z)
            # Expand $w$ for the generator blocks
            return w[None, :, :].expand(self.n_gen_blocks, -1, -1)

    def get_noise(self, batch_size: int):
        """
        ### Generate noise

        This generates noise for each [generator block](index.html#generator_block)
        """
        # List to store noise
        noise = []
        # Noise resolution starts from $4$
        resolution = 4

        # Generate noise for each generator block
        for i in range(self.n_gen_blocks):
            # The first block has only one $3 \times 3$ convolution
            if i == 0:
                n1 = None
            # Generate noise to add after the first convolution layer
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)
            # Generate noise to add after the second convolution layer
            n2 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)

            # Add noise tensors to the list
            noise.append((n1, n2))

            # Next block has $2 \times$ resolution
            resolution *= 2

        # Return noise tensors
        return noise

    def generate_images(self, batch_size: int):
        """
        ### Generate images

        This generate images using the generator
        """

        # Get $w$
        w = self.get_w(batch_size)
        # Get noise
        noise = self.get_noise(batch_size)

        # Generate images
        images = self.generator(w, noise)

        # Return images and $w$
        return images, w

    def step(self, idx: int):
        """
        ### Training Step
        """

        # Train the discriminator

        # Reset gradients
        self.discriminator_optimizer.zero_grad()

        # Accumulate gradients for `gradient_accumulate_steps`
        for i in range(self.gradient_accumulate_steps):
            log_losses = (i == self.gradient_accumulate_steps-1)

            # Sample images from generator
            generated_images, _ = self.generate_images(self.batch_size)
            # Discriminator classification for generated images
            fake_output = self.discriminator(generated_images.detach())

            # Get real images from the data loader
            real_images = next(self.loader).to(self.device)
            # We need to calculate gradients w.r.t. real images for gradient penalty
            if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                real_images.requires_grad_()
            # Discriminator classification for real images
            real_output = self.discriminator(real_images)

            # Get discriminator loss
            real_loss, fake_loss = self.discriminator_loss(real_output, fake_output, self.discriminator_relu)
            disc_loss = real_loss + fake_loss

            # Add gradient penalty
            if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                # Calculate and log gradient penalty
                gp = self.gradient_penalty(real_images, real_output)
                if log_losses:
                    self.tb.add_scalar('loss.gp', gp, idx)
                # Multiply by coefficient and add gradient penalty
                disc_loss = disc_loss + 0.5 * self.gradient_penalty_coefficient * gp * self.lazy_gradient_penalty_interval

            # Add range penalty
            if self.range_penalty_delta > 0:
                range_penalty = self.range_penalty_delta*(real_output.max() - fake_output.min())
                disc_loss += range_penalty

            # Compute gradients
            disc_loss.backward()

            # Log discriminator loss
            if log_losses:
                self.tb.add_scalar('loss.discriminator', disc_loss, idx)

        if (idx + 1) % self.log_generated_interval == 0:
            # Log discriminator model parameters occasionally
            #tracker.add('discriminator', self.discriminator)
            pass

        # Clip gradients for stabilization
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        # Take optimizer step
        self.discriminator_optimizer.step()

        # Train the generator

        # Reset gradients
        self.generator_optimizer.zero_grad()
        self.mapping_network_optimizer.zero_grad()

        # Accumulate gradients for `gradient_accumulate_steps`
        for i in range(self.gradient_accumulate_steps):
            # Sample images from generator
            generated_images, w = self.generate_images(self.batch_size)
            # Discriminator classification for generated images
            fake_output = self.discriminator(generated_images)

            # Get generator loss
            gen_loss = self.generator_loss(fake_output)

            # Add path length penalty
            if idx > self.lazy_path_penalty_after and (idx + 1) % self.lazy_path_penalty_interval == 0:
                # Calculate path length penalty
                plp = self.path_length_penalty(w, generated_images)
                # Ignore if `nan`
                if not torch.isnan(plp):
                    self.tb.add_scalar('loss.plp', plp, idx)
                    gen_loss = gen_loss + plp

            # Calculate gradients
            gen_loss.backward()

            # Log generator loss
            if i == self.gradient_accumulate_steps-1:
                self.tb.add_scalar('loss.generator', gen_loss, idx)

        if (idx + 1) % self.log_generated_interval == 0:
            # Log discriminator model parameters occasionally
            #tracker.add('generator', self.generator)
            #tracker.add('mapping_network', self.mapping_network)
            pass

        # Clip gradients for stabilization
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.mapping_network.parameters(), max_norm=1.0)

        # Take optimizer step
        self.generator_optimizer.step()
        self.mapping_network_optimizer.step()

        # Log generated images
        if (idx + 1) % self.log_generated_interval == 0:
            self.tb.add_images('generated', generated_images[:25], idx)

        # Save model checkpoints
        if (idx + 1) % self.save_checkpoint_interval == 0:
            self.save_checkpoint(idx, gen_loss, disc_loss)
            pass

        # Flush tracker
        self.tb.flush()

    def train(self):
        """
        ## Train model
        """

        # Loop for `training_steps`
        for i in range(self.training_steps):
            # Take a training step
            self.step(i)

    def save_checkpoint(self, step, gen_loss, disc_loss):
        torch.save({
            'step': step,
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_state_dict': self.generator.state_dict(),
            'mapping_network_state_dict': self.mapping_network.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'mapping_network_optimizer_state_dict': self.mapping_network_optimizer.state_dict(),
            'generator_loss': gen_loss,
            'discriminator_loss': disc_loss
        }, self.tb.log_dir + '/checkpoint_step_' + str(step) + '.pt')

def main():
    """
    ### Train StyleGAN2
    """
    print('See TensorBoard for progress')

    # Create an experiment
    experiment = Experiment({
        'name': 'corrupted_no_penalty_32_v2',
        'image_size': 32,
        'log_generated_interval': 200,
        'discriminator_relu': False,
        'corruption_fraction': 0.2,
        'range_penalty_delta': 0#0.25
    })

    # Initialize
    experiment.init()

    # Train
    experiment.train()

#
if __name__ == '__main__':
    main()
