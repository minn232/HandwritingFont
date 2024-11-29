import os
import torch
import torch.nn as nn
from torchvision.utils import save_image

# Import from modules
from common.dataset import TrainDataProvider, save_fixed_sample
from common.models import Encoder, Decoder, Discriminator, Generator
from common.function import init_embedding, embedding_lookup, batch_norm, conv2d, deconv2d

# Define the training process
class FontTrainer:
    def __init__(self, data_dir, save_dir, batch_size=32, img_size=128, lr=0.001, GPU=False):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.lr = lr
        self.GPU = GPU

        # Load data provider
        self.data_provider = TrainDataProvider(data_dir)
        self.total_batches = self.data_provider.compute_total_batch_num(batch_size)
        print(f"Total batches: {self.total_batches}")

        # Initialize models
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = Discriminator(category_num=10)  # Adjust font category as needed

        # Load models to GPU if required
        if GPU:
            self.encoder.cuda()
            self.decoder.cuda()
            self.discriminator.cuda()

        # Define loss functions
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

        # Optimizers
        self.g_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr, betas=(0.5, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def train(self, max_epochs=150, save_interval=5):
        for epoch in range(max_epochs):
            batch_iter = self.data_provider.get_train_iter(self.batch_size)
            for i, batch in enumerate(batch_iter):
                font_ids, batch_images = batch
                if self.GPU:
                    batch_images = batch_images.cuda()

                # Split images into source and target
                real_source = batch_images[:, 1, :, :].unsqueeze(1)
                real_target = batch_images[:, 0, :, :].unsqueeze(1)

                # Generate fake images
                embeddings = init_embedding(embedding_num=10, embedding_dim=512)  # Adjust as needed
                fake_target, encoded_source, _ = Generator(real_source, self.encoder, self.decoder, embeddings, font_ids)

                # Discriminator training
                real_output, _, _ = self.discriminator(torch.cat([real_source, real_target], dim=1))
                fake_output, _, _ = self.discriminator(torch.cat([real_source, fake_target], dim=1))
                d_loss = self.bce_loss(real_output, torch.ones_like(real_output)) + \
                         self.bce_loss(fake_output, torch.zeros_like(fake_output))

                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Generator training
                g_loss = self.l1_loss(real_target, fake_target) + \
                         self.bce_loss(fake_output, torch.ones_like(fake_output))

                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging
                if (i + 1) % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{max_epochs}], Step [{i + 1}/{self.total_batches}], "
                          f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

            # Save models at intervals
            if (epoch + 1) % save_interval == 0:
                torch.save(self.encoder.state_dict(), os.path.join(self.save_dir, f"encoder_epoch_{epoch + 1}.pth"))
                torch.save(self.decoder.state_dict(), os.path.join(self.save_dir, f"decoder_epoch_{epoch + 1}.pth"))
                torch.save(self.discriminator.state_dict(), os.path.join(self.save_dir, f"discriminator_epoch_{epoch + 1}.pth"))

if __name__ == "__main__":
    trainer = FontTrainer(data_dir="data/fonts", save_dir="models", batch_size=16, img_size=128, GPU=True)
    trainer.train(max_epochs=150)
