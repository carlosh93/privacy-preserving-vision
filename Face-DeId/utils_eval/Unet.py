import torch
import torch.nn as nn
from ldms_box_metrics import get_eval_loader


# Define the U-Net architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)
        x2 = self.middle(x1)

        # Decoder
        x3 = self.decoder(x2)

        return x3


import torch
import torch.nn as nn
import torch.optim as optim


# Define a function to train the U-Net model
def train_unet(model, num_epochs, learning_rate, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.SmoothL1Loss()  # You can change the loss function as per your task
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = get_eval_loader(root=args.val_img_dir + '/male/', img_size=args.img_size, batch_size=args.val_batch_size, imagenet_normalize=False)

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for org in train_loader:
            org = org.to(device)

            inputs = F.interpolate(org, (16, 16))
            inputs = F.interpolate(inputs, (64, 64))
            targets = F.interpolate(org, (64, 64))

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, targets) * 100

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch%20 == 0:
            plt.subplot(131), plt.imshow(targets[0].permute(1, 2, 0).cpu().data)
            plt.subplot(132), plt.imshow(outputs[0].permute(1, 2, 0).cpu().data)
            plt.subplot(133), plt.imshow(inputs[0].permute(1, 2, 0).cpu().data), plt.show()

        # Print the average loss for the epoch
        average_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {average_loss:.6f}")

    print("Training complete")