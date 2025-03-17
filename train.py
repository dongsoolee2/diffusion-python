import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from datasets import load_dataset
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

image_size = 96
batch_size = 64
num_epoch = 30

def show_images(x):
    x = x * 0.5 + 0.5
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarry(np.array(grid_im).astype(np.uint8))
    return grid_im

def make_grid(images, size=image_size):
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im

def transform(examples, image_size=image_size):
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

def get_model(image_size=image_size):
    model = UNet2DModel(
        sample_size=image_size,  
        in_channels=3,  
        out_channels=3,  
        layers_per_block=2, 
        block_out_channels=(64, 128, 128, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model

def train(dataset=dataset, device=device, batch_size=batch_size, num_epoch=num_epoch):
    dataset.set_transform(transform)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2"
    )

    model = get_model()
    model.to(device);

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

    losses = []

    for epoch in range(num_epoch):
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"].to(device)
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            loss = F.mse_loss(noise_pred, noise)
            loss.backward(loss)
            losses.append(loss.item())

            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 5 == 0:
            loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
            print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")
    
    return losses, model, noise_scheduler