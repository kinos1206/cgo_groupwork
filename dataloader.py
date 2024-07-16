import torch
import torchvision
import torchvision.transforms as transforms


def load_MNIST(batch=128):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=2)

    val_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch, shuffle=True, num_workers=2)

    return {'train': train_loader, 'validation': val_loader}
