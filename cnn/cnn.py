from torchvision import datasets, transforms
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import sys
config = {
    'batch_size': 64,
    'test_batch_size': 1000,
    'epochs': 10,
    'learning_rate': 0.01,
    'gamma': 0.7,
    'seed': 0,
    'save_model': True
}

def load_data():
    """
    Load MNIST data from Pytorch dataset. 
    """
    generator = torch.Generator().manual_seed(config['seed'])   # For random number. 
    train_args = {
        'batch_size': config['batch_size'], 'shuffle': True, 'generator': generator
    }
    test_args = {
        'batch_size': config['test_batch_size'], 'shuffle': True, 'generator': generator
    }
    transform = transforms.Compose([  # transform the data
        transforms.ToTensor(),  # transform to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # normalize
    ])
    # download the dataset
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    # add the config to the dataloader
    train_loader = torch.utils.data.DataLoader(dataset1, **train_args)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_args)
    return train_loader, test_loader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(9216, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    total_loss = 0.0
    correct = 0.0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()   # Gradient to 0.
        output = model(data)
        loss = F.nll_loss(output, target)
        total_loss += loss.item() * len(data)
        predicted = torch.max(output, 1)[1]
        correct += (predicted == target).sum().item()
        loss.backward()
        optimizer.step()
        sys.stdout.write(f"\r {epoch}: {batch_idx} / {int(len(train_loader.dataset) / config['batch_size'])}")
        sys.stdout.flush()
    train_acc, train_loss = correct / len(train_loader.dataset), total_loss / len(train_loader.dataset)
    print("")
    return train_acc, train_loss


def test(args, model, device, test_loader):
    pass
    model.eval()
    test_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item() * len(data)
            predicted = output.max(1, keepdim=True)[1]
            correct += predicted.eq(target.view_as(predicted)).sum().item()
            sys.stdout.write(f"\r {epoch}: {batch_idx} / {int(len(test_loader.dataset) / config['test_batch_size'])}")
            sys.stdout.flush()
        test_loss /= len(test_loader.dataset)
        correct = correct / len(test_loader.dataset)
        print("")
    return correct, test_loss


if __name__ == '__main__':
    use_cuda = False
    use_mps = False
    train_loader, test_loader = load_data()
    torch.manual_seed(config['seed'])
    device = torch.device("cpu")
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config['learning_rate'])
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, config['epochs'] + 1):
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch)
        print(f"Train accuracy: {train_acc}, train loss: {train_loss}")
        test_acc, test_loss = test(config, model, device, test_loader)
        print(f"Test accuracy: {test_acc}, test loss: {test_loss}")
    print("Process done.")