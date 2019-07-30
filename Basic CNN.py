import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 17)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.conv4(x)

        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def train(net, optimizer, loss_fn, num_epoch, data_loader, device='cpu'):
        net.train()

        for epoch in range(num_epoch):
            running_loss = 0.0
            for i, data in enumerate(data_loader):
                inputs, labels = data[0], data[1]

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                print('[%d, %5d] loss = %.3f' % (epoch+1, i+1, running_loss/100))

                running_loss = 0.0


def evaluate(net, data_loader, device='cpu'):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc


def main():
    transform = transforms.Compose([
        transforms.Resize([56, 56], interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_set = torchvision.datasets.ImageFolder(root='C:/Users/Adrian Chen/3D Objects/data/flowers/train', transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=32,
                                               shuffle=True)

    test_set = torchvision.datasets.ImageFolder(root='C:/Users/Adrian Chen/3D Objects/data/flowers/val', transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                               batch_size=32,
                                               shuffle=False)


    device = torch.device('cuda:0')

    net = CNN()
    # net.to(device)

    xentropy = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=0.001)
    num_epoch = 30

    train(net=net, optimizer=optimizer, loss_fn=xentropy, num_epoch=num_epoch, data_loader=train_loader)

    train_acc = evaluate(net=net, data_loader=train_loader)

    test_acc = evaluate(net=net, data_loader=test_loader)

    print('Train_acc: %.2f %%' % (100*train_acc))
    print('Test_acc: %.2f %%' % (100*test_acc))


if __name__ == '__main__':
    main()
