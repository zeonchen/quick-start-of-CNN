import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
import torchvision.transforms as transforms

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
    net = models.resnet50(pretrained=True) #download ResNet50

    for name, param in net.named_parameters():
        if name.startswith('fc'): #retrain fully connected layers
            param.requires_grad = True
            num_classes = 17
            featureSize = net.fc.in_features
            net.fc = nn.Linear(featureSize, num_classes)
        else: #keep others the same 
            param.requires_grad = False

    train_dir = './data/flowers/train'
    val_dir = './data/flowers/val'

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_set = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )

    test_set = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    device = torch.device('cuda:0')


    xentropy = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=0.001)
    num_epoch = 8

    train(net=net, optimizer=optimizer, loss_fn=xentropy, num_epoch=num_epoch, data_loader=train_loader)

    train_acc = evaluate(net=net, data_loader=train_loader)

    test_acc = evaluate(net=net, data_loader=test_loader)

    print('Train_acc: %.2f %%' % (100*train_acc))
    print('Test_acc: %.2f %%' % (100*test_acc))


if __name__ == '__main__':
    main()
