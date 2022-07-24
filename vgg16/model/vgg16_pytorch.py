import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
# print(DEVICE)

EPOCHS = 100
BATCH_SIZE = 100

class VGG16(torch.nn.Module):

    def __init__(self, n_classes):
        super(VGG16, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv11_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv12_bn = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv21_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv22_bn = nn.BatchNorm2d(128)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv31_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv32_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv33_bn = nn.BatchNorm2d(256)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv41_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv42_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv43_bn = nn.BatchNorm2d(512)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv51_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv52_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv53_bn = nn.BatchNorm2d(512)
        self.fc7 = nn.Linear(512, 4096)
        self.fc8 = nn.Linear(4096, 4096)
        self.fc9 = nn.Linear(4096, 1000)
        self.fc10 = nn.Linear(1000, n_classes)

    def forward(self, x):
        x = F.relu(self.conv11_bn(self.conv1_1(x)))
        x = F.relu(self.conv12_bn(self.conv1_2(x)))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv22_bn(self.conv2_1(x)))
        x = F.relu(self.conv21_bn(self.conv2_2(x)))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv31_bn(self.conv3_1(x)))
        x = F.relu(self.conv32_bn(self.conv3_2(x)))
        x = F.relu(self.conv33_bn(self.conv3_3(x)))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv41_bn(self.conv4_1(x)))
        x = F.relu(self.conv42_bn(self.conv4_2(x)))
        x = F.relu(self.conv43_bn(self.conv4_3(x)))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv51_bn(self.conv5_1(x)))
        x = F.relu(self.conv52_bn(self.conv5_2(x)))
        x = F.relu(self.conv53_bn(self.conv5_3(x)))
        x = F.max_pool2d(x, (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = self.fc10(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

model = VGG16(n_classes = 10).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # 배치 오차를 합산
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()

            # 가장 높은 값을 가진 인덱스가 바로 예측값
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


# ## 코드 돌려보기
# 자, 이제 모든 준비가 끝났습니다. 코드를 돌려서 실제로 학습이 되는지 확인해봅시다!
if __name__ == "__main__" :

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./.data',
                       train=True,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./.data',
                       train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, epoch)
        test_loss, test_accuracy = evaluate(model, test_loader)

        print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
            epoch, test_loss, test_accuracy))
