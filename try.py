import torch.utils.data as data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.v2 as tfs
import torch.optim as optim
from tqdm import tqdm


# Создание класса для загрузки данных и преобразования их в тензоры
class CustomDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.data = pd.read_csv(path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]  # Получаем строку с индексом idx
        label = row.iloc[0]  # Целевая переменная
        pixels = row.iloc[1:].values.astype(np.uint8).reshape(50, 50)  # Пиксели изображения
        img = Image.fromarray(pixels)  # Создаем изображение из пикселей

        if self.transform:
            img = self.transform(img)

        return img, label


transform = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.5], std=[0.5]),
    tfs.ToDtype(torch.float32, scale=True)])

d_train = CustomDataset('trainnn.csv', transform=transform)
print(d_train.__len__())
img, label = d_train[0]

print(f'label: {label}')
print(f'img shape: {img.shape}')

plt.imshow(img.squeeze(), cmap='gray')
plt.title('label:')
plt.axis('off')
plt.show()

dataloader = data.DataLoader(d_train, batch_size=32, shuffle=True)
for images, labels in dataloader:
    print(images.shape)
    print(labels.shape)
    print(labels)
    break

model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Flatten(),

    nn.Linear(128 * 6 * 6, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 37)
)

optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)
loss_fn = nn.CrossEntropyLoss()
epochs = 3

model.train()

for epoch in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(dataloader, leave=False)
    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_fn(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f'Epoch: {epoch + 1}, loss: {loss_mean:.4f}')


st = model.state_dict()
torch.save(st, 'model.pth')

d_test = CustomDataset('testnn.csv', train=False, transform=transform)
test_data = data.DataLoader(d_test, batch_size=32, shuffle=False)

correct = 0
total = 0
model.eval()
test_tqdm = tqdm(test_data, leave=True)
for x_test, y_test in test_tqdm:
    with torch.no_grad():
        p = model(x_test)
        _, predicted = torch.max(p, 1)
        total += y_test.size(0)
        correct += (predicted == y_test).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

test_dataset = CustomDataset('test.csv', train=False, transform=transform)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)
predictions = []

with torch.no_grad():
    for x_test, _ in test_loader:
        x_test = x_test.to(device)
        p = model(x_test)
        _, predicted = torch.max(p, 1)
        predictions.extend(predicted.cpu().numpy())


print(predictions)
