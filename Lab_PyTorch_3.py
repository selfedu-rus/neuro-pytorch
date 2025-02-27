import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary


# Приведенные ниже две строки являются необязательными и предназначены только для того, чтобы избежать использования SSL
# и связанные с этим ошибки при загрузке набора данных CIFAR-10
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Установка настроек для вывода данных
plt.rcParams['figure.figsize'] = 14, 6

# Нармализация датасета
normalize_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))])

# Загрузка датасета CIFAR10 и формирование обучающего и тестового датасета
train_dataset = torchvision.datasets.CIFAR10(
    root="./CIFAR10/train", train=True,
    transform=normalize_transform,
    download=True)

test_dataset = torchvision.datasets.CIFAR10(
    root="./CIFAR10/test", train=False,
    transform=normalize_transform,
    download=True)

# Формирование загрузчиков данных для дальнейшей работы из обучающей и выборки
batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)



#Вывод на экран 25 изображений из первого батча
dataiter = iter(train_loader)
images, labels = next(dataiter)
plt.imshow(np.transpose(torchvision.utils.make_grid(images[:25], normalize=True, padding=1, nrow=5).numpy(), (1, 2, 0)))
plt.show()

classes = []
for batch_idx, data in enumerate(train_loader, 0):
    x, y = data
    classes.extend(y.tolist())

# Вычисление уникальных классов и соответствующих им значений, визуализация результата
unique, counts = np.unique(classes, return_counts=True)
names = list(test_dataset.class_to_idx.keys())
plt.bar(names, counts)
plt.xlabel("Уникальные классы")
plt.ylabel("Количество изображений в тестовой выборке")
plt.show()


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            # Вход = 3 x 32 x 32, Выход = 32 x 32 x 32
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # Вход = 32 x 32 x 32, Выход = 32 x 16 x 16
            torch.nn.MaxPool2d(kernel_size=2),

            # Вход = 32 x 16 x 16, Выход = 64 x 16 x 16
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # Вход = 64 x 16 x 16, Выход = 64 x 8 x 8
            torch.nn.MaxPool2d(kernel_size=2),

            # Вход = 64 x 8 x 8, Выход = 64 x 8 x 8
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # Вход = 64 x 8 x 8, Выход = 64 x 4 x 4
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.model(x)


# Выбор платформы для обучения модели
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN().to(device)

# Определение гиперпараметров модели
num_epochs = 1
learning_rate = 0.001
weight_decay = 0.01
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Начинаем процесс обучения
train_loss_list = []
for epoch in range(num_epochs):
    print(f'Эпоха {epoch + 1}/{num_epochs}:', end=' ')
    train_loss = 0

# Работа с обучающим пакетом данных
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # Извлечение изображений и целевых меток для батча
        images = images.to(device)
        labels = labels.to(device)

        # Вычисление выходных данных модели и перекрестных потерь энтропии
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Обновление весов в соответствии с рассчитанными потерями
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Вывод значения функции потерь за каждую эпоху
    train_loss_list.append(train_loss / len(train_loader))
    print(f"Значение функции потерь = {train_loss_list[-1]}")

# Построение гарфика протерь для всех эпох обучения
plt.plot(range(1, num_epochs + 1), train_loss_list)
plt.xlabel("Номер эпохи")
plt.ylabel("Значение функции потерь")
plt.show()

test_acc = 0
model.eval()

with torch.no_grad():
    # Работа с обучающим пакетом данных
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        y_true = labels.to(device)

        # Вычисление прогноза модели для батча
        outputs = model(images)

        # Вычисленные метки прогнозирования на основе моделей
        _, y_pred = torch.max(outputs.data, 1)

        # Сравнение прогнозируемых и истинных меток
        test_acc += (y_pred == y_true).sum().item()


    print(f"Точность на тестовой выборке = {100 * test_acc / len(test_dataset)} %")



# Генерирование прогнозов для количества изображений 'num_images' из последнего батча тестового выборки
num_images = 5
y_true_name = [names[y_true[idx]] for idx in range(num_images)]
y_pred_name = [names[y_pred[idx]] for idx in range(num_images)]

# Создаем заголовки для графика
title = f"Истинные метки: {y_true_name}, спрогнозированные метки: {y_pred_name}"

# Наносим изображения на график с их фактическими и прогнозируемыми надписями в заголовке
plt.imshow(np.transpose(torchvision.utils.make_grid(images[:num_images].cpu(), normalize=True, padding=1).numpy(), (1, 2, 0)))
plt.title(title)
plt.axis("off")
plt.show()

# Вывод параметров модели
print(summary(model,(3,32,32)))

#Шаг 1: Загрузка данных и печать нескольких образцов изображений из обучающего набора.

#Прежде чем приступить к внедрению CNN, нам сначала нужно загрузить набор данных на наш локальный компьютер, на котором
# мы будем обучать нашу модель. Для этой цели мы будем использовать утилиту torchvision и загружать набор данных CIFAR-10
# в обучающие и тестовые наборы в каталогах “./CIFAR10/train” и “./CIFAR10/test,“ соответственно. Мы также применяем
# нормализованное преобразование, при котором процедура выполняется по трем каналам для всех изображений.
#Теперь у нас есть обучающий и тестовый наборы данных с 50000 и 10000 изображениями, соответственно, размером 32x32x3.
# После этого мы преобразуем эти наборы данных в загрузчики данных размером пакета 128 для лучшего обобщения и ускорения
# процесса обучения.
#Наконец, мы выводим несколько образцов изображений из 1-го обучающего пакета, чтобы получить представление об изображениях,
# с которыми мы имеем дело, используя утилиту make_grid от torchvision.

#Шаг 2: Построение распределения набора данных по классам

#Как правило, хорошей идеей является построение распределения классов обучающего набора. Это помогает проверить,
# сбалансирован ли предоставленный набор данных. Для этого мы пакетами перебираем весь обучающий набор и собираем
# соответствующие классы для каждого экземпляра. Наконец, мы вычисляем количество уникальных классов и наносим их на график.

#Шаг 3: Реализация архитектуры CNN

#Что касается архитектуры, мы будем использовать простую модель, в которой используются три слоя свертки с глубинами
# 32, 64 и 64, соответственно, за которыми следуют два полностью связанных слоя для выполнения классификации.

#Каждый сверточный слой включает в себя операцию свертки, включающую фильтр свертки 3 × 3, за которой следует операция
# активации ReLU для внесения нелинейности в систему и операция максимального объединения с фильтром 2 × 2 для уменьшения
# размерности карты объектов.
#После завершения сверточных блоков мы сглаживаем многомерный слой в низкоразмерную структуру для запуска наших блоков
# классификации. После первого линейного слоя последний выходной слой (также линейный слой) содержит десять нейронов для
# каждого из десяти уникальных классов в нашем наборе данных.

#Для построения нашей модели мы создадим класс CNN, унаследованный от класса torch.nn.Module для использования преимуществ
# утилит Pytorch. Кроме того, мы будем использовать контейнер torch.nn.Sequential для объединения наших слоев один за другим.

#Слои Conv2D(), ReLU(), и MaxPool2D() выполняют операции свертки, активации и объединения в пул. Мы использовали заполнение,
# равное 1, чтобы предоставить ядру достаточно места для обучения, поскольку заполнение увеличивает площадь покрытия
# изображения, особенно пикселей во внешней рамке.
#После сверточных блоков, Linear() полностью связанные слои выполняют классификацию.

#Шаг 4: Определение параметров обучения и начало процесса обучения

#Мы начинаем процесс обучения с выбора устройства для обучения нашей модели, т.Е. центрального процессора или графического
# процессора. Затем мы определяем гиперпараметры нашей модели, которые являются следующими:

#Мы обучаем наши модели на протяжении 50 эпох, и поскольку у нас многоклассовая проблема, мы использовали
# кросс-энтропийные потери в качестве нашей целевой функции.
#Мы использовали популярный оптимизатор Adam с скоростью обучения 0.001 и weight_decay 0.01, чтобы предотвратить
# переобучение посредством регуляризации для оптимизации целевой функции.
#Наконец, мы начинаем наш цикл обучения, который включает в себя вычисление выходных данных для каждой партии и потерь
# путем сравнения прогнозируемых меток с истинными метками. В конце мы нанесли на график потери при обучении для каждой
# соответствующей эпохи, чтобы убедиться, что процесс обучения прошел в соответствии с планом.

#Шаг 5: Вычисление точности модели на тестовом наборе

#Теперь, когда наша модель обучена, нам нужно проверить ее производительность на тестовом наборе. Для этого мы перебираем
# весь набор тестов партиями и вычисляем показатель точности путем сравнения истинных и прогнозируемых меток для каждой
# партии.

#Шаг 6: Генерация прогнозов для образцов изображений в тестовом наборе

# Точность модели была вычислена и отображена на предыдушем шаге. Чтобы проверить ее работоспособность, мы можем
# сгенерировать некоторые прогнозы для некоторых образцов изображений. Для этого мы берем первые пять изображений из
# последней партии тестового набора и наносим их с помощью утилиты make_grid от torchvision. Затем мы собираем их
# истинные метки и прогнозы из модели и показываем их в заголовке графика.