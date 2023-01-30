import torch

import os
import numpy as np
import cv2


# Создание датасета
class Dataset2class(torch.utils.data.Dataset):
    def __init__(self, path_dir1: str, path_dir2: str, size: int): # создадим параметры для всего класса
        super().__init__()

        self.path_dir1 = path_dir1
        self.path_dir2 = path_dir2

        self.dir1_list = sorted(os.listdir(path_dir1)) # Сортировка для понятности
        self.dir2_list = sorted(os.listdir(path_dir2))

        self.size = size

    def __len__(self): # Определим всю длину
        return len(self.dir1_list) + len(self.dir2_list)

    def __getitem__(self, item): # Создадим поэлементно все что нам нужно (словарь с изображением и его классом)

        if item < len(self.dir1_list): # Так как у нас картинки для разных классов лежат в разных папках, то присваивать класс будем так
            class_id = 0
            img_path = os.path.join(self.path_dir1, self.dir1_list[item])
        else:
            class_id = 1
            item -= len(self.dir1_list)
            img_path = os.path.join(self.path_dir2, self.dir2_list[item])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.0

        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA) # Уменьшим картинку чтобы меньше параметров было

        img = img.transpose((2, 0, 1)) # Переводим матрицу под нужный размер (не очень понял)

        t_img = torch.from_numpy(img) # Заводим в тензор (такой формат для сетки)
        t_class_id = torch.tensor(class_id) # Класс тоже в тензор

        return {'img': t_img, 'label': t_class_id} # Выдаем в форме словаря для удобства
