import base64, os, csv, torch, torchvision
from io import BytesIO
import subprocess
import sys
from torch.utils.data import Dataset
import time
from tqdm import tqdm
from multiprocessing import Process
from torchvision.tv_tensors import Image as Tv_Image

subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
from PIL import Image

def str_to_tensor(img_string):
    b64_img = base64.b64decode(img_string)
    io_img = BytesIO(b64_img)
    return torchvision.tv_tensors.Image(Image.open(io_img))

def str_to_img(img_string):
    b64_img = base64.b64decode(img_string)
    io_img = BytesIO(b64_img)
    return Image.open(io_img)

cropped = 'data/ms-celeb-1m/croped_face_images/FaceImageCroppedWithOutAlignment.tsv'
aligned = 'data/ms-celeb-1m/aligned_face_images/FaceImageCroppedWithAlignment.tsv'

def convert_and_save_cropped():
    with open('data/ms-celeb-1m/cropped_face_images/FaceImageCroppedWithOutAlignment.tsv') as f:
        reader = csv.reader(f, delimiter='\t')
        a = 0
        for i in tqdm(reader):
            id, rank, faceid, unknown, data = i[0], i[1], i[4], i[5],  i[6]
            with open('data/ms-celeb-1m/processed/cropped/labels.txt', 'a') as g:
                csv_writer = csv.writer(g)
                csv_writer.writerow({f'{a}-{faceid}-{rank}.png',  faceid, id, rank, unknown})
            with open(f'data/ms-celeb-1m/processed/cropped/{a}-{faceid}-{rank}.png', 'wb') as h:
                img = str_to_img(data)
                img.save(h)
            a = a + 1

def convert_and_save_cropped_chunked(n, m):
    with open('data/ms-celeb-1m/cropped_face_images/FaceImageCroppedWithOutAlignment.tsv') as f:
        reader = csv.reader(f, delimiter='\t')
        a = 0
        for i in tqdm(reader):
            if a > n:
                id, rank, faceid, unknown, data = i[0], i[1], i[4], i[5],  i[6]
                # with open('data/ms-celeb-1m/processed/cropped/labels.txt', 'a') as g:
                #     csv_writer = csv.writer(g)
                #     csv_writer.writerow({f'{a}-{faceid}-{rank}.png',  faceid, id, rank, unknown})
                with open(f'data/ms-celeb-1m/processed/{a}-{faceid}-{rank}.png', 'wb') as h:
                    img = str_to_img(data)
                    img.save(h)
            if a == m:
                return
            a = a + 1


class MSCeleb(Dataset):
    """MS-CELEB-1M dataset.
       Creates a data set with either the aligned or aligned and cropped images"""
    def __init__(self, data_type, img_set, data_root = 'data/ms-celeb-1m/', device = 'cpu', transform=None):
        """
        Arguments:
            data_root (string): Path to the ms-celeb-1m directory
            transform (callable, optional): Optional transform to be applied
                on a sample.
            img_set (string): cropped or aligned
        """
        try:
            assert img_set == 'aligned' or img_set == 'cropped'
        except AssertionError:
            print('invalid img_set')
            return 
        try:
            assert data_root.endswith('/')
        except AssertionError:
            print('data_root must end with a forward slash (/)')
            return
        self.device = device
        self.data_root = data_root
        self.img_set = img_set
        self.transform = transform
        self.data_type = data_type
        
        if img_set == 'cropped':
            self.data_path = f'{data_root}cropped_face_images/FaceImageCroppedWithOutAlignment.tsv'
        elif img_set == 'aligned':
            self.data_path = f'{data_root}aligned_face_images/FaceImageCroppedWithAlignment.tsv'
    def __len__(self):
        with open(self.data_path, 'r') as f:
            return sum(1 for row in f)
    def __getitem__(self, idx):
        with open(self.data_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i == idx and self.data_type == 'tensor':
                    output = str_to_tensor(row[-1])
                    break
                elif i == idx:
                    output = str_to_img(row[-1])
                    break
        if self.transform:
            output = self.transform(output)
        return output



def main():
    convert_and_save_cropped_chunked(667034, 667040)

if __name__ == '__main__':
    main()


def locate_bad_image(n, m, full):
    a = n
    if n == m:
        return 0, False
    dataset = torch.utils.data.Subset(full, range(n, m))
    data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, prefetch_factor=4)
    try:
        for i in data:
            a = a + 1
            continue
    except OSError:
        print(a)
        return a, True
def find_all_bad_images(n,m):
    b = True
    store = []
    dataset = dset.ImageFolder(root=data_root, transform=transforms)
    while b:
        a, b = locate_bad_image(n,m, dataset)
        store.append(a)
        n = a + 1
    return store


