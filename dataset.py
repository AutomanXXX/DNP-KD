from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np

def get_data_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        #transforms.CenterCrop(args.input_size),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

class MVTecDataset_train(torch.utils.data.Dataset):
    def __init__(self, root, transform):
 
        self.img_path = root
        self.transform = transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_type = 'good'
        # print(self.img_path)
        # print(defect_type)
        img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
        img_tot_paths.extend(img_paths)
        gt_tot_paths.extend([0] * len(img_paths))
        tot_labels.extend([0] * len(img_paths))
        tot_types.extend(['good'] * len(img_paths))

        dir_path, _  = os.path.split(self.img_path)
        dir_path, _  = os.path.split(dir_path)
        self._class_ = sorted(os.listdir(dir_path))
        # print(self._class_)
        # exit()

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)


        path = os.path.normpath(self.img_paths[idx])
        path_list = path.split(os.sep)
        class_label_idx = self._class_.index(path_list[-4])
        # class_label = torch.zeros(len(self._class_)+1)
        # class_label[class_label_idx]=1
        class_label=class_label_idx

        # print(path_list[-4])
        # print(self._class_)
        # print(class_label)
        # exit()

        return img, label, class_label


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        
        self.transform = transform
        self.gt_transform = gt_transform

        self.ori_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.CenterCrop(256)])

        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1


    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = sorted(glob.glob(os.path.join(self.img_path, defect_type) + "/*.png"))
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = sorted(glob.glob(os.path.join(self.img_path, defect_type) + "/*.png"))
                gt_paths = sorted(glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png"))
                # img_paths.sort()
                # gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        ori_img = self.ori_transforms(img)
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        defect_type=img_path.split("/")[-2]
        return img, gt, label, img_type,ori_img, defect_type


class VisaDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        
        self.transform = transform
        self.gt_transform = gt_transform

        self.ori_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.CenterCrop(256)])
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1


    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = sorted(glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG"))
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = sorted(glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG"))
                gt_paths = sorted(glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png"))
                # img_paths.sort()
                # gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        ori_img = self.ori_transforms(img)
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        defect_type=img_path.split("/")[-2]

        return img, gt, label, img_type, ori_img, defect_type
   
class VisaDataset_train(torch.utils.data.Dataset):
    def __init__(self, root, transform):
 
        self.img_path = root
        self.transform = transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_type = 'good'
        img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
        img_tot_paths.extend(img_paths)
        gt_tot_paths.extend([0] * len(img_paths))
        tot_labels.extend([0] * len(img_paths))
        tot_types.extend(['good'] * len(img_paths))

        dir_path, _  = os.path.split(self.img_path)
        dir_path, _  = os.path.split(dir_path)
        self._class_ = os.listdir(dir_path)


        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)


        path = os.path.normpath(self.img_paths[idx])
        path_list = path.split(os.sep)
        class_label_idx = self._class_.index(path_list[-4])
        class_label=class_label_idx

        return img, label, class_label