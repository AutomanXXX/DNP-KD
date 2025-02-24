import os
import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from resnet_w_class import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2
from dataset import MVTecDataset,VisaDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from Subnet_w_fftmask import SubNet_w_FFTMask
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle
import warnings
import csv
from torchmetrics import PrecisionRecallCurve

warnings.filterwarnings('ignore')
def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def ColorDifference(imgo,imgr):
    imglabo=cv2.cvtColor(imgo,cv2.COLOR_BGR2LAB)
    imglabr=cv2.cvtColor(imgr,cv2.COLOR_BGR2LAB)
    diff=(imglabr-imglabo)*(imglabr-imglabo)
    RD=diff[:,:,1]
    BD=diff[:,:,2]
    Result=RD+BD
    Result=cv2.blur(Result,(11,11))*0.001
    return Result

def mean_smoothing(amaps, kernel_size: int = 21) :

    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)


def evaluation(encoder, bn, decoder, subnet_w_fftmask, dataloader,device,_class_):

    bn.eval()
    decoder.eval()


    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []

    cos_loss = torch.nn.CosineSimilarity()

    img_show_path = 'Test_img_ours_mvtec' + '/' + _class_ + '/'  #保存路径
    if not os.path.exists(img_show_path):
        os.makedirs(img_show_path)
    t=0

    with torch.no_grad():
        for img, gt, label, _,_,_ in dataloader:
            img = img.to(device)

            inputs = encoder(img)
            out_subnet = subnet_w_fftmask(inputs)
            mid = bn(out_subnet)
            outputs = decoder(mid)

            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')


            anomaly_map = gaussian_filter(anomaly_map , sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if label.item()!=0 and gt.max()==1:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),anomaly_map[np.newaxis,:,:]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())#1，1，256，256->65536,
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))



        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
        ap_sp = round(average_precision_score(gt_list_sp, pr_list_sp), 3)
        ap_pixel = round(average_precision_score(gt_list_px, pr_list_px), 3)

    return auroc_px, auroc_sp, round(np.mean(aupro_list),3), ap_sp, ap_pixel

def test(_class_,ckpt_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(256, 256)
    if args.ckpt_name=='DNP-KD_mvtec_Unified_50':
        test_path = './mvtecdataset/' + _class_
    else:
        test_path = './visadataset/' + _class_

    ckp_path = './checkpoints/' + ckpt_name + '.pth'
    if args.ckpt_name=='DNP-KD_mvtec_Unified_50':
        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    else:
        test_data = VisaDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    if args.ckpt_name=='DNP-KD_mvtec_Unified_50':
        subnet_w_fftmask = SubNet_w_FFTMask(num_classes=15)
    else:
        subnet_w_fftmask = SubNet_w_FFTMask(num_classes=12)
    subnet_w_fftmask = subnet_w_fftmask.to(device)

    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    
    subnet_w_fftmask.load_state_dict(ckp['subnet_w_fftmask'])
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    auroc_px, auroc_sp, aupro_px, ap_sp, ap_pixel = evaluation(encoder, bn, decoder, subnet_w_fftmask, test_dataloader, device, _class_)
    print(_class_,'img:',auroc_sp,', pix:',auroc_px,', PRO:',aupro_px,', ap_img:',ap_sp,', ap_pix:',ap_pixel)
    return auroc_sp, auroc_px, aupro_px, ap_sp, ap_pixel


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df._append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ckpt_name', type=str, default='DNP-KD_mvtec_Unified_50')
    parser.add_argument('--gpu', default='4', type=str,
                        help='GPU id to use.')
    args = parser.parse_args()

    if args.ckpt_name=='DNP-KD_mvtec_Unified_50':
        obj_list =[
                    'carpet',      #0
                    'grid',        #1
                    'leather',     #2
                    'tile',        #3
                    'wood',        #4
                    'pill',        #5
                    'transistor',  #6
                    'cable',       #7
                    'zipper',      #8
                    'toothbrush',  #9
                    'metal_nut',   #10
                    'hazelnut',    #11
                    'screw',       #12
                    'capsule',     #13
                    'bottle'       #14
                    ]  
    else:
        obj_list = ['pcb1', 'pcb2', 'pcb3', 'pcb4', 'macaroni1', 'macaroni2', 'capsules', 'candle',
                'cashew', 'chewinggum', 'fryum','pipe_fryum']   

    obj_auroc_pixel_list = []
    obj_auroc_image_list = []
    obj_aupro_list = []
    obj_ap_image_list = []
    obj_ap_pixel_list = []
    obj_iou_pixel_list = []

    results=[]
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    for i in obj_list:
        auroc,auroc_pixel,aupro_list, ap_sp, ap_pixel  = test(i,args.ckpt_name)

        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        obj_aupro_list.append(aupro_list)
        obj_ap_image_list.append(ap_sp)
        obj_ap_pixel_list.append(ap_pixel)
        results.append({
        "class": i,
        "img": round(auroc*100,1),
        "pix": round(auroc_pixel*100,1),
        "PRO": round(aupro_list*100,1),
        "img_ap": round(ap_sp*100,1),
        "pix_ap": round(ap_pixel*100,1)
    })
    print("----obj_average----")
    print(round(np.mean(obj_auroc_image_list),3))
    print(round(np.mean(obj_auroc_pixel_list),3))
    print(round(np.mean(obj_aupro_list),3))
    print(round(np.mean(obj_ap_image_list),3))
    print(round(np.mean(obj_ap_pixel_list),3))

    results.append({
    "class": "average",
    "img": round(np.mean(obj_auroc_image_list)*100,1),
    "pix": round(np.mean(obj_auroc_pixel_list)*100,1),
    "PRO": round(np.mean(obj_aupro_list)*100,1),
    "img_ap": round(np.mean(obj_ap_image_list)*100,1),
    "pix_ap": round(np.mean(obj_ap_pixel_list)*100,1)
})
    print(args.ckpt_name)
    import csv
    csv_file='./outputs/ck'+args.ckpt_name+'.csv'
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        # Write the data to a CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["class", "img", "pix", "PRO","img_ap","pix_ap"])
        writer.writeheader()
        for result in results:
            writer.writerow(result)

