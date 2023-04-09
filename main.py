
import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as compare_ssim
from IQA_pytorch import SSIM, utils
from PIL import Image
from sklearn.metrics import mean_absolute_error


def figure(def_call, process, original, compressed):
    if def_call == 'SSIM':
        origin_squee = original.squeeze(0)
        compre_squee = compressed.squeeze(0)
        process_squee = process.squeeze(0)

        tf = transforms.ToPILImage()
        img1 = tf(origin_squee)
        img2 = tf(compre_squee)
        img3 = tf(process_squee)

        fig = plt.figure()
        rows = 1
        cols = 3

        ax1 = fig.add_subplot(rows, cols, 1)
        ax1.set_title('process')
        ax1.imshow(img3)

        ax1 = fig.add_subplot(rows, cols, 2)
        ax1.set_title('original')
        ax1.imshow(img1)

        ax2 = fig.add_subplot(rows, cols, 3)
        ax2.set_title('results')
        ax2.imshow(img2)

        fig.tight_layout()
        plt.show()
    elif def_call == 'PSNR':
        fig = plt.figure(figsize=(8, 8))

        plt.subplot(1, 3, 1)
        plt.imshow(process[:, :, ::-1])  # BGR -> RGB
        plt.title('process')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 3, 2)
        plt.imshow(original[:, :, ::-1])  # BGR -> RGB
        plt.title('original')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 3, 3)
        plt.imshow(compressed[:, :, ::-1])  # BGR -> RGB
        plt.title('results')
        plt.xticks([])
        plt.yticks([])
        plt.show()

def img_transforms(original, compressed, process, ratio, case, def_call):
    if def_call == 'SSIM':
        _, _, h, w = compressed.shape
        # print(compressed.shape)
        # h, w = compressed.shape
        h, w = math.ceil(h), math.ceil(w)

    elif def_call == 'PSNR':
        h, w, _ = compressed.shape
        h, w = math.ceil(h), math.ceil(w)

    ratio_10 = math.ceil(h * 0.1)
    ratio_20 = math.ceil(h * 0.2)
    ratio_30 = math.ceil(h * 0.3)

    if case == 1:
        # print('case 1')
        if ratio == 10:
            if def_call == 'SSIM':
                original = original[:, :, :ratio_10, :ratio_10]
                compressed = compressed[:, :, :ratio_10, :ratio_10]
                process = process[:, :, :ratio_10, :ratio_10]
            else:
                original = original[:ratio_10, :ratio_10]
                compressed = compressed[:ratio_10, :ratio_10]
                process = process[:ratio_10, :ratio_10]

        elif ratio == 20:
            if def_call == 'SSIM':
                original = original[:, :, :ratio_20, :ratio_20]
                compressed = compressed[:, :, :ratio_20, :ratio_20]
                process = process[:, :, :ratio_20, :ratio_20]
            else:
                original = original[:ratio_20, :ratio_20]
                compressed = compressed[:ratio_20, :ratio_20]
                process = process[:ratio_20, :ratio_20]

        elif ratio == 30:
            if def_call == 'SSIM':
                original = original[:, :, :ratio_30, :ratio_30]
                compressed = compressed[:, :, :ratio_30, :ratio_30]
                process = process[:, :, :ratio_30, :ratio_30]
            else:
                original = original[:ratio_30, :ratio_30]
                compressed = compressed[:ratio_30, :ratio_30]
                process = process[:ratio_30, :ratio_30]

    elif case == 2:
        #print('case 2')
        if ratio == 10:
            if def_call == 'SSIM':
                original = original[:, :, :ratio_10, w-ratio_10:]
                compressed = compressed[:, :, :ratio_10, w-ratio_10:]
                process = process[:, :, :ratio_10, w - ratio_10:]
            else:
                original = original[:ratio_10, w-ratio_10:]
                compressed = compressed[:ratio_10, w-ratio_10:]
                process = process[:ratio_10, w - ratio_10:]

        elif ratio == 20:
            if def_call == 'SSIM':
                original = original[:, :, :ratio_20, w-ratio_20:]
                compressed = compressed[:, :, :ratio_20, w-ratio_20:]
                process = process[:, :, :ratio_20, w - ratio_20:]
            else:
                original = original[:ratio_20, w-ratio_20:]
                compressed = compressed[:ratio_20, w-ratio_20:]
                process = process[:ratio_20, w - ratio_20:]

        elif ratio == 30:
            if def_call == 'SSIM':
                original = original[:, :, :ratio_30, w-ratio_30:]
                compressed = compressed[:, :, :ratio_30, w-ratio_30:]
                process = process[:, :, :ratio_30, w - ratio_30:]
            else:
                original = original[:ratio_30, w-ratio_30:]
                compressed = compressed[:ratio_30, w-ratio_30:]
                process = process[:ratio_30, w - ratio_30:]

    elif case == 3:
        #print('case 3')
        if ratio == 10:
            if def_call == 'SSIM':
                original = original[:, :, h-ratio_10:, :ratio_10]
                compressed = compressed[:, :, h-ratio_10:, :ratio_10]
                process = process[:, :, h - ratio_10:, :ratio_10]
            else:
                original = original[h-ratio_10:, :ratio_10]
                compressed = compressed[h-ratio_10:, :ratio_10]
                process = process[h - ratio_10:, :ratio_10]

        elif ratio == 20:
            if def_call == 'SSIM':
                original = original[:, :, h-ratio_20+15:, :ratio_20+20]
                compressed = compressed[:, :, h-ratio_20+15:, :ratio_20+20]
                process = process[:, :, h - ratio_20+15:, :ratio_20+20]
            else:
                original = original[h-ratio_20+15:, :ratio_20+20]
                compressed = compressed[h-ratio_20+15:, :ratio_20+20]
                process = process[h - ratio_20+15:, :ratio_20+20]

        elif ratio == 30:
            if def_call == 'SSIM':
                original = original[:, :, h-ratio_30:, :ratio_30]
                compressed = compressed[:, :, h-ratio_30:, :ratio_30]
                process = process[:, :, h - ratio_30:, :ratio_30]
            else:
                original = original[h-ratio_30:, :ratio_30]
                compressed = compressed[h-ratio_30:, :ratio_30]
                process = process[h - ratio_30:, :ratio_30]

    elif case == 4:
        #print('case 4')
        if ratio == 10:
            if def_call == 'SSIM':
                original = original[:, :, h-ratio_10:, w-ratio_10:]
                compressed = compressed[:, :, h-ratio_10:, w-ratio_10:]
                process = process[:, :, h - ratio_10:, w - ratio_10:]
            else:
                original = original[h - ratio_10:, w - ratio_10:]
                compressed = compressed[h - ratio_10:, w - ratio_10:]
                process = process[h - ratio_10:, w - ratio_10:]

        elif ratio == 20:
            if def_call == 'SSIM':
                original = original[:, :, h-ratio_20+10:, w-ratio_20-20:]
                compressed = compressed[:, :, h-ratio_20+10:, w-ratio_20-20:]
                process = process[:, :, h - ratio_20+10:, w - ratio_20-20:]
            else:
                original = original[h - ratio_20+10:, w - ratio_20-20:]
                compressed = compressed[h - ratio_20+10:, w - ratio_20-20:]
                process = process[h - ratio_20+10:, w - ratio_20+20:]

        elif ratio == 30:
            if def_call == 'SSIM':
                original = original[:, :, h-ratio_30:, w-ratio_30:]
                compressed = compressed[:, :, h-ratio_30:, w-ratio_30:]
                process = process[:, :, h - ratio_30:, w - ratio_30:]
            else:
                original = original[h - ratio_30:, w - ratio_30:]
                compressed = compressed[h - ratio_30:, w - ratio_30:]
                process = process[h - ratio_30:, w - ratio_30:]
    return original, compressed, process

def psnr(origin_path, compre_path, processing_path, processing_name_li, img_name_li,
         compre_name_li, ratio, case, def_call, image_transforms=False, fig_view=False):
    psnr_li = []
    cnt = 0
    l1_list = []
    l2_list = []
    for i in range(len(img_name_li)):
        cnt += 1
        original = cv2.imread(origin_path + '{}.png'.format(img_name_li[i]))
        compressed = cv2.imread(compre_path + '{}.png'.format(compre_name_li[i]), 1)
        process = cv2.imread(processing_path + '{}.png'.format(processing_name_li[i]))

        original = cv2.resize(original, (256, 256))
        compressed = cv2.resize(compressed, (256, 256))
        process = cv2.resize(process, (256, 256))

        if image_transforms:
            original, compressed, process = img_transforms(original, compressed, process, ratio, case, def_call)
        # print('count : {} / {}'.format(cnt, len(img_name_li)))

        mse = np.mean((original - compressed) ** 2)
        if (mse == 0):  # MSE is zero means no noise is present in the signal .
            # Therefore PSNR have no importance.
            psnr = 100
        else:
            # return psnr
            max_pixel = 255.0
            psnr = 20 * math.log10(max_pixel / math.sqrt(mse))

        l1_norm = (np.sum(np.abs(original - compressed)))/(256*256)
        l2_norm = (np.sum(((original - compressed)**2)))/(256*256)

        if fig_view:
            print('PSNR : {}'.format(psnr))
            print("L1_norm : {}".format(l1_norm))
            print("L2_norm : {}".format(l2_norm))
            figure(def_call, process, original, compressed)
        psnr_li.append(round(psnr, 2))
        l1_list.append(round(l1_norm,1))
        l2_list.append(round(l2_norm,1))


    return psnr_li, l1_list , l2_list

def ssim(origin_path, compre_path, processing_path, processing_name_li, img_name_li, compre_name_li, ratio, case, def_call,
         image_transforms=False, fig_view=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    score_li = []

    cnt = 0
    for i in range(len(img_name_li)):
        cnt += 1
        original = origin_path + '{}.png'.format(img_name_li[i])
        compressed = compre_path + '{}.png'.format(compre_name_li[i])
        process = processing_path + '{}.png'.format(processing_name_li[i])

        original = utils.prepare_image(Image.open(original).convert("RGB")).to(device)
        compressed = utils.prepare_image(Image.open(compressed).convert("RGB")).to(device)
        process = utils.prepare_image(Image.open(process).convert("RGB")).to(device)

        original = F.interpolate(original, (256, 256))
        compressed = F.interpolate(compressed, (256, 256))
        process = F.interpolate(process, (256, 256))

        if image_transforms:
            original, compressed, process = img_transforms(original, compressed, process, ratio, case, def_call)

        # print('count : {} / {}'.format(cnt, len(img_name_li)))

        model = SSIM(channels=3)

        score = model(original, compressed, as_loss=False)

        # print(score.item())
        if fig_view:
            print('SSIM(cnt:{}) : {}'.format(cnt, score.item()))
            print('img path : ', origin_path + '{}.png'.format(img_name_li[i]))
            print('result path : ', compre_path + '{}.png'.format(compre_name_li[i]))
            figure(def_call, process, original, compressed)

        score_li.append(round(score.item(),4))
    return score_li


if __name__ == "__main__":

    def_call = 'PSNR'  # SSIM, PSNR, l1_error, l2_error
    image_transforms = True # 영역별로
    fig_view = False #True #False
    rotation = True
    ratio = 20
    case = 4  # 1 : 왼쪽위, 2 : 오른쪽위, 3 : 왼쪽아래, 4 : 오른쪽아래
    processing_path = '../model_test/22.11.14_model_test/22.11.11_custom/processing_custom/test/'

    origin_path = '../model_test/22.11.14_model_test/22.11.11_custom/original_custom/test/'
    compre_path = './Threshold_mask_results/gated/'

    count = 0
    l1_rotation = []
    l2_rotation = []
    l1_std = []
    l2_std = []
    psnr_li = []
    ssim_li = []
    ssim_std_li = []
    if rotation == True:
        for i in range(1,3):
            case = i  # 1 : 왼쪽위, 2 : 오른쪽위, 3 : 왼쪽아래, 4 : 오른쪽아래
            ratio=10


            img_name_li = [i.split('.')[0] for i in os.listdir(origin_path)]
            compre_name_li = [i.split('.')[0] for i in os.listdir(compre_path)]
            processing_name_li = [i.split('.')[0] for i in os.listdir(processing_path)]

            img_name_li.sort(key=int)
            compre_name_li.sort(key=int)
            processing_name_li.sort(key=int)

            if def_call == 'SSIM':
                ssim_value = ssim(origin_path, compre_path, processing_path, processing_name_li, img_name_li, compre_name_li,
                            ratio, case, def_call, image_transforms, fig_view)
                ssim_low = [ssim for ssim in ssim_value if ssim < 0.65]

                ssim_li.append((i, round(np.mean(ssim_low), 2)))
                ssim_std_li.append((i, round(np.mean(np.std(ssim_low)), 4)))
                # print("SSIM: ", ssim_value)
                # print("SSIM mean: {:.4f}".format(np.mean(ssim_value)))
                # print("ssim_low", len(ssim_low))
                print("ssim_low", ssim_low)
                # print("SSIM std: {:.4f}".format(ssim_std_li))

            elif def_call == 'PSNR':
                value, l1, l2 = psnr(origin_path, compre_path, processing_path, processing_name_li, img_name_li, compre_name_li,
                             ratio, case, def_call, image_transforms, fig_view)
                l1_low = [l1_value for l1_value in l1 if l1_value > 0.1]
                l2_low = [l2_value for l2_value in l2 if 0<l2_value <3]

                l1_rotation.append((i,round(np.mean(l1_low),2)))
                l2_rotation.append((i,round(np.mean(l2_low),2)))
                l1_std.append(round(np.std(l1_low), 2))
                l2_std.append(round(np.std(l2_low), 2))
                psnr_li.append((i,round(np.mean(value),2)))

                print(value)
                print(f"PSNR average value is {round(np.mean(value),2)} dB")
                print("L1", l1_low)
                # print("L1 error mean : {:.1f}".format(np.mean(l1)))
                print("L2: ", l2_low)
                # print("L2 error mean : {:.1f}".format(np.mean(l2)))
                # print("L1_std: ", l1_std)
                # print("L2_std: ", l2_std)
        for i in range(3,5):
            case = i  # 1 : 왼쪽위, 2 : 오른쪽위, 3 : 왼쪽아래, 4 : 오른쪽아래
            ratio=20


            img_name_li = [i.split('.')[0] for i in os.listdir(origin_path)]
            compre_name_li = [i.split('.')[0] for i in os.listdir(compre_path)]
            processing_name_li = [i.split('.')[0] for i in os.listdir(processing_path)]

            img_name_li.sort(key=int)
            compre_name_li.sort(key=int)
            processing_name_li.sort(key=int)
            if def_call == 'SSIM':
                ssim_value = ssim(origin_path, compre_path, processing_path, processing_name_li, img_name_li, compre_name_li,
                            ratio, case, def_call, image_transforms, fig_view)
                ssim_low = [ssim for ssim in ssim_value if ssim < 0.65]

                ssim_li.append((i, round(np.mean(ssim_low), 2)))
                ssim_std_li.append((i, round(np.mean(np.std(ssim_low)), 4)))
                # print("SSIM: ", ssim_value)
                # print("SSIM mean: {:.4f}".format(np.mean(ssim_value)))

                # print("ssim_low", len(ssim_low))
                print("ssim_low", ssim_low)
                # print("SSIM std: {:.4f}".format(ssim_std_li))
            elif def_call == 'PSNR':
                value, l1, l2 = psnr(origin_path, compre_path, processing_path, processing_name_li, img_name_li, compre_name_li,
                             ratio, case, def_call, image_transforms, fig_view)
                l1_low = [l1_value for l1_value in l1 if l1_value > 0.1]
                l2_low = [l2_value for l2_value in l2 if l2_value > 0.1]
                l1_std.append(round(np.std(l1_low), 2))
                l2_std.append(round(np.std(l2_low), 2))
                l1_rotation.append((i, round(np.mean(l1_low), 2)))
                l2_rotation.append((i, round(np.mean(l2_low), 2)))
                psnr_li.append((i, round(np.mean(value), 2)))

                print(value)
                print(f"PSNR average value is {round(np.mean(value), 2)} dB")
                print("L1_low", l1_low)
                print("L2_low: ", l2_low)

        if def_call == 'SSIM':
            print("ssim", ssim_li)
            print("ssim_std", ssim_std_li)
        elif def_call == 'PSNR':
            print("psnr", psnr_li)
            print("L1 rotation: ", l1_rotation)
            print("L2 rotation: ", l2_rotation)
            print("L1 rotation mean: ", np.mean(l1_rotation))
            print("L2 rotation mean: ", np.mean(l2_rotation))
            print("L1 std: ", l1_std)
            print("L2 std: ", l2_std)
            print("L1 std mean", np.mean(l1_std))
            print("L2 std mean", np.mean(l2_std))
