import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch
import torch.utils.data as data
from getROI import *
from getFlow import *
import torch.nn.functional as F


def np_load_frame_roi(filename, resize_height, resize_width, bbox):
    (xmin, ymin, xmax, ymax) = bbox
    image_decoded = cv2.imread(filename)
    image_decoded = image_decoded[ymin:ymax, xmin:xmax]

    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    # image_resized = image_resized.astype(dtype=np.float32)
    # image_resized = (image_resized / 127.5) - 1.0
    return image_resized


def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].
    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    # image_resized = image_resized.astype(dtype=np.float32)
    # image_resized = (image_resized )/255.0

    # image_resized = (image_resized / 127.5) - 1.0
    return image_resized


class VadDataset(data.Dataset):
    def __init__(self, flowargs, video_folder, transform, resize_height, resize_width, dataset='', time_step=4, num_pred=1,
                 bbox_folder=None, device='cuda:0', flow_folder=None):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.dataset = dataset  # ped2 or avenue or ShanghaiTech

        self.bbox_folder = bbox_folder  # 如果box已经预处理了，则直接将npy数据读出来, 如果没有，则在get_item的时候计算
        if bbox_folder == None:  # 装载yolo模型
            self.yolo_weights = 'yolov5/weights/yolov5s.pt'
            self.yolo_device = device
            self.yolo_model = attempt_load(self.yolo_weights, map_location=self.yolo_device)  # load FP32 model

        self.flow_folder = flow_folder
        if self.flow_folder == None:  # 装载flownet

            self.device = device
            self.flownet = FlowNet2(flowargs).to(self.device)
            dict_ = torch.load("flownet2/FlowNet2_checkpoint.pth.tar")
            self.flownet.load_state_dict(dict_["state_dict"])

        self.setup()
        self.samples = self.get_all_samples()

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            # video_name = video.split('/')[-1]           #视频的目录名即类别如01, 02, 03, ...
            video_name = os.path.split(video)[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))  # 每个目录下的所有视频帧
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])  # 每个目录下视频帧的个数

        video_name = os.path.split(videos[0])[-1]
        self.img_size = cv2.imread(self.videos[video_name]['frame'][0]).shape  # [h, w, c]
        if self.bbox_folder != None:  # 如果box已经预处理了，则直接将npy数据读出来
            for bbox_file in sorted(os.listdir(self.bbox_folder)):
                video_name = bbox_file.split('.')[0]
                self.videos[video_name]['bbox'] = np.load(os.path.join(self.bbox_folder, bbox_file),
                                                          allow_pickle=True)  # 每个目录下所有视频帧预提取的bbox

        if self.flow_folder != None:  # 如果已经预处理了，直接读取
            for flow_dir in sorted(os.listdir(self.flow_folder)):
                video_name = flow_dir
                path = os.path.join(self.flow_folder, flow_dir)
                self.videos[video_name]['flow'] = sorted(glob.glob(os.path.join(path, '*')))
        # print(self.videos[video_name]['flow'])

    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            # video_name = video.split('/')[-1]
            video_name = os.path.split(video)[-1]
            for i in range(len(self.videos[video_name]['frame']) - self._time_step):  # 减掉_time_step为了刚好能够滑窗到视频尾部
                frames.append(self.videos[video_name]['frame'][i])  # frames存储着训练时每段视频片段的首帧，根据首帧向后进行滑窗即可得到这段视频片段

        return frames

    def __getitem__(self, index):
        # video_name = self.samples[index].split('/')[-2]      #self.samples[index]取到本次迭代取到的视频首帧，根据首帧能够得到其所属类别及图片名
        # frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])
        # windows

        video_name = os.path.split(os.path.split(self.samples[index])[0])[1]
        frame_name = int(os.path.split(self.samples[index])[1].split('.')[-2])
        print(video_name)
        print(frame_name)

        if self.bbox_folder is not None:  # 已经提取好了bbox，直接读出来
            # bboxes = self.videos[video_name]['bbox'][frame_name + self._time_step - 1]
            bboxes = self.videos[video_name]['bbox'][frame_name]
        else:  # 需要重新计算
            last_frame = [self.videos[video_name]['frame'][frame_name+i] for i in [self._time_step-1, self._time_step]]
            bboxes = RoI(last_frame, self.dataset, self.yolo_model, self.yolo_device)
        w_ratio = self._resize_width / self.img_size[1]      #这里的resize是整张图的resize
        h_ratio = self._resize_height / self.img_size[0]
        trans_bboxes = [[int(box[0] * w_ratio), int(box[1] * h_ratio),
                         int(box[2] * w_ratio), int(box[3] * h_ratio)] for box in bboxes] # example[(290, 91, 301, 109), (332, 94, 343, 113)]

        if self.flow_folder is not None:  # 已经提取好了，直接读出来
            # last_frame_flow = torch.load(self.videos[video_name]['flow'][frame_name + self._time_step - 1])
            last_frame_flow = torch.load(self.videos[video_name]['flow'][frame_name])

        else:
            last_frame_flow = get_frame_flow(self.videos[video_name]['frame'][frame_name + self._time_step - 1],
                                             self.videos[video_name]['frame'][frame_name + self._time_step],
                                             self.flownet,
                                             self.device, 512, 384)
        trans_flow = last_frame_flow.unsqueeze(0)
        trans_flow = F.interpolate(trans_flow, size=([self._resize_width, self._resize_height]), mode='bilinear', align_corners=False)
        trans_flow = trans_flow.squeeze(0)

        # batch = []
        # for i in range(self._time_step + self._num_pred):
        #     image = np_load_frame(self.videos[video_name]['frame'][frame_name + i], self._resize_height,
        #                           self._resize_width)  # 根据首帧图片名便可加载一段视频片段
        #     if self.transform is not None:
        #         batch.append(self.transform(image))
        # batch = torch.stack(batch, dim=0)

        batch = []
        for bbox in bboxes:
            # img
            object_batch = []
            for i in range(self._time_step + self._num_pred):
                image = np_load_frame_roi(self.videos[video_name]['frame'][frame_name + i], 64,
                                          64, bbox)  # 这里的resize是裁剪框的resize
                if self.transform is not None:
                    object_batch.append(self.transform(image))
            object_batch = torch.stack(object_batch, dim=0)
            # print("object_batch.shape: ", object_batch.shape)
            batch.append(object_batch)

        batch = torch.stack(batch, dim=0)  # 大小为[目标个数, _time_step+num_pred, 图片的通道数, _resize_height, _resize_width]

        batch2 = []
        for i in range(self._time_step + self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name + i], self._resize_height,
                                  self._resize_width)  # 根据首帧图片名便可加载一段视频片段
            if self.transform is not None:
                batch2.append(self.transform(image))
        batch2 = torch.stack(batch2, dim=0)

        return batch, trans_bboxes, trans_flow, batch2

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    # test dataloader
    import torchvision
    from torchvision import datasets
    from torch.utils.data import Dataset, DataLoader
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms

    parser = argparse.ArgumentParser()

    # for flownet2, no need to modify
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    batch_size = 1
    datadir = "../AllDatasets/avenue/testing/frames"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # flow 和 yolo 在线计算
    # train_data = VadDataset(args, video_folder=datadir, bbox_folder=None, dataset="avenue", flow_folder=None,
    #                         device=device, transform=transforms.Compose([transforms.ToTensor()]),
    #                         resize_height=256, resize_width=256)

    # 使用保存的.npy
    train_data = VadDataset(args,video_folder= datadir, bbox_folder = "./bboxes/avenue/test", flow_folder="./flow/avenue/test",
                            transform=transforms.Compose([transforms.ToTensor()]),
                            resize_height=480, resize_width=640)

    # # 仅在线计算flow
    # train_data = VadDataset(video_folder= datadir, bbox_folder = "./bboxes/ped2/test", flow_folder=None,
    #                         transform=transforms.Compose([transforms.ToTensor()]),
    #                         resize_height=256, resize_width=256, device = device)

    # 仅在线计算yolo
    # train_data = VadDataset(video_folder= datadir, bbox_folder = None,dataset='ped2',
    #                         flow_folder="./flow/ped2/test",
    #                         transform=transforms.Compose([transforms.ToTensor()]),
    #                         resize_height=256, resize_width=256,
    #                         device = device)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    # for j, (imgs, bbox, flow) in enumerate(train_loader):
    #     print(j)

    unloader = transforms.ToPILImage()

    X, bboxes, flow, batch2= next(iter(train_loader))

    print(X.shape, flow.shape)
    print(batch2.shape)
    print(bboxes[0])
    X = X.squeeze(0)
    # 显示一个batch, pil显示的颜色是有问题的，只是大概看一下
    index = 1
    for i in range(X.shape[0]):
        for j in range(1):
            plt.subplot(X.shape[0], 1, index)
            index += 1
            img = X[i,j,:,:,:].mul(255).byte()
            img = img.cpu().numpy().transpose((1,2,0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = X[i,j,:,:,:].cpu().clone()
            # img = unloader(img)
            # img = np.array(img)
            # for bbox in bboxes:
            #     plot_one_box(bbox, img)
            plt.imshow(img)

    plt.savefig('objectloss.jpg')
    plt.close()

    index = 1
    for i in range(batch2.shape[0]):
        for j in range(1):
            plt.subplot(batch2.shape[0], 1, index)
            index += 1
            img = batch2[i,j,:,:,:].mul(255).byte()
            img = img.cpu().numpy().transpose((1,2,0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = X[i,j,:,:,:].cpu().clone()
            # img = unloader(img)
            # img = np.array(img)
            for bbox in bboxes:
                plot_one_box(bbox, img)
            plt.imshow(img)

    print(index)
    plt.savefig('objectloss2.jpg')