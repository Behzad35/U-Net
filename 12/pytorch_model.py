import torch as t
import torch.nn as nn
import os
import numpy as np
import time
from wand.image import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import math
from torchsummary import summary

# large u-net similar to the one from the original publication (https://arxiv.org/pdf/1505.04597.pdf)

channel_size = 2
use_real_images = True
use_gpu = False

image_path = "training_data/images"
annotation_path = "training_data/annotations"
image_resolution = 512
batch_size = 8
num_batches = 16


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # t.set_default_tensor_type(t.DoubleTensor)
        t.manual_seed(101)

        self.conv0 = nn.Conv2d(
            in_channels=3,
            out_channels=2 ** channel_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )
        self.conv1 = nn.Conv2d(
            in_channels=2 ** channel_size,
            out_channels=2 ** channel_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )

        self.conv2 = nn.Conv2d(
            in_channels=2 ** channel_size,
            out_channels=2 ** (channel_size + 1),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )
        self.conv3 = nn.Conv2d(
            in_channels=2 ** (channel_size + 1),
            out_channels=2 ** (channel_size + 1),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )

        self.conv4 = nn.Conv2d(
            in_channels=2 ** (channel_size + 1),
            out_channels=2 ** (channel_size + 2),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )
        self.conv5 = nn.Conv2d(
            in_channels=2 ** (channel_size + 2),
            out_channels=2 ** (channel_size + 2),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )

        self.conv6 = nn.Conv2d(
            in_channels=2 ** (channel_size + 2),
            out_channels=2 ** (channel_size + 3),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )
        self.conv7 = nn.Conv2d(
            in_channels=2 ** (channel_size + 3),
            out_channels=2 ** (channel_size + 3),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )

        self.conv8 = nn.Conv2d(
            in_channels=2 ** (channel_size + 3),
            out_channels=2 ** (channel_size + 4),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )
        self.conv9 = nn.Conv2d(
            in_channels=2 ** (channel_size + 4),
            out_channels=2 ** (channel_size + 3),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )

        self.conv10 = nn.Conv2d(
            in_channels=2 ** (channel_size + 4),
            out_channels=2 ** (channel_size + 3),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )
        self.conv11 = nn.Conv2d(
            in_channels=2 ** (channel_size + 3),
            out_channels=2 ** (channel_size + 2),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )

        self.conv12 = nn.Conv2d(
            in_channels=2 ** (channel_size + 3),
            out_channels=2 ** (channel_size + 2),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )
        self.conv13 = nn.Conv2d(
            in_channels=2 ** (channel_size + 2),
            out_channels=2 ** (channel_size + 1),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )

        self.conv14 = nn.Conv2d(
            in_channels=2 ** (channel_size + 2),
            out_channels=2 ** (channel_size + 1),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )
        self.conv15 = nn.Conv2d(
            in_channels=2 ** (channel_size + 1),
            out_channels=2 ** channel_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )

        self.conv16 = nn.Conv2d(
            in_channels=2 ** (channel_size + 1),
            out_channels=2 ** (channel_size + 1),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )
        self.conv17 = nn.Conv2d(
            in_channels=2 ** (channel_size + 1),
            out_channels=2 ** (channel_size + 1),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode="zeros",
        )
        self.conv18 = nn.Conv2d(
            in_channels=2 ** (channel_size + 1),
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            padding_mode="zeros",
        )

        self.transposed_conv0 = nn.ConvTranspose2d(
            in_channels=2 ** (channel_size + 3),
            out_channels=2 ** (channel_size + 3),
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )
        self.transposed_conv1 = nn.ConvTranspose2d(
            in_channels=2 ** (channel_size + 2),
            out_channels=2 ** (channel_size + 2),
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )
        self.transposed_conv2 = nn.ConvTranspose2d(
            in_channels=2 ** (channel_size + 1),
            out_channels=2 ** (channel_size + 1),
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )
        self.transposed_conv3 = nn.ConvTranspose2d(
            in_channels=2 ** channel_size,
            out_channels=2 ** channel_size,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )

        self.relu = nn.ReLU()
        self.avg_pool2d = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.relu(self.conv0(x))
        skip_out_0 = self.relu(self.conv1(out))
        out = self.avg_pool2d(skip_out_0)

        out = self.relu(self.conv2(out))
        skip_out_1 = self.relu(self.conv3(out))
        out = self.avg_pool2d(skip_out_1)

        out = self.relu(self.conv4(out))
        skip_out_2 = self.relu(self.conv5(out))
        out = self.avg_pool2d(skip_out_2)

        out = self.relu(self.conv6(out))
        skip_out_3 = self.relu(self.conv7(out))
        out = self.avg_pool2d(skip_out_3)

        out = self.relu(self.conv8(out))
        out = self.relu(self.conv9(out))
        out = self.transposed_conv0(out)

        out = t.cat((out, skip_out_3), 1)
        out = self.relu(self.conv10(out))
        out = self.relu(self.conv11(out))
        out = self.transposed_conv1(out)

        out = t.cat((out, skip_out_2), 1)
        out = self.relu(self.conv12(out))
        out = self.relu(self.conv13(out))
        out = self.transposed_conv2(out)

        out = t.cat((out, skip_out_1), 1)
        out = self.relu(self.conv14(out))
        out = self.relu(self.conv15(out))
        out = self.transposed_conv3(out)

        out = t.cat((out, skip_out_0), 1)
        out = self.relu(self.conv16(out))
        out = self.relu(self.conv17(out))
        out = self.conv18(out)

        return out


class ImageDataset(Dataset):
    def __init__(self, data):
        super(ImageDataset, self).__init__()
        self.data = data
        # self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.RandomHorizontalFlip(), tv.transforms.RandomVerticalFlip(), tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean, train_std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image(filename=os.path.join(image_path, self.data.iloc[index, 0]))
        image = np.moveaxis(np.array(image), -1, 0) / 256.0
        image = np.flip(image, 1)
        image = t.Tensor(image.copy())

        label = Image(filename=os.path.join(annotation_path, self.data.iloc[index, 1]))
        label = np.flip(label, 0)
        label = np.moveaxis(label, -1, 0)[0]
        label = t.sub(t.Tensor(label.copy()), 1)

        # image = self._transform(image)

        return (image, label)


def initInputImage():
    res = t.Tensor(t.zeros(batch_size, 3, image_resolution, image_resolution))
    for i in range(image_resolution):
        for j in range(image_resolution):
            for b in range(batch_size):
                res[b, 0, i, j] = (
                    math.sin((1 + 2 * i) / (image_resolution * 2) * math.pi)
                    * math.sin((1 + 2 * j) / (image_resolution * 2) * math.pi)
                    + b * 0.1
                )
                res[b, 1, i, j] = (
                    math.sin((1 + 2 * i) / (image_resolution * 2) * math.pi)
                    * math.sin((1 + 2 * j) / (image_resolution * 2) * math.pi)
                    + b * 0.1
                )
                res[b, 2, i, j] = (
                    math.sin((1 + 2 * i) / (image_resolution * 2) * math.pi)
                    * math.sin((1 + 2 * j) / (image_resolution * 2) * math.pi)
                    + b * 0.1
                )

    return res


def initReferenceImage():
    res = t.Tensor(t.zeros(batch_size, image_resolution, image_resolution))
    for i in range(image_resolution):
        for j in range(image_resolution):
            for b in range(batch_size):
                if (
                    math.sin((1 + 2 * i) / (image_resolution * 2) * math.pi)
                    * math.sin((1 + 2 * j) / (image_resolution * 2) * math.pi)
                    + b * 0.1
                    >= 0.66
                ):
                    res[b, i, j] = 2

                elif (
                    math.sin((1 + 2 * i) / (image_resolution * 2) * math.pi)
                    * math.sin((1 + 2 * j) / (image_resolution * 2) * math.pi)
                    + b * 0.1
                    >= 0.33
                ):
                    res[b, i, j] = 1

    return res


if __name__ == "__main__":
    t.set_num_threads(4)
    model = UNet()
    
    summary(model, (3,512,512))
    for name, param in model.named_parameters():
        print('name: ', name)
        print(type(param))
        print('param.shape: ', param.shape)
        print('param.requires_grad: ', param.requires_grad)
        print('=====')
        
       # np_arr = param.detach().cpu().numpy()
       # string = name + ".npy"
       # np.save(string, np_arr)

    
    loss_function = nn.CrossEntropyLoss()
    if use_gpu:
        model = model.cuda()
        loss_function = loss_function.cuda()
    optimizer = t.optim.Adam(model.parameters(), lr=0.001)

    data = pd.read_csv("data.csv", sep=";")
    train_dl = DataLoader(ImageDataset(data), batch_size=batch_size, num_workers=4)

    if not use_real_images:
        input = initInputImage()
        reference_output = initReferenceImage().long()
        if use_gpu:
            input = input.cuda()
            reference_output = reference_output.cuda()

    total_training_time = 0.0
    total_forward_time = 0.0
    total_backward_time = 0.0
    total_execution_time = time.time()

    for epoch in range(5):
        average_loss = 0.0

        for batch in iter(train_dl):
            if use_real_images:
                input = batch[0]
                reference_output = batch[1].long()
                if use_gpu:
                    input = input.cuda()
                    reference_output = reference_output.cuda()
            t_start = time.time()
            output = model(input)
            total_forward_time += time.time() - t_start
            optimizer.zero_grad()
            loss_value = loss_function(output, reference_output)

            average_loss += float(loss_value)

            t2_start = time.time()
            loss_value.backward()
            total_backward_time += time.time() - t2_start
            optimizer.step()

            t_end = time.time()
            total_training_time += t_end - t_start

        average_loss /= num_batches
        print(average_loss)

    total_execution_time = time.time() - total_execution_time

    print("\ntime for training: " + str(round(total_training_time, 4)) + " s")
    print("\ntime for forward: " + str(round(total_forward_time, 4)) + " s")
    print("\ntime for backward: " + str(round(total_backward_time, 4)) + " s")
    print(
        "\ntime for reading images: "
        + str(round(total_execution_time - total_training_time, 4))
        + " s"
    )
