# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:04:01 2023

@author: u0117961
"""
# change the number of stages
import pickle as pkl
import numpy as np
import os
import torch
import tensorflow as tf
from tensorflow import keras
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from torch import optim
import copy
from datetime import datetime
from torchmetrics.functional import f1_score as F1S
from torchmetrics import CohenKappa
from torchmetrics import MatthewsCorrCoef
from sklearn.metrics import classification_report
import time
import math

# %%
print("there is gpu or nor", torch.cuda.is_available())
print(
    "there are ",
    torch.cuda.device_count(),
    " pieces of ",
    torch.cuda.get_device_name(0),
)


# %%
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(
            out_channels, out_channels, 1
        )  # 这一层的作用在于整合不同channel(filters)之间的信息
        self.dropout = nn.Dropout()

    def forward(self, x):
        # print("DilatedResidualLayer")
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(
            dim, num_f_maps, 1
        )  # dim=channel数目 num_f_maps=filter数目
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps))
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList(
            [
                copy.deepcopy(
                    SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)
                )
                for s in range(num_stages - 1)
            ]
        )

    def forward(self, x):
        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        outputs = outputs.permute(1, 0, 2, 3)
        return outputs


def evaluation_idx(con):
    acc = (con[0, 0] + con[1, 1]) / (con[0, 0] + con[1, 1] + con[0, 1] + con[1, 0])
    pre = (con[1, 1]) / (con[1, 1] + con[0, 1])
    rec = (con[1, 1]) / (con[1, 1] + con[1, 0])
    fsc = (2 * con[1, 1]) / (2 * con[1, 1] + con[0, 1] + con[1, 0])
    return acc, pre, rec, fsc


# %%
Batch_size = 64
val_batch = 8
num_stages = 2
num_layer = 9
num_filter = 64
num_feature = 6
num_class = 3
Epoch = 100
learning_rate = 5e-4
DOWNSAMPLING = 4
sample_freq = 64
duration = 60
strde = 60
data_freq = sample_freq // DOWNSAMPLING
seg_length = data_freq * duration
strde_length = sample_freq * strde
cohenkappa = CohenKappa(num_classes=num_class, task="multiclass")
matthews_corrcoef = MatthewsCorrCoef(num_classes=num_class)
# %%
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
with open("/scratch/leuven/341/vsc34197/pkl_data/meal_X_no_overlap.pkl", "rb") as f:
    dataset = pkl.load(f)
with open("/scratch/leuven/341/vsc34197/pkl_data/meal_Y_no_overlap.pkl", "rb") as f:
    annoLabel = pkl.load(f)
with open("/scratch/leuven/341/vsc34197/pkl_data/dl_34_X_rest.pkl", "rb") as f:
    DL_data = pkl.load(f)
with open("/scratch/leuven/341/vsc34197/pkl_data/dl_34_Y_rest.pkl", "rb") as f:
    DL_anno = pkl.load(f)
with open("/scratch/leuven/341/vsc34197/pkl_data/oreba_100_X.pkl", "rb") as f:
    OR_data = pkl.load(f)
with open("/scratch/leuven/341/vsc34197/pkl_data/oreba_100_Y.pkl", "rb") as f:
    OR_anno = pkl.load(f)

# %%
str_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# %%
for times in range(0, 4):
    print(
        "-------------------------------Round {} start-------------------------------".format(
            times + 1
        )
    )
    str_org = (
        "/data/leuven/341/vsc34197/day_long/data_split/k_fold/"
        + "20230815_170535"
        + "/"
        + str(times)
    )
    pa_train = str_org + "/train.npy"
    pa_valid = str_org + "/valid.npy"
    pa_test = str_org + "/test.npy"
    train_list = np.load(pa_train)
    valid_list = np.load(pa_valid)
    test_list = np.load(pa_test)
    print(train_list)
    print(valid_list)
    print(test_list)
    model_dir = (
        "/data/leuven/341/vsc34197/day_long/model/mst/fd_34_rest/tts/"
        + str_time
        + "/"
        + str(times)
        + "/"
    )
    results_dir = (
        "/data/leuven/341/vsc34197/day_long/results/mst/fd_34_rest/tts/"
        + str_time
        + "/"
        + str(times)
        + "/"
    )
    model_dir_temp = (
        "/data/leuven/341/vsc34197/day_long/mst/fd_34_rest/tts/" + str_time + "/"
    )
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(model_dir_temp):
        os.makedirs(model_dir_temp)
    for i in range(len(annoLabel)):
        if i == 0:
            x_train = dataset[i]
            y_train = annoLabel[i]
        else:
            x_train = np.concatenate((x_train, dataset[i]), axis=0)
            y_train = np.concatenate((y_train, annoLabel[i]), axis=0)

    for i in range(len(OR_anno)):
        x_train = np.concatenate((x_train, OR_data[i]), axis=0)
        y_train = np.concatenate((y_train, OR_anno[i]), axis=0)
    # print(x_train.shape)
    # print(y_train.shape)
    # train set
    for i in range(len(train_list)):
        idx = train_list[i]
        x_train = np.concatenate((x_train, DL_data[idx]), axis=0)
        y_train = np.concatenate((y_train, DL_anno[idx]), axis=0)
    x_train = x_train.astype(np.float32)
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_train.shape)
    # print(y_train.shape)

    added_seg = seg_length - len(y_train) % seg_length
    if added_seg != seg_length:
        # print("let us add some zero data to meet the seg_length")
        # print("initial length of data", len(y_train))
        added_shape2 = x_train[0:added_seg, :]
        added_zero_data2 = np.zeros(added_shape2.shape)
        added_zero_data2 = added_zero_data2.astype(np.float32)
        x_train = np.concatenate((x_train, added_zero_data2), axis=0)
        added_shape2 = y_train[0:added_seg]
        added_zero_data2 = np.zeros(added_shape2.shape)
        added_zero_data2 = added_zero_data2.astype(np.int64)
        y_train = np.concatenate((y_train, added_zero_data2), axis=0)
        # print("after added, length of data", len(y_train))
    else:
        added_seg = 0
        # print("the inigital seg length meets the seg_length")

    (x_train) = tf.keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        targets=None,
        sequence_length=seg_length,
        sequence_stride=strde_length,
        batch_size=None,
        sampling_rate=DOWNSAMPLING,
    )
    (y_train) = tf.keras.preprocessing.timeseries_dataset_from_array(
        y_train,
        targets=None,
        sequence_length=seg_length,
        sequence_stride=strde_length,
        batch_size=None,
        sampling_rate=DOWNSAMPLING,
    )
    Sequence_data = list()
    Sequence_label = list()
    Sequence_label_class = list()
    counter = 0
    for oneSequence in x_train:
        Sequence_data.append(oneSequence)
    for oneSequence in y_train:
        x = tf.dtypes.cast(oneSequence, tf.int64)
        Sequence_label.append(x)

    Sequence_data = tf.convert_to_tensor(Sequence_data)
    Sequence_label = tf.convert_to_tensor(Sequence_label)
    Sequence_data = Sequence_data.numpy()
    Sequence_label = Sequence_label.numpy()

    added_len = Batch_size - len(Sequence_data) % Batch_size
    if added_len != Batch_size:
        # print("let us add some zero data to meet the batchsize")
        # print("initial length of data", len(Sequence_data))
        added_shape = Sequence_data[0:added_len, :, :]
        added_zero_data = np.zeros(added_shape.shape)
        added_zero_data = added_zero_data.astype(np.float32)
        Sequence_data = np.concatenate((Sequence_data, added_zero_data), axis=0)
        added_shape = Sequence_label[0:added_len, :]
        added_zero_data = np.zeros(added_shape.shape)
        added_zero_data = added_zero_data.astype(np.int64)
        Sequence_label = np.concatenate((Sequence_label, added_zero_data), axis=0)
        # print("after added, length of data", len(Sequence_data))
    else:
        added_len = 0
        # print("the inigital length meets the batchsize")

    Sequence_data = torch.from_numpy(Sequence_data)
    Sequence_label = torch.from_numpy(Sequence_label)

    Sequence_data = torch.transpose(Sequence_data, 1, 2)
    dataset_total = TensorDataset(Sequence_data, Sequence_label)
    dataLoader_train = DataLoader(
        dataset_total, batch_size=Batch_size, drop_last=False, shuffle=False
    )

    # valid set
    for i in range(len(valid_list)):
        idx = valid_list[i]
        if i == 0:
            x_val = DL_data[idx]
            y_val = DL_anno[idx]
        else:
            x_val = np.concatenate((x_val, DL_data[idx]), axis=0)
            y_val = np.concatenate((y_val, DL_anno[idx]), axis=0)
    x_val = x_val.astype(np.float32)

    added_seg = seg_length - len(y_val) % seg_length
    if added_seg != seg_length:
        # print("let us add some zero data to meet the seg_length")
        # print("initial length of data", len(y_val))
        added_shape2 = x_val[0:added_seg, :]
        added_zero_data2 = np.zeros(added_shape2.shape)
        added_zero_data2 = added_zero_data2.astype(np.float32)
        x_val = np.concatenate((x_val, added_zero_data2), axis=0)
        added_shape2 = y_val[0:added_seg]
        added_zero_data2 = np.zeros(added_shape2.shape)
        added_zero_data2 = added_zero_data2.astype(np.int64)
        y_val = np.concatenate((y_val, added_zero_data2), axis=0)
        # print("after added, length of data", len(y_val))
    else:
        added_seg = 0
        # print("the inigital seg length meets the seg_length")

    (x_val) = tf.keras.preprocessing.timeseries_dataset_from_array(
        x_val,
        targets=None,
        sequence_length=seg_length,
        sequence_stride=strde_length,
        batch_size=None,
        sampling_rate=DOWNSAMPLING,
    )
    (y_val) = tf.keras.preprocessing.timeseries_dataset_from_array(
        y_val,
        targets=None,
        sequence_length=seg_length,
        sequence_stride=strde_length,
        batch_size=None,
        sampling_rate=DOWNSAMPLING,
    )
    Sequence_data = list()
    Sequence_label = list()
    Sequence_label_class = list()
    counter = 0
    for oneSequence in x_val:
        Sequence_data.append(oneSequence)
    for oneSequence in y_val:
        x = tf.dtypes.cast(oneSequence, tf.int64)
        Sequence_label.append(x)
    Sequence_data = tf.convert_to_tensor(Sequence_data)
    Sequence_label = tf.convert_to_tensor(Sequence_label)
    # print("person id label is",Sequence_label.shape)
    # dataset_total= tf.data.Dataset.from_tensor_slices((Sequence_data,Sequence_label))
    Sequence_data = Sequence_data.numpy()
    Sequence_label = Sequence_label.numpy()
    added_len = Batch_size - len(Sequence_data) % Batch_size
    if added_len != Batch_size:
        # print("let us add some zero data to meet the batchsize")
        # print("initial length of data", len(Sequence_data))
        added_shape = Sequence_data[0:added_len, :, :]
        added_zero_data = np.zeros(added_shape.shape)
        added_zero_data = added_zero_data.astype(np.float32)
        Sequence_data = np.concatenate((Sequence_data, added_zero_data), axis=0)
        added_shape = Sequence_label[0:added_len, :]
        added_zero_data = np.zeros(added_shape.shape)
        added_zero_data = added_zero_data.astype(np.int64)
        Sequence_label = np.concatenate((Sequence_label, added_zero_data), axis=0)
        # print("after added, length of data", len(Sequence_data))
    else:
        added_len = 0
        # print("the inigital length meets the batchsize")

    Sequence_data = torch.from_numpy(Sequence_data)
    Sequence_label = torch.from_numpy(Sequence_label)
    Sequence_data = torch.transpose(Sequence_data, 1, 2)
    datasetVal = TensorDataset(Sequence_data, Sequence_label)
    dataLoader_val = DataLoader(
        datasetVal, batch_size=Batch_size, drop_last=False, shuffle=False
    )
    if len(dataLoader_val) == 0:
        dataLoader_val = DataLoader(
            datasetVal, batch_size=Batch_size, drop_last=False, shuffle=False
        )
        # import torchvision
    device = torch.device("cuda")
    loss_ce = nn.CrossEntropyLoss(ignore_index=-100)
    # loss_ce=loss_ce.to(torch.device("cuda"))
    loss_mse = nn.MSELoss(reduction="none")
    # loss_mse=loss_mse.to(device)

    # 创建模型
    mstcn1 = MultiStageModel(
        num_stages, num_layer, num_filter, num_feature, num_class
    )  # 第一个是stage个数 第二个是每个stage的层数 第三个是每层filter个数 第四个是input的channel数 第五个是类别数
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     if torch.cuda.device_count()>2:
    #         #print("Let's use", torch.cuda.device_count(), "GPUs!")
    #         mstcn1 = nn.DataParallel(mstcn1, device_ids=[0, 1])
    #     else:
    #         #print("Let's use", torch.cuda.device_count(), "GPUs!")
    #         mstcn1 = nn.DataParallel(mstcn1)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        mstcn1 = nn.DataParallel(mstcn1)

    mstcn1 = mstcn1.to(device)
    optimizer = optim.Adam(mstcn1.parameters(), lr=learning_rate)
    # 设置训练参数
    epoch = Epoch
    best_kappa_val = 0
    # PATH = "/data/leuven/341/vsc34197/radar/d3/39/split/bestModel"+str(times)+".pth"
    PATH = model_dir_temp + "bestModel" + str(times) + ".pth"

    for i in range(epoch):
        print("--------------------epoch{} start----------------------".format(i + 1))
        # 遍历该epoch中的每个batch
        if i <= 3:
            t0 = time.time()
        epoch_loss = 0
        correct = 0
        total = 0
        kap_epoch = 0
        mcc_epoch = 0
        f1_score_train = 0
        f1_score_counter = 0
        tp, target_true, pred_true = 0, 0, 0
        # y_pred = []
        # y_true = []
        mstcn1 = mstcn1.train()
        for batch_ndx, sample in enumerate(dataLoader_train):
            # print("batch index is ",batch_ndx)
            oneBatchData = sample[0]
            oneBatchLabel = sample[1]
            oneBatchData = oneBatchData.to(device)
            oneBatchLabel = oneBatchLabel.to(device)
            oneBatchLabel2 = F.one_hot(oneBatchLabel, num_classes=3)
            oneBatchLabel2 = oneBatchLabel2.type(torch.FloatTensor).to(device)
            # print("input oneBatchdata size: ", oneBatchData.size())
            # print("input oneBatchlabel2 size: ", oneBatchLabel2.size())
            optimizer.zero_grad()
            predictions = mstcn1(oneBatchData)
            # print("output predictions size: ", predictions.size())
            predictions = predictions.permute(1, 0, 2, 3)
            # print("output predictions size2: ", predictions.size())
            # print("Label size: ", oneBatchLabel2.size())
            # predictions= predictions.transpose(0,1)
            # predictions=torch.flatten(predictions,0,1)
            # predictions = torch.unsqueeze(predictions, 0)
            # print("output size: ", predictions.size())
            loss = 0
            # 遍历所有stage的预测算出总loss
            for p in predictions:
                # print("output p size: ",p.size())
                loss += loss_ce(
                    p.transpose(2, 1).contiguous().view(-1, 3),
                    oneBatchLabel2.view(-1, 3),
                )
                loss += 0.15 * torch.mean(
                    torch.clamp(
                        loss_mse(
                            F.log_softmax(p[:, :, 1:], dim=1),
                            F.log_softmax(p.detach()[:, :, :-1], dim=1),
                        ),
                        min=0,
                        max=16,
                    )
                )
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            # print("output predictions[-1].data size: ", predictions[-1].data.size())
            # print("output oneBatchLabel size: ", oneBatchLabel.size())
            _, predicted = torch.max(
                predictions[-1].data, 1
            )  # [30,2,2000] =>max => [30,2000] 这里-1代表最后一层stage的预测输出
            # print("raw predicted p for all classes is", (predictions[-1].data)[0,:,1996],(predictions[-1].data).shape)
            correct += (
                ((predicted == oneBatchLabel).float().squeeze(1)).sum().item()
            )  # prediectd的大小是[30,2000] 放的是类的index 和oneBatchLabel大小完全一样因此可以直接比较
            if i % 3 == 0:
                if batch_ndx == 0:
                    y_true = oneBatchLabel.cpu().flatten()
                    y_pred = predicted.cpu().flatten()
                else:
                    # gt_batch=oneBatchLabel.cpu().flatten()
                    # pred_batch=predicted.cpu().flatten()
                    y_pred = torch.cat((y_pred, predicted.cpu().flatten()))
                    y_true = torch.cat((y_true, oneBatchLabel.cpu().flatten()))
            # f1_score_train += F1S(gt_batch, pred_batch,average='macro', num_classes=3)
            # kap_epoch+=cohenkappa(oneBatchLabel.cpu().flatten(), predicted.cpu().flatten())
            # mcc_epoch+=matthews_corrcoef(oneBatchLabel.cpu().flatten(), predicted.cpu().flatten())
            f1_score_counter += 1
            total += len(
                torch.flatten(oneBatchLabel)
            )  # [30,2000] [batchsize,seqlength]

        print("EPOCH loss is ", epoch_loss)
        print("acc = %f" % (float(correct) / total))
        # print("avg_f1_score = %f" % ( float(f1_score_train)/f1_score_counter))
        if i % 3 == 0:
            kap_epoch = cohenkappa(y_true, y_pred)
            # mcc_epoch=matthews_corrcoef(y_true, y_pred)
            print("kappa = %f" % (kap_epoch))
            # print("mcc = %f" % (mcc_epoch))
        ################################evaluate in validate data#######################################
        total_val_loss = 0
        correct_val = 0
        mstcn1 = mstcn1.eval()
        total_val = 0
        kap_val = 0
        mcc_val = 0
        f1_score_val = 0
        f1_score_counter_val = 0
        tp, target_true, pred_true = 0, 0, 0
        # y_pred = []
        # y_true = []
        with torch.no_grad():
            for batch_ndx, sample in enumerate(dataLoader_val):
                # print(sample[0].shape[0])
                # print("55555")
                oneBatchData = sample[0]
                oneBatchLabel = sample[1]
                oneBatchData = oneBatchData.to(device)
                oneBatchLabel = oneBatchLabel.to(device)
                oneBatchLabel2 = F.one_hot(oneBatchLabel, num_classes=3)
                oneBatchLabel2 = oneBatchLabel2.type(torch.FloatTensor).to(device)
                predictions_val = mstcn1(oneBatchData)
                predictions_val = predictions_val.permute(1, 0, 2, 3)
                # predictions_val= predictions_val.transpose(0,1)
                # predictions_val=torch.flatten(predictions_val,0,1)
                # predictions_val = torch.unsqueeze(predictions_val, 0)
                loss_val = 0
                for p in predictions_val:
                    # print(type(oneBatchLabel))
                    loss_val += loss_ce(
                        p.transpose(2, 1).contiguous().view(-1, 3),
                        oneBatchLabel2.view(-1, 3),
                    )
                    loss_val += 0.15 * torch.mean(
                        torch.clamp(
                            loss_mse(
                                F.log_softmax(p[:, :, 1:], dim=1),
                                F.log_softmax(p.detach()[:, :, :-1], dim=1),
                            ),
                            min=0,
                            max=16,
                        )
                    )
                total_val_loss = total_val_loss + loss_val
                _, predicted_val = torch.max(
                    predictions_val[-1].data, 1
                )  # [30,2,2000] =>max => [30,2000]
                correct_val += (
                    ((predicted_val == oneBatchLabel).float().squeeze(1)).sum().item()
                )
                # f1_score_val += (F1S(oneBatchLabel.cpu(), predicted_val.cpu(),pos_label=1, average='binary'))

                # gt_batch=oneBatchLabel.cpu().flatten()
                # pred_batch=predicted_val.cpu().flatten()
                # y_pred=np.concatenate((y_pred, pred_batch.numpy()), axis=None)
                # y_true=np.concatenate((y_true, gt_batch.numpy()), axis=None)
                if i % 3 == 0:
                    if batch_ndx == 0:
                        y_true = oneBatchLabel.cpu().flatten()
                        y_pred = predicted_val.cpu().flatten()
                    else:
                        # gt_batch=oneBatchLabel.cpu().flatten()
                        # pred_batch=predicted.cpu().flatten()
                        y_pred = torch.cat((y_pred, predicted_val.cpu().flatten()))
                        y_true = torch.cat((y_true, oneBatchLabel.cpu().flatten()))

                # f1_score_val += F1S(oneBatchLabel.cpu().flatten(), predicted_val.cpu().flatten(),average='macro', num_classes=3)
                # kap_val+=cohenkappa(oneBatchLabel.cpu().flatten(), predicted_val.cpu().flatten())
                # mcc_val+=matthews_corrcoef(oneBatchLabel.cpu().flatten(), predicted_val.cpu().flatten())
                f1_score_counter_val += 1
                total_val += len(
                    torch.flatten(oneBatchLabel)
                )  # [30,2000] [batchsize,seqlength]
            print("acc_val = %f" % (float(correct_val) / total_val))
            # print("avg_f1_score_val = %f" % ( float(f1_score_val)/f1_score_counter_val))
            if i % 3 == 0:
                kap_val = cohenkappa(y_true, y_pred)
                mcc_val = matthews_corrcoef(y_true, y_pred)
                print("kappa_val = %f" % (kap_val))
                # print("mcc_val = %f" % (mcc_val))
                # print(conf_matrx)
                # f1_scoreValue=(float(f1_score_val)/f1_score_counter_val)
                # kap_val_avg=float(kap_val)/f1_score_counter_val
                # if i==epoch-1:
                #   torch.save(mstcn1.state_dict(),PATH)     # save the first one as intital one
                if kap_val > best_kappa_val and i > epoch // 2:
                    print("better f1 goted", str(i))
                    best_kappa_val = kap_val
                    torch.save(mstcn1.state_dict(), PATH)
            # model_dict=torch.load(PATH)
        if i <= 3:
            t1 = time.time()
            print("epoch time: ", t1 - t0)

    # test set
    print("*************start to test data************", times)
    mstcn1.load_state_dict(torch.load(PATH))
    i = 0
    for i in range(len(test_list)):
        idx = test_list[i]
        x_test = DL_data[idx]
        y_test = DL_anno[idx]
        x_test = x_test.astype(np.float32)

        added_seg = seg_length - len(y_test) % seg_length
        if added_seg != seg_length:
            # print("let us add some zero data to meet the seg_length")
            # print("initial length of data", len(y_test))
            added_shape2 = x_test[0:added_seg, :]
            added_zero_data2 = np.zeros(added_shape2.shape)
            added_zero_data2 = added_zero_data2.astype(np.float32)
            x_test = np.concatenate((x_test, added_zero_data2), axis=0)
            added_shape2 = y_test[0:added_seg]
            added_zero_data2 = np.zeros(added_shape2.shape)
            added_zero_data2 = added_zero_data2.astype(np.int64)
            y_test = np.concatenate((y_test, added_zero_data2), axis=0)
            # print("after added, length of data", len(x_test))
        else:
            added_seg = 0
            # print("the inigital seg length meets the seg_length")

        (dataset_test) = tf.keras.preprocessing.timeseries_dataset_from_array(
            x_test,
            targets=None,
            sequence_length=seg_length,
            sequence_stride=strde_length,
            batch_size=None,
            sampling_rate=DOWNSAMPLING,
        )
        (target_test) = tf.keras.preprocessing.timeseries_dataset_from_array(
            y_test,
            targets=None,
            sequence_length=seg_length,
            sequence_stride=strde_length,
            batch_size=None,
            sampling_rate=DOWNSAMPLING,
        )
        Sequence_data = list()
        Sequence_label = list()
        for oneSequence in dataset_test:
            Sequence_data.append(oneSequence)
            # Sequence_label.append(oneSequence[:,7])
        for oneSequence in target_test:
            x = tf.dtypes.cast(oneSequence, tf.int64)
            Sequence_label.append(x)
        Sequence_data = tf.convert_to_tensor(Sequence_data)
        Sequence_label = tf.convert_to_tensor(Sequence_label)
        Sequence_data = Sequence_data.numpy()
        Sequence_label = Sequence_label.numpy()

        added_len = Batch_size - len(Sequence_data) % Batch_size
        if added_len != Batch_size:
            # print("let us add some zero data to meet the batchsize")
            # print("initial length of data", len(Sequence_data))
            added_shape = Sequence_data[0:added_len, :, :]
            added_zero_data = np.zeros(added_shape.shape)
            added_zero_data = added_zero_data.astype(np.float32)
            Sequence_data = np.concatenate((Sequence_data, added_zero_data), axis=0)
            added_shape = Sequence_label[0:added_len, :]
            added_zero_data = np.zeros(added_shape.shape)
            added_zero_data = added_zero_data.astype(np.int64)
            Sequence_label = np.concatenate((Sequence_label, added_zero_data), axis=0)
            # print("after added, length of data", len(Sequence_data))
        else:
            added_len = 0
            # print("the inigital length meets the batchsize")
        Sequence_data = torch.from_numpy(Sequence_data)
        Sequence_label = torch.from_numpy(Sequence_label)
        Sequence_data = torch.transpose(Sequence_data, 1, 2)
        dataset_test = TensorDataset(Sequence_data, Sequence_label)
        dataLoader_test = DataLoader(
            dataset_test, batch_size=Batch_size, drop_last=False, shuffle=False
        )
        total_val_loss = 0
        correct_val = 0
        mstcn1 = mstcn1.eval()
        total_val = 0
        kap_val = 0
        mcc_val = 0
        f1_score_val = 0
        f1_score_counter_val = 0
        tp, target_true, pred_true = 0, 0, 0

        # pred_per=[]
        pred_per = np.empty((0, seg_length))
        true_per = np.empty((0, seg_length))
        with torch.no_grad():
            for batch_ndx, sample in enumerate(dataLoader_test):
                # print(sample[0].shape[0])
                # print("55555")
                oneBatchData = sample[0]
                oneBatchLabel = sample[1]
                oneBatchData = oneBatchData.to(device)
                oneBatchLabel = oneBatchLabel.to(device)
                predictions_val = mstcn1(oneBatchData)
                predictions_val = predictions_val.permute(1, 0, 2, 3)
                # predictions_val= predictions_val.transpose(0,1)
                # predictions_val=torch.flatten(predictions_val,0,1)
                # predictions_val = torch.unsqueeze(predictions_val, 0)
                # print("test output size: ", predictions_val.size())
                # print("test Label size: ", oneBatchLabel.size())
                loss_val = 0
                _, predicted_val = torch.max(
                    predictions_val[-1].data, 1
                )  # [30,2,2000] =>max => [30,2000]
                pred_per = np.concatenate(
                    (pred_per, predicted_val.cpu().numpy()), axis=0
                )
                true_per = np.concatenate(
                    (true_per, oneBatchLabel.cpu().numpy()), axis=0
                )
                correct_val += (
                    ((predicted_val == oneBatchLabel).float().squeeze(1)).sum().item()
                )

                # f1_score_val += F1S(oneBatchLabel.cpu().flatten(), predicted_val.cpu().flatten(),average='macro', num_classes=3)
                # kap_val+=cohenkappa(oneBatchLabel.cpu().flatten(), predicted_val.cpu().flatten())
                # mcc_val+=matthews_corrcoef(oneBatchLabel.cpu().flatten(), predicted_val.cpu().flatten())
                f1_score_counter_val += 1
                total_val += len(
                    torch.flatten(oneBatchLabel)
                )  # [30,2000] [batchsize,seqlength]

                if batch_ndx == 0:
                    y_true = oneBatchLabel.cpu().flatten()
                    y_pred = predicted_val.cpu().flatten()
                else:
                    # gt_batch=oneBatchLabel.cpu().flatten()
                    # pred_batch=predicted.cpu().flatten()
                    y_pred = torch.cat((y_pred, predicted_val.cpu().flatten()))
                    y_true = torch.cat((y_true, oneBatchLabel.cpu().flatten()))

            print("acc_val = %f" % (float(correct_val) / total_val))
            # print("avg_f1_score_val = %f" % ( float(f1_score_val)/f1_score_counter_val))
            # print("kappa_val = %f" % (float(kap_val)/f1_score_counter_val))
            # print("mcc_val = %f" % (float(mcc_val)/f1_score_counter_val))
            kap_val = cohenkappa(y_true, y_pred)
            mcc_val = matthews_corrcoef(y_true, y_pred)
            print("kappa_val = %f" % (kap_val))
            print("mcc_val = %f" % (mcc_val))

            # print("initial prediciton length", len(pred_per))
            pred_per2 = pred_per[0 : (len(pred_per) - added_len), :]
            true_per2 = true_per[0 : (len(pred_per) - added_len), :]
            y_true_2 = true_per2.flatten()
            y_pred_2 = pred_per2.flatten()
            conf_matrx_2 = confusion_matrix(y_true_2, y_pred_2)
            print("Confusion Matrix after cutted\n")
            print(conf_matrx_2)
            if len(conf_matrx_2) == 2 and len(conf_matrx_2[0]) == 2:
                print(
                    classification_report(y_true_2, y_pred_2, target_names=["0", "1"])
                )
            else:
                print(
                    classification_report(
                        y_true_2, y_pred_2, target_names=["0", "1", "2"]
                    )
                )
            print("after remove added zero padding prediciton length", len(pred_per2))
            f_name = ".npy"
            f_ptr = results_dir + str(i) + f_name
            np.save(f_ptr, pred_per2)
