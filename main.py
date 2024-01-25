import argparse
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import time
import yaml
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1, 2, 3'
import torch
import visdom
import get_cls_map
from model import utils, WFCG
from loadData import data_reader, split_data
from createGraph import rdSLIC, create_graph

# 设置 PyTorch 的随机数种子
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='FDGC')
parser.add_argument('--device', type=str, default='cuda:2' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--path-config', type=str, default='/home3/ywl/PycharmProject/WFCG/config/config.yaml')
parser.add_argument('-pc', '--print-config', action='store_true', default=False)
parser.add_argument('-pdi', '--print-data-info', action='store_true', default=False)
parser.add_argument('-sr', '--show-results', action='store_true', default=False)
parser.add_argument('--save-results', action='store_true', default=True)
args = parser.parse_args()  # running in command line

print(args.device)


# load data
def load_data():
    # data = data_reader.IndianRaw().normal_cube  # 加载数据的同时进行归一化了
    # data_gt = data_reader.IndianRaw().truth  # ground_truth
    # data = data_reader.yaolin().normal_cube
    # data_gt = data_reader.yaolin().truth
    # data = data_reader.jiaqi_yue().normal_cube
    # data_gt = data_reader.jiaqi_yue().truth
    data = data_reader.PaviaURaw().normal_cube  # 加载数据的同时进行归一化了
    data_gt = data_reader.PaviaURaw().truth  # ground_truth
    # data = data_reader.Hong_HU_Raw().normal_cube  # 加载数据的同时进行归一化了
    # data_gt = data_reader.Hong_HU_Raw().truth  # ground_truth

    return data, data_gt


# 对高光谱数据 X 应用 PCA 变换
# def applyPCA(X, numComponents):
#     newX = np.reshape(X, (-1, X.shape[2]))
#     pca = PCA(n_components=numComponents, whiten=True)
#     newX = pca.fit_transform(newX)
#     newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
#
#     return newX

# viz = visdom.Visdom(env="WFCG")
# if not viz.check_connection:
#     print("visdom is not connected.Did you run 'python -m visdom.server'?")

data, data_gt = load_data()
class_num = np.max(data_gt)
height, width, bands = data.shape
gt_reshape = np.reshape(data_gt, [-1])  # (h*w,)
# load config

config = yaml.load(open(args.path_config, "r"), Loader=yaml.FullLoader)
dataset_name = config["data_input"]["dataset_name"]
samples_type = config["data_split"]["samples_type"]
train_num = config["data_split"]["train_num"]  # 无作用
val_num = config["data_split"]["val_num"]  # 无作用
train_ratio = config["data_split"]["train_ratio"]
val_ratio = config["data_split"]["val_ratio"]
superpixel_scale = config["data_split"]["superpixel_scale"]
max_epoch = config["network_config"]["max_epoch"]
learning_rate = config["network_config"]["learning_rate"]
weight_decay = config["network_config"]["weight_decay"]
lb_smooth = config["network_config"]["lb_smooth"]
path_weight = config["result_output"]["path_weight"]
path_result = config["result_output"]["path_result"]

if args.print_config:
    print(config)

runs = 5
list_result = []
overall = 0
for j in range(runs):

    # split datasets
    train_index, val_index, test_index = split_data.split_data(gt_reshape,
                                                               class_num, train_ratio, val_ratio, train_num, val_num,
                                                               samples_type)

    # create graph  获得标签，shape为【h*w，】
    train_samples_gt, test_samples_gt, val_samples_gt = create_graph.get_label(gt_reshape,
                                                                               train_index, val_index,
                                                                               test_index)
    # print(train_samples_gt.shape)                  # (21025,)

    # shape为（h*w，c），只有选择到的像素行为全为1
    train_label_mask, test_label_mask, val_label_mask = create_graph.get_label_mask(train_samples_gt,
                                                                                    test_samples_gt, val_samples_gt,
                                                                                    data_gt, class_num)
    # label transfer to one-hot encode
    train_gt = np.reshape(train_samples_gt, [height, width])  # （h,w)
    test_gt = np.reshape(test_samples_gt, [height, width])
    val_gt = np.reshape(val_samples_gt, [height, width])

    if args.print_data_info:
        data_reader.data_info(train_gt, val_gt, test_gt)

    # 创建标签的one-hot向量
    train_gt_onehot = create_graph.label_to_one_hot(train_gt, class_num)  # shape（h*w，c） c为类别的数量
    test_gt_onehot = create_graph.label_to_one_hot(test_gt, class_num)
    val_gt_onehot = create_graph.label_to_one_hot(val_gt, class_num)

    # superpixels划分
    ls = rdSLIC.LDA_SLIC(data, train_gt, class_num - 1)
    tic0 = time.time()
    Q, S, A, Seg = ls.simple_superpixel(
        scale=superpixel_scale)  # superpixel_scale=100  Q是看像素属于哪个超像素，S是超像素的特征（由所有像素特征的平均值求得），A是超像素的邻接矩阵（值为不同超像素之间的相似度），Segments是超像素分割后的二维数组（每个位置的值表示该像属于哪个超像素中）
    toc0 = time.time()
    LDA_SLIC_Time = toc0 - tic0

    print("LDA_SLIC_Time", LDA_SLIC_Time)

    Q = torch.from_numpy(Q).to(args.device)
    A = torch.from_numpy(A).to(args.device)

    train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(args.device)
    test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(args.device)
    val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(args.device)

    train_gt_onehot = torch.from_numpy(train_gt_onehot.astype(np.float32)).to(args.device)
    test_gt_onehot = torch.from_numpy(test_gt_onehot.astype(np.float32)).to(args.device)
    val_gt_onehot = torch.from_numpy(val_gt_onehot.astype(np.float32)).to(args.device)

    train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(args.device)
    test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(args.device)
    val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(args.device)

    net_input = np.array(data, np.float32)

    # 尝试使用pca降维
    # net_input = applyPCA(net_input, numComponents=100).astype(np.float32)

    net_input = torch.from_numpy(net_input.astype(np.float32)).to(args.device)
    # model
    net = WFCG.WFCG(height, width, bands, class_num, Q, A).to(args.device)
    # net = torch.nn.DataParallel(net, device_ids=[0, 1, 2])
    # train
    print("\n\n==================== train ====================\n")
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)  # , weight_decay=0.0001
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True,
    #                                                        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
    #                                                        eps=1e-08)
    zeros = torch.zeros([height * width]).to(args.device).float()
    tic1 = time.time()
    best_loss = 99999

    win = None
    win1 = None
    for i in range(max_epoch + 1):
        optimizer.zero_grad()  # zero the gradient buffers
        net.train()
        output = net(net_input)
        loss = utils.compute_loss(output, train_gt_onehot, train_label_mask)
        loss.backward(retain_graph=False)
        optimizer.step()  # Does the update

        # if i%10==0:
        with torch.no_grad():
            net.eval()
            output = net(net_input)
            trainloss = utils.compute_loss(output, train_gt_onehot, train_label_mask)
            trainOA = utils.evaluate_performance(output, train_samples_gt, train_gt_onehot, zeros)
            valloss = utils.compute_loss(output, val_gt_onehot, val_label_mask)
            valOA = utils.evaluate_performance(output, val_samples_gt, val_gt_onehot, zeros)
            # print("{}\ttrain loss={:.4f}\t train OA={:.4f} val loss={:.4f}\t val OA={:.4f}".format(str(i + 1), trainloss, trainOA, valloss, valOA))

            if valloss < best_loss:
                best_loss = valloss
                torch.save(net.state_dict(), path_weight + r"model.pt")
                # print('save model...')
        # scheduler.step(valloss)
        torch.cuda.empty_cache()
        net.train()

        # if win is None:
        #     win = viz.line(X=np.array([i]), Y=np.array([valloss.data.cpu().numpy()]),
        #                    opts=dict(title='Epoch Val_Loss lr=0.001 india', xlabel='Epoch',
        #                              ylabel='val_loss'))
        # else:
        #     viz.line(X=np.array([i]), Y=np.array([valloss.data.cpu().numpy()]), win=win, update='append')
        # if win1 is None:
        #     win1 = viz.line(X=np.array([i]), Y=np.array([valOA.cpu().detach().numpy()]),
        #                     opts=dict(title='Epoch valAccuracy lr=0.001', xlabel='Epoch',
        #                               ylabel='val_accuracy'))
        # else:
        #     viz.line(X=np.array([i]), Y=np.array([valOA.cpu().detach().numpy()]), win=win1, update='append')

        if i % 10 == 0:
            print(
                "{}\ttrain loss={:.4f}\t train OA={:.4f} val loss={:.4f}\t val OA={:.4f}".format(str(i + 1), trainloss,
                                                                                                 trainOA, valloss,
                                                                                                 valOA))
    toc1 = time.time()
    print("training time:", toc1 - tic1)
    print("\n\n====================training done. starting evaluation...========================\n")

    # test
    torch.cuda.empty_cache()
    with torch.no_grad():
        net.load_state_dict(torch.load(path_weight + r"model.pt"))
        net.eval()
        tic2 = time.time()
        output = net(net_input)
        # print("output.shape", output.shape)
        toc2 = time.time()
        testloss = utils.compute_loss(output, test_gt_onehot, test_label_mask)
        testOA = utils.evaluate_performance(output, test_samples_gt, test_gt_onehot, zeros)

        # 这里是为了画出结果图 output.shape(h*w,c类别长度）
        ypred = np.argmax(output.cpu().numpy(), axis=1)
        ypred = np.reshape(ypred, (data_gt.shape[0], data_gt.shape[1]))
        get_cls_map.get_cls_map(y_pred=ypred, y=data_gt, j=j)

        print("{}\ttest loss={:.4f}\t test OA={:.4f}".format(str(i + 1), testloss, testOA))
    overall += testOA
    # list_result.append(overall)
    print("test time:", toc2 - tic2)
    torch.cuda.empty_cache()
    del net
print("University of Pavia_0.1%train_dataset five times average OA", overall / runs)
LDA_SLIC_Time = toc0 - tic0
# print("LDA-SLIC costs time: {}".format(LDA_SLIC_Time))
training_time = toc1 - tic1 + LDA_SLIC_Time
testing_time = toc2 - tic2 + LDA_SLIC_Time
training_time, testing_time

# classification report
test_label_mask_cpu = test_label_mask.cpu().numpy()[:, 0].astype('bool')
test_samples_gt_cpu = test_samples_gt.cpu().numpy().astype('int64')
predict = torch.argmax(output, 1).cpu().numpy()
classification = classification_report(test_samples_gt_cpu[test_label_mask_cpu],
                                       predict[test_label_mask_cpu] + 1, digits=4)
kappa = cohen_kappa_score(test_samples_gt_cpu[test_label_mask_cpu], predict[test_label_mask_cpu] + 1)

if args.show_results:
    print(classification, kappa)

# store results
if args.save_results:
    print("save results")
    run_date = time.strftime('%Y%m%d-%H%M-', time.localtime(time.time()))
    f = open(path_result + run_date + dataset_name + '.txt', 'a+')
    str_results = '\n ======================' \
                  + '\nrun data = ' + run_date \
                  + "\nlearning rate = " + str(learning_rate) \
                  + "\nepochs = " + str(max_epoch) \
                  + "\nsamples_type = " + str(samples_type) \
                  + "\ntrain ratio = " + str(train_ratio) \
                  + "\nval ratio = " + str(val_ratio) \
                  + "\ntrain num = " + str(train_num) \
                  + "\nval num = " + str(val_num) \
                  + '\ntrain time = ' + str(training_time) \
                  + '\ntest time = ' + str(testing_time) \
                  + '\n' + classification \
                  + "kappa = " + str(kappa) \
                  + '\n'
    f.write(str_results)
    f.close()
