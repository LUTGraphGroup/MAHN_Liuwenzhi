import time
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn import metrics
from utils import load_data, build_graph, weight_reset
from model import MAHN
import dgl


def Train(device, directory, epochs, attn_size, attn_heads, out_dim, dropout, slope, lr, wd, random_seed, sample_num, model_type):
    print('sample_num',sample_num)
    print('dropout',dropout)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    g_D0, g_ME0, g_D1, g_ME1, g, g0, disease_vertices, metabolite_vertices, ID, IME, IMI, samples, \
    DMI_associations, MIME_associations = build_graph(directory, random_seed,device, sample_num)
    samples_df = pd.DataFrame(samples, columns=['disease', 'metabolite', 'label'])

    print('## vertices:', g.number_of_nodes())
    print('## edges:', g.number_of_edges())
    print('## disease nodes:', torch.sum(g.ndata['type'] == 1).cpu().numpy())
    print('## metabolite nodes: ', torch.sum(g.ndata['type'] == 0).cpu().numpy())
    print('## microbe nodes: ', torch.sum(g0.ndata['type'] == 2).cpu().numpy())

    auc_result = []
    acc_result = []
    pre_result = []
    recall_result = []
    f1_result = []
    prc_result = []

    fprs = []
    tprs = []
    precisions = []
    recalls = []

# 设置五折交叉验证
    i = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    for train_idx, test_idx in kf.split(samples[:, 2]):
        i += 1
        print('Training for Fold', i)
# 将训练集的标记为1，(测试集)其余为0
        samples_df['train'] = 0
        samples_df['train'].iloc[train_idx] = 1
        train_tensor = torch.from_numpy(samples_df['train'].values.astype('int64')).to(device)
        edge_data = {'train': train_tensor}
# 对g和g0中疾病和metabolite的两个异质图的训练边进行标记
        g.edges[disease_vertices, metabolite_vertices].data.update(edge_data)
        g.edges[metabolite_vertices, disease_vertices].data.update(edge_data)

        g0.edges[disease_vertices, metabolite_vertices].data.update(edge_data)
        g0.edges[metabolite_vertices, disease_vertices].data.update(edge_data)

        train_eid = g.filter_edges(lambda edges: edges.data['train']).to(device)
        g_train = g.edge_subgraph(train_eid, preserve_nodes=True)
        g_train = g_train.to(device)
        g_train0 = g0.edge_subgraph(train_eid, preserve_nodes=True)
        g_train0 = g_train0.to(device)

        # 训练标签和测试标签用于最后对比
        label_train = g_train.edata['label'].unsqueeze(1).to(device)
        src_train, dst_train = g_train.all_edges()

        test_eid = g.filter_edges(lambda edges: edges.data['train'] == 0).to(device)
        src_test, dst_test = g.find_edges(test_eid)
        label_test = g.edges[test_eid].data['label'].unsqueeze(1).to(device)
        print('## Training edges:', len(train_eid))
        print('## Testing edges:', len(test_eid))

        if model_type == 'MAHN':
            model = MAHN(G_D0=g_D0, G_D1=g_D1, G_ME0=g_ME0, G_ME1=g_ME1,
                           hidden_dim=64, G=g_train0,
                           meta_paths_list=['dme', 'med', 'dmi', 'mime'],
                           feature_attn_size=attn_size,
                           num_heads=attn_heads,
                           num_diseases=ID.shape[0],
                           num_metabolite=IME.shape[0],
                           num_microbe=IMI.shape[0],
                           d_sim_dim=ID.shape[1],
                           me_sim_dim=IME.shape[1],
                           mi_sim_dim=IMI.shape[1],
                           out_dim=out_dim,
                           dropout=dropout,
                           slope=slope, device=device
                           )
            model.apply(weight_reset)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            loss = nn.BCELoss()  # Binary Cross Entropy Loss二元交叉熵损失函数

            best_val_auc = 0
            best_epoch = 0
            for epoch in range(epochs):
                start = time.time()
                model.train()  # 训练

                with torch.autograd.set_detect_anomaly(True):
                    g_D0 = g_D0.to(device)
                    g_D1 = g_D1.to(device)
                    g_ME0 = g_ME0.to(device)
                    g_ME1 = g_ME1.to(device)
                    g_train0 = g_train0.to(device)
                    g0 = g0.to(device)
                    src_train = src_train.to(device)
                    dst_train = dst_train.to(device)

                    score_train = model(g_D0, g_D1, g_ME0, g_ME1, g_train0, g0, src_train, dst_train).to(device)
                    loss_train = loss(score_train, label_train).to(device)

                    optimizer.zero_grad()
                    loss_train.backward()
                    optimizer.step()

                model.eval()  # 验证
                with torch.no_grad():
                    score_val = model(g_D0, g_D1, g_ME0, g_ME1, g, g0, src_test, dst_test).to(device)
                    loss_val = loss(score_val, label_test).to(device)

                score_train_cpu = np.squeeze(score_train.detach().cpu().numpy())
                score_val_cpu = np.squeeze(score_val.detach().cpu().numpy())
                label_train_cpu = np.squeeze(label_train.detach().cpu().numpy())
                label_val_cpu = np.squeeze(label_test.detach().cpu().numpy())

                train_auc = metrics.roc_auc_score(label_train_cpu, score_train_cpu)
                val_auc = metrics.roc_auc_score(label_val_cpu, score_val_cpu)

                pred_val = [0 if j < 0.5 else 1 for j in score_val_cpu]
                acc_val = metrics.accuracy_score(label_val_cpu, pred_val)
                pre_val = metrics.precision_score(label_val_cpu, pred_val)
                recall_val = metrics.recall_score(label_val_cpu, pred_val)
                f1_val = metrics.f1_score(label_val_cpu, pred_val)
                end = time.time()
                print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.item(),
                      'Val Loss: %.4f' % loss_val.cpu().detach().numpy(),
                      'Acc: %.4f' % acc_val, 'Pre: %.4f' % pre_val, 'Recall: %.4f' % recall_val, 'F1: %.4f' % f1_val,
                      'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc, 'Time: %.2f' % (end - start))

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), 'best_model.pth')  # 保存每一折的最优模型

            model.load_state_dict(torch.load('best_model.pth'))  # 重载最优模型进行测试
            model.eval()  # 测试
            with torch.no_grad():
                score_test = model(g_D0, g_D1, g_ME0, g_ME1, g, g0, src_test, dst_test).to(device)

            score_test_cpu = np.squeeze(score_test.detach().cpu().numpy())

            label_test_cpu = np.squeeze(label_test.detach().cpu().numpy())

            fpr, tpr, thresholds = metrics.roc_curve(label_test_cpu, score_test_cpu)
            precision, recall, _ = metrics.precision_recall_curve(label_test_cpu, score_test_cpu)
            test_auc = metrics.auc(fpr, tpr)
            test_prc = metrics.auc(recall, precision)

            pred_test = [0 if j < 0.5 else 1 for j in score_test_cpu]
            acc_test = metrics.accuracy_score(label_test_cpu, pred_test)
            pre_test = metrics.precision_score(label_test_cpu, pred_test)
            recall_test = metrics.recall_score(label_test_cpu, pred_test)
            f1_test = metrics.f1_score(label_test_cpu, pred_test)

            print('Fold:', i, 'Best Epoch:', best_epoch, 'Test acc: %.4f' % acc_test, 'Test Pre: %.4f' % pre_test,
                  'Test Recall: %.4f' % recall_test, 'Test F1: %.4f' % f1_test, 'Test PRC: %.4f' % test_prc,
                  'Test AUC: %.4f' % test_auc)

            auc_result.append(test_auc)
            acc_result.append(acc_test)
            pre_result.append(pre_test)
            recall_result.append(recall_test)
            f1_result.append(f1_test)
            prc_result.append(test_prc)

            fprs.append(fpr)
            tprs.append(tpr)
            precisions.append(precision)
            recalls.append(recall)

    print('## Training Finished !')
    print('-----------------------------------------------------------------------------------------------')
    print('Auc', auc_result)
    print('Acc', acc_result)
    print('Pre', pre_result)
    print('Recall', recall_result)
    print('F1', f1_result)
    print('Prc', prc_result)
    print('AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc_result), np.std(auc_result)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre_result), np.std(pre_result)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_result), np.std(recall_result)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_result), np.std(f1_result)),
          'PRC mean: %.4f, variance: %.4f \n' % (np.mean(prc_result), np.std(prc_result)))
    print('fprs', fprs)
    print('tprs', tprs)
    print('precisions', precisions)
    print('recalls', recalls)
    return fprs, tprs, auc_result, precisions, recalls, prc_result


