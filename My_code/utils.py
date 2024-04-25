import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import metrics
import torch
import torch.nn as nn
import dgl
import dgl.function as fn

# 数据读取
def load_data(directory, random_seed):
    A_DME_D = pd.read_excel(directory + '/A_DME_D.xlsx', header=0, sheet_name='Sheet1').to_numpy()
    A_DME_ME = pd.read_excel(directory + '/A_DME_ME.xlsx', header=0, sheet_name='Sheet1').to_numpy()
    A_DMI_D = pd.read_excel(directory + '/A_DMI_D.xlsx', header=0, sheet_name='Sheet1').to_numpy()
    A_MIME_ME = pd.read_excel(directory + '/A_MIME_ME.xlsx', header=0, sheet_name='Sheet1').to_numpy()
    D_SSM = pd.read_excel(directory + '/disease_Semantic_simi.xlsx', header=0, sheet_name='Sheet1').to_numpy()
    D_GSM = pd.read_excel(directory + '/disease_Gaussian_Simi.xlsx', header=0, sheet_name='Sheet1').to_numpy()
    ME_FSM = pd.read_excel(directory + '/metabolite_func_simi.xlsx', header=0, sheet_name='Sheet1').to_numpy()
    ME_GSM = pd.read_excel(directory + '/metabolite_Gaussian_Simi.xlsx', header=0, sheet_name='Sheet1').to_numpy()
    MI_GSM_1 = pd.read_excel(directory + '/microbe_Gaussian_Simi_1.xlsx', header=0, sheet_name='Sheet1').to_numpy()
    MI_GSM_2 = pd.read_excel(directory + '/microbe_Gaussian_Simi_2.xlsx', header=0, sheet_name='Sheet1').to_numpy()
    all_associations = pd.read_excel(directory + '/association_DME.xlsx', header=0, sheet_name='Sheet1', names=['disease', 'metabolite', 'label'])
    DMI_associations = pd.read_excel(directory + '/association_DMI.xlsx', header=0, sheet_name='Sheet1', names=['disease', 'microbe', 'label'])
    MIME_associations = pd.read_excel(directory + '/association_MIME.xlsx', header=0, sheet_name='Sheet1', names=['microbe', 'metabolite', 'label'])
    IMI = (MI_GSM_1 + MI_GSM_2)/2
    ID = D_SSM
    IME = ME_FSM
    for i in range(D_SSM.shape[0]):
        for j in range(D_SSM.shape[1]):
            if ID[i][j] == 0:
                ID[i][j] = D_GSM[i][j]

    for i in range(ME_FSM.shape[0]):
        for j in range(ME_FSM.shape[1]):
            if IME[i][j] == 0:
                IME[i][j] = ME_GSM[i][j]
    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)

    DMI_associations1 = DMI_associations.loc[DMI_associations['label'] == 1]
    MIME_associations1 = MIME_associations.loc[MIME_associations['label'] == 1]
    sample_df = known_associations.append(random_negative)  # 正负样本1:1采样

# 指针重置
    sample_df.reset_index(drop=True, inplace=True)
    DMI_associations1.reset_index(drop=True, inplace=True)
    MIME_associations1.reset_index(drop=True, inplace=True)
    samples = sample_df.values      # 获得重新编号的新样本
    DMI_associations = DMI_associations1.values
    MIME_associations = MIME_associations1.values
    return A_DME_D, A_DME_ME, A_DMI_D, A_MIME_ME, ID, IME, IMI, samples, DMI_associations, MIME_associations


def build_graph(directory, random_seed, device,sample_num):
    A_DME_D, A_DME_ME, A_DMI_D, A_MIME_ME, ID, IME, IMI, samples, DMI_associations, MIME_associations = load_data(directory, random_seed)
    # 构造D-ME-D元路径的图
    g_D0 = dgl.DGLGraph().to(device)
    g_D0.add_nodes(A_DME_D.shape[0])
    g_D0 = dgl.add_self_loop(g_D0).to(device)
    rows, cols = np.where(A_DME_D == 1)
    g_D0.add_edges(rows, cols)
    d_sim = torch.zeros(g_D0.number_of_nodes(), ID.shape[1]).to(device)
    d_sim[:, :] = torch.from_numpy(ID.astype('float32')).to(device)
    g_D0.ndata['d_sim'] = d_sim.to(device)

    # 构造ME-D-ME元路径的图
    New_adj1 = sage_sample(A_DME_ME, IME,sample_num)
    g_ME0 = dgl.DGLGraph().to(device)
    g_ME0.add_nodes(New_adj1.shape[1])
    g_ME0 = dgl.add_self_loop(g_ME0).to(device)
    rows, cols = np.where(New_adj1 == 1)
    g_ME0.add_edges(rows, cols)
    me_sim = torch.zeros(g_ME0.number_of_nodes(), IME.shape[1]).to(device)
    me_sim[:, :] = torch.from_numpy(IME.astype('float32')).to(device)
    g_ME0.ndata['me_sim'] = me_sim.to(device)

    # 构造D-MI-D元路径的图
    g_D1 = dgl.DGLGraph().to(device)
    g_D1.add_nodes(A_DMI_D.shape[0])
    g_D1 = dgl.add_self_loop(g_D1).to(device)
    rows, cols = np.where(A_DMI_D == 1)
    g_D1.add_edges(rows, cols)
    d_sim = torch.zeros(g_D1.number_of_nodes(), ID.shape[1]).to(device)
    d_sim[:, :] = torch.from_numpy(ID.astype('float32')).to(device)
    g_D1.ndata['d_sim'] = d_sim.to(device)

    # 构造ME-MI-ME元路径的图
    New_adj2 = sage_sample(A_MIME_ME, IME, sample_num)
    g_ME1 = dgl.DGLGraph().to(device)
    g_ME1.add_nodes(New_adj2.shape[1])
    g_ME1 = dgl.add_self_loop(g_ME1).to(device)
    rows, cols = np.where(New_adj2 == 1)
    g_ME1.add_edges(rows, cols)

    me_sim = torch.zeros(g_ME1.number_of_nodes(), IME.shape[1]).to(device)
    me_sim[:, :] = torch.from_numpy(IME.astype('float32')).to(device)
    g_ME1.ndata['me_sim'] = me_sim.to(device)

    g = dgl.DGLGraph().to(device)
    g.add_nodes(ID.shape[0] + IME.shape[0])
    node_type = torch.zeros(g.number_of_nodes(), dtype=torch.int64).to(device)
    node_type[: ID.shape[0]] = 1
    g.ndata['type'] = node_type.to(device)

    d_sim = torch.zeros(g.number_of_nodes(), ID.shape[1]).to(device)
    d_sim[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32')).to(device)
    g.ndata['d_sim'] = d_sim.to(device)

    me_sim = torch.zeros(g.number_of_nodes(), IME.shape[1]).to(device)
    me_sim[ID.shape[0]: ID.shape[0]+IME.shape[0], :] = torch.from_numpy(IME.astype('float32')).to(device)
    g.ndata['me_sim'] = me_sim.to(device)

    disease_ids = list(range(1, ID.shape[0]+1))
    metabolite_ids = list(range(1, IME.shape[0]+1))

    disease_ids_invmap = {id_: i for i, id_ in enumerate(disease_ids)}
    metabolite_ids_invmap = {id_: i for i, id_ in enumerate(metabolite_ids)}

    sample_disease_vertices = [disease_ids_invmap[id_] for id_ in samples[:, 0]]
    sample_metabolite_vertices = [metabolite_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 1]]

    g.add_edges(sample_disease_vertices, sample_metabolite_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})
    g.add_edges(sample_metabolite_vertices, sample_disease_vertices,   # 添加双向边（无向）
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})

    g0 = dgl.DGLGraph().to(device)
    g0.add_nodes(ID.shape[0] + IME.shape[0] + IMI.shape[0])
    node_type = torch.zeros(g0.number_of_nodes(), dtype=torch.int64).to(device)
    node_type[: ID.shape[0]] = 1
    node_type[ID.shape[0] + IME.shape[0]:] = 2
    g0.ndata['type'] = node_type.to(device)

    d_sim = torch.zeros(g0.number_of_nodes(), ID.shape[1]).to(device)
    d_sim[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32')).to(device)
    g0.ndata['d_sim'] = d_sim.to(device)

    me_sim = torch.zeros(g0.number_of_nodes(), IME.shape[1]).to(device)
    me_sim[ID.shape[0]: ID.shape[0]+IME.shape[0], :] = torch.from_numpy(IME.astype('float32')).to(device)
    g0.ndata['me_sim'] = me_sim.to(device)

    mi_sim = torch.zeros(g0.number_of_nodes(), IMI.shape[1]).to(device)
    mi_sim[ID.shape[0]+IME.shape[0]: ID.shape[0]+IME.shape[0]+IMI.shape[0], :] = torch.from_numpy(IMI.astype('float32')).to(device)
    g0.ndata['mi_sim'] = mi_sim.to(device)

    microbe_ids = list(range(1, IMI.shape[0]+1))
    microbe_ids_invmap = {id_: i for i, id_ in enumerate(microbe_ids)}

    dmi_disease_vertices = [disease_ids_invmap[id_] for id_ in DMI_associations[:, 0]]
    dmi_microbe_vertices = [microbe_ids_invmap[id_] + ID.shape[0] + IME.shape[0] for id_ in DMI_associations[:, 1]]
    mime_microbe_vertices = [microbe_ids_invmap[id_] + ID.shape[0] + IME.shape[0] for id_ in MIME_associations[:, 0]]
    mime_metabolite_vertices = [metabolite_ids_invmap[id_] + ID.shape[0] for id_ in MIME_associations[:, 1]]

    g0.add_edges(sample_disease_vertices, sample_metabolite_vertices,
                data={'dme': torch.from_numpy(samples[:, 2].astype('float32'))})
    g0.add_edges(sample_metabolite_vertices, sample_disease_vertices,
                data={'med': torch.from_numpy(samples[:, 2].astype('float32'))})
    g0.add_edges(dmi_disease_vertices, dmi_microbe_vertices,
                data={'dmi': torch.from_numpy(DMI_associations[:, 2].astype('float32'))})
    g0.add_edges(dmi_microbe_vertices, dmi_disease_vertices,
                data={'mid': torch.from_numpy(DMI_associations[:, 2].astype('float32'))})
    g0.add_edges(mime_microbe_vertices, mime_metabolite_vertices,
                data={'mime': torch.from_numpy(MIME_associations[:, 2].astype('float32'))})
    g0.add_edges(mime_metabolite_vertices, mime_microbe_vertices,
                data={'memi': torch.from_numpy(MIME_associations[:, 2].astype('float32'))})
    return g_D0, g_ME0, g_D1, g_ME1, g, g0, sample_disease_vertices, sample_metabolite_vertices, ID, IME, IMI, samples, DMI_associations, MIME_associations

def sage_sample(Adj, Fea, num_neighbors):
    num_nodes = Adj.shape[0]
    weights_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if Adj[i][j] == 1:
                weight = np.dot(Fea[i], Fea[j])
                weights_matrix[i][j] = weight
    top_k_neighbors = num_neighbors
    num_nodes = weights_matrix.shape[0]
    new_adj_matrix = np.zeros_like(weights_matrix)
    for i in range(num_nodes):
        weights_vector = weights_matrix[i]
        nonzero_indices = np.where(weights_vector != 0)[0]
        nonzero_weights = weights_vector[nonzero_indices]
        sorted_indices = np.argsort(nonzero_weights)[::-1]  # 降序排序
        selected_indices = nonzero_indices[sorted_indices[:top_k_neighbors]]
        new_adj_matrix[i, selected_indices] = 1
    return new_adj_matrix

def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

def plot_auc_curves(fprs, tprs, auc, directory, name):
    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f' % (i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    plt.plot(mean_fpr, mean_tpr, color='BlueViolet', alpha=0.9, label='Mean AUC: %.4f' % mean_auc)
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.savefig(directory+'/%s.pdf' % name, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prc_curves(precisions, recalls, prc, directory, name):
    mean_recall = np.linspace(0, 1, 20000)
    precision = []
    for i in range(len(recalls)):
        precision.append(interp(1-mean_recall, 1-recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.4, linestyle='--', label='Fold %d AUPR: %.4f' % (i + 1, prc[i]))
    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    # mean_prc = metrics.auc(mean_recall, mean_precision)
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    plt.plot(mean_recall, mean_precision, color='BlueViolet', alpha=0.9,
             label='Mean AUPR: %.4f' % mean_prc)  # AP: Average Precision
    plt.plot([1, 0], [0, 1], linestyle='--', color='black', alpha=0.4)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.legend(loc='lower left')
    plt.savefig(directory + '/%s.pdf' % name, dpi=300, bbox_inches='tight')
    plt.close()