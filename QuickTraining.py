
import numpy as np
import torch
from DataBatchGenerator import DataBatchGenerator, DataBatchGenerator_netOnly, DataBatchGenerator_MultiAttrOnly
from Model_ import GraphEModel


def QuickTraining(net_adj, att_adj, label=None, batch_size=40, shuffle=False, net_hadmard_coeff=10.0,
                  att_hadmard_coeff=5.0,net_feature_num=512,att_feature_num=512, net_losscoef_1=1,
                  net_losscoef_2=1, att_semanticcoef_1=1, att_squarelosscoef_2=1, net_lr=1e-2, att_lr=1e-2,
                  net_epoch=300, att_epoch=300, phase='att'):

    if label is None:
        label_vec = np.zeros(att_adj.shape[0])
    else:
        label_vec = label

    batchGenerator = DataBatchGenerator(net_adj, att_adj, label_vec, batch_size, shuffle, net_hadmard_coeff, att_hadmard_coeff)

    WDNE_net_layers_list = [
        {'type': 'DENSE', 'features': net_feature_num, 'act_funtion': 'RELU', 'bias': True}
    ]
    WDNE_att_layers_list = [
        {'type': 'DENSE', 'features': att_feature_num, 'act_funtion': 'RELU', 'bias': True}
    ]
    WDNE_loss_settings_list = {
        'all': [
        ],
        'net': [
            {'loss_name': "structur_proximity_1order", 'coef': net_losscoef_1},
            {'loss_name': "structur_proximity_2order", 'coef': net_losscoef_2},
        ],
        'att': [
            {'loss_name': "semantic_proximity_2order", 'coef': att_semanticcoef_1},
            {'loss_name': "square_diff_embedding_proximity", 'coef': att_squarelosscoef_2},
        ],
    }

    WDNE_optimizator_net_settings_list = {
            "opt_name": "adam",
            "lr_rate": net_lr
    }
    WDNE_optimizator_att_settings_list = {
            "opt_name": "adam",
            "lr_rate": att_lr
    }

    WDNE_regularization_net_settings_list = [
        {'reg_name': 'L2', 'coeff': 0.001}
    ]

    WDNE_regularization_att_settings_list = [
        {'reg_name': 'L2', 'coeff': 0.001}
    ]

    WDNE_checkpoint_config ={
        "type": ["best_model_loss", "first_train", "last_train"],
        "times": 20,
        "overwrite": False,
        "path_file": f"/content/models_checkpoint/WDNE_temp",
        "name_file": f"WDNE_temp_checkpoint",
        "path_not_exist": "create",
        "path_exist": "use",
    }

    WDNE_config = {
        "net_dim": net_adj.shape[1],
        "net_layers_list": WDNE_net_layers_list,
        "net_latent_dim": net_feature_num,
        "att_dim": att_adj.shape[1],
        "att_layers_list": WDNE_att_layers_list,
        "att_latent_dim": att_feature_num,

    "loss_functions": WDNE_loss_settings_list,

    "optimizator_net": WDNE_optimizator_net_settings_list,
    "optimizator_att": WDNE_optimizator_att_settings_list,

    "regularization_net": WDNE_regularization_net_settings_list,
    "regularization_att": WDNE_regularization_att_settings_list,

    "checkpoint_config": WDNE_checkpoint_config,
    "model_name": "WDNE_wiki_opt1",
    "training_config": "N>A",
    }

    WDNE_model = GraphEModel(WDNE_config)
    WDNE_epochs_config = {
        'att': att_epoch,
        'net': net_epoch,
    }

    print(torch.cuda.is_available())
    device = torch.device('cuda')
    WDNE_model.to(device)
    WDNE_model.cuda()

    WDNE_model.models_training(datagenerator=batchGenerator, loss_verbose=False, epochs=WDNE_epochs_config, path_embedding=f"/content/models_checkpoint/WDNE_temp/")
    embedding = WDNE_model.get_embedding(phase=phase, type_output="numpy")

    return embedding
    

def QuickTraining_multiAttr(net_adj, multiAtt, IndexofSamples, label = None, net_batch_size=40,
                  netshuffle = False, att_batch_size=5, attShuffle = True, net_hadmard_coeff=10.0,
                  att_hadmard_coeff=5.0, net_feature_num=512,att_feature_num=512, net_losscoef_1=1,
                  net_losscoef_2=1, att_semanticcoef_1=1, att_squarelosscoef_2=1, net_lr=1e-2, att_lr=1e-2,
                  net_epoch=300, att_epoch=300):

    if label is None:
        label_vec = np.zeros(net_adj.shape[0])
    else:
        label_vec = label

    net_batchGenerator = DataBatchGenerator_netOnly(net_adj, label_vec, net_batch_size, netshuffle, net_hadmard_coeff)
    att_batchGenerator = DataBatchGenerator_MultiAttrOnly(net_adj, multiAtt, label_vec, att_batch_size, attShuffle, IndexofSamples)
    batchGenerator = {'net': net_batchGenerator, 'att': att_batchGenerator}

    WDNE_net_layers_list = [
        {'type': 'DENSE', 'features': net_feature_num, 'act_funtion': 'RELU','bias':True}
    ]
    WDNE_att_layers_list = [
        {'type': 'DENSE', 'features': att_feature_num, 'act_funtion': 'RELU','bias':True}
    ]
    WDNE_loss_settings_list = {
        'all': [
        ],
        'net': [
            {'loss_name': "structur_proximity_1order", 'coef': net_losscoef_1},
            {'loss_name': "structur_proximity_2order", 'coef': net_losscoef_2},
        ],
        'att': [
            {'loss_name': "semantic_proximity_2order", 'coef': att_semanticcoef_1},
            {'loss_name': "square_diff_embedding_proximity", 'coef': att_squarelosscoef_2},
        ],
    }
    WDNE_optimizator_net_settings_list = {
            "opt_name": "adam",
            "lr_rate": net_lr
    }
    WDNE_optimizator_att_settings_list = {
            "opt_name": "adam",
            "lr_rate": att_lr
    }

    WDNE_regularization_net_settings_list = [
        {'reg_name': 'L2', 'coeff': 0.001}
    ]

    WDNE_regularization_att_settings_list = [
        {'reg_name': 'L2', 'coeff': 0.001}
    ]

    WDNE_checkpoint_config ={
        "type": ["best_model_loss", "first_train", "last_train"],
        "times": 20,
        "overwrite": False,
        "path_file": f"/content/models_checkpoint/WDNE_temp",
        "name_file": f"WDNE_temp_checkpoint",
        "path_not_exist": "create",
        "path_exist": "use",
    }

    WDNE_config = {
        "net_dim": net_adj.shape[1],
        "net_layers_list": WDNE_net_layers_list,
        "net_latent_dim": net_feature_num,

        "att_dim": multiAtt[0].shape[1],
        "att_layers_list": WDNE_att_layers_list,
        "att_latent_dim": att_feature_num,

    "loss_functions": WDNE_loss_settings_list,

    "optimizator_net": WDNE_optimizator_net_settings_list,
    "optimizator_att": WDNE_optimizator_att_settings_list,

    "regularization_net": WDNE_regularization_net_settings_list,
    "regularization_att": WDNE_regularization_att_settings_list,

    "checkpoint_config": WDNE_checkpoint_config,
    "model_name": "WDNE_temp",
    "training_config": "multiAttr",
    }

    WDNE_model = GraphEModel(WDNE_config)
    WDNE_epochs_config = {
        'att': att_epoch,
        'net': net_epoch,
    }

    print(torch.cuda.is_available())
    device = torch.device('cuda')
    WDNE_model.to(device)
    WDNE_model.cuda()

    results = WDNE_model.models_training(datagenerator=batchGenerator, loss_verbose=False, epochs=WDNE_epochs_config, path_embedding=f"/content/models_checkpoint/WDNE_temp/")

    return results

