
import numpy as np
import wntr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from QuickTraining import QuickTraining


def main():
    '''################# load WDN model data #################'''
    filename = 'Network.inp'
    wn = wntr.network.WaterNetworkModel(filename)
    ID = wn.node_name_list
    num_nodes = len(ID)
    if wn.patterns:
        pattern_name = list(wn.patterns.keys())[0]
        pattern_length = len(wn.patterns[pattern_name].multipliers)
    else:
        print("No patterns found in the model.")

    net_adj = np.load('WDN_net_adj.npy')
    att_adj = np.zeros((num_nodes, pattern_length + 1))

    '''################# start the embedding process #################'''
    embedding = QuickTraining(net_adj, att_adj, label=None, batch_size=50, shuffle=False, net_hadmard_coeff=100.0,
                  att_hadmard_coeff=5.0, net_feature_num=512, att_feature_num=512, net_losscoef_1=1,
                  net_losscoef_2=5e-2, att_semanticcoef_1=1e-3, att_squarelosscoef_2=1, net_lr=1e-2, att_lr=1e-2,
                  net_epoch=2000, att_epoch=1, phase='net')

    '''################# Analyze and plot results #################'''
    originalLabels = dict() 
    patternList = []
    for i in range(len(ID)):
        useNode = wn.get_node(ID[i])
        try:
            readPattern = useNode.demand_timeseries_list[0].pattern_name
            originalLabels[ID[i]] = int(readPattern)
            patternList.append(int(readPattern)-1)
        except:
            patternList.append(1)
            pass

    pca1 = PCA(n_components = 2)
    pca1 = pca1.fit(embedding)
    embedding_PCA = pca1.transform(embedding)
    drawAttr = originalLabels.copy()

    fig, axs = plt.subplots() 
    cm = plt.cm.get_cmap('Spectral_r')

    useaxs = fig.add_subplot(1, 2, 1)
    wntr.graphics.plot_network(wn, ax=useaxs, node_attribute=drawAttr, node_labels=False,
        node_size=22, link_width=0.5, add_colorbar=False, node_color=patternList)

    useaxs = fig.add_subplot(1, 2, 2)
    useaxs.scatter(embedding_PCA[:, 0], embedding_PCA[:, 1], s=13, c=patternList, cmap=cm)

    plt.show()


if __name__ == '__main__':
    main()


