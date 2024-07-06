import numpy as np
import torch


class DataBatchGenerator():

    def __init__(self, net, att, labels, batch_size, shuffle, net_hadmard_coeff, att_hadmard_coeff):
        self.net = net
        self.att = att
        self.labels = labels
        self.number_of_samples = len(att)
        self.batch_size = batch_size
        self.number_of_batches = self.number_of_samples / batch_size
        self.shuffle = shuffle
        self.net_hadmard_coeff = net_hadmard_coeff
        self.att_hadmard_coeff = att_hadmard_coeff

    def generate(self):
        sample_index = np.arange(self.net.shape[0])

        counter = 0
        if self.shuffle:
            np.random.shuffle(sample_index)

        while (counter*(self.batch_size) <= self.number_of_samples):
            start_samples_index = self.batch_size * counter
            end_samples_index = self.batch_size * (counter + 1)

            if end_samples_index > self.number_of_samples:
                end_samples_index = self.number_of_samples
                start_samples_index = end_samples_index-self.batch_size

            # list of samples's index
            samples_index = sample_index[start_samples_index: end_samples_index]

            # submatrix of W and A, cut for sample index
            net_batch = self.net[samples_index, :]
            att_batch = self.att[samples_index, :]
            net_batch_adj = self.net[samples_index, :][:, samples_index]
            node_label = self.labels[samples_index]
            node_index = samples_index

            # B_net and B_att param of hadmard operation
            B_net = np.ones(net_batch.shape)
            B_net[net_batch != 0] = self.net_hadmard_coeff

            B_att = np.ones(att_batch.shape)
            B_att[att_batch != 0] = self.att_hadmard_coeff

            # trasform np array to tensor
            net_batch_tensor = torch.from_numpy(net_batch).float()
            att_batch_tensor = torch.from_numpy(att_batch).float()
            net_batch_adj_tensor = torch.from_numpy(net_batch_adj).float()
            B_net_tensor = torch.from_numpy(B_net).float()
            B_att_tensor = torch.from_numpy(B_att).float()

            inputs = [net_batch_tensor.cuda(), att_batch_tensor.cuda(), net_batch_adj_tensor.cuda()]
            B_params = [B_net_tensor.cuda(), B_att_tensor.cuda()]
            batch_info = [node_index, node_label]

            # feed the fit() function with new data
            yield inputs, B_params, batch_info
            counter += 1


class DataBatchGenerator_netOnly():
    def __init__(self, net, labels, batch_size, shuffle, net_hadmard_coeff):
        self.net = net
        self.labels = labels
        self.number_of_samples = len(net)
        self.batch_size = batch_size
        self.number_of_batches = self.number_of_samples / batch_size
        self.shuffle = shuffle
        self.net_hadmard_coeff = net_hadmard_coeff

    def generate(self):
        sample_index = np.arange(self.net.shape[0])

        counter = 0
        if self.shuffle:
            np.random.shuffle(sample_index)

        while (counter*(self.batch_size) <= self.number_of_samples):
            start_samples_index = self.batch_size * counter
            end_samples_index = self.batch_size * (counter + 1)

            if end_samples_index > self.number_of_samples:
                end_samples_index = self.number_of_samples
                start_samples_index = end_samples_index-self.batch_size

            # list of samples's index
            samples_index = sample_index[start_samples_index: end_samples_index]

            # submatrix of W and A, cut for sample index
            net_batch = self.net[samples_index, :]
            net_batch_adj = self.net[samples_index, :][:, samples_index]
            node_label = self.labels[samples_index]
            node_index = samples_index

            # B_net and B_att param of hadmard operation
            B_net = np.ones(net_batch.shape)
            B_net[net_batch != 0] = self.net_hadmard_coeff

            # trasform np array to tensor
            net_batch_tensor = torch.from_numpy(net_batch).float()
            net_batch_adj_tensor = torch.from_numpy(net_batch_adj).float()
            B_net_tensor = torch.from_numpy(B_net).float()

            inputs = [net_batch_tensor.cuda(), net_batch_adj_tensor.cuda()]
            B_params = B_net_tensor.cuda()
            batch_info = [node_index, node_label]

            # feed the fit() function with new data
            yield inputs, B_params, batch_info
            counter += 1


class DataBatchGenerator_MultiAttrOnly():
    def __init__(self,net, multiAtt, labels, batch_size, shuffle, IndexofSamples):
        self.multiAtt = multiAtt
        self.net = net
        self.labels = labels
        self.number_of_samples = len(multiAtt)
        self.batch_size = batch_size
        self.number_of_batches = self.number_of_samples / batch_size
        self.shuffle = shuffle
        self.att_len = multiAtt[0].shape[0]
        self.batch_len = self.att_len * batch_size
        self.batch_width = multiAtt[0].shape[1]
        self.IndexofSamples = IndexofSamples

    def generate(self):
        sample_index = np.arange(self.number_of_samples)
        counter = 0
        if self.shuffle:
            np.random.shuffle(sample_index)

        while (counter*(self.batch_size) <= self.number_of_samples):
            start_samples_index = self.batch_size * counter
            end_samples_index = self.batch_size * (counter + 1)

            if end_samples_index > self.number_of_samples:
                end_samples_index = self.number_of_samples
                start_samples_index = end_samples_index-self.batch_size

            att_batch = np.zeros([self.batch_len,self.batch_width])
            samples_index = []

            for i, idx in enumerate(sample_index[start_samples_index:end_samples_index]):
                att_batch[i*self.att_len:(i+1)*self.att_len, :] = self.multiAtt[idx]
                samples_index.extend(self.IndexofSamples)

            node_index = samples_index
            node_label = self.labels[samples_index]
            net_batch_adj = self.net[samples_index, :][:, samples_index]

            # B_net and B_att param of hadmard operation
            B_att = np.ones(att_batch.shape)

            # trasform np array to tensor
            att_batch_tensor = torch.from_numpy(att_batch).float()
            B_att_tensor = torch.from_numpy(B_att).float()
            net_batch_adj_tensor = torch.from_numpy(net_batch_adj).float()

            inputs = [att_batch_tensor.cuda(), net_batch_adj_tensor.cuda()]
            B_params = B_att_tensor.cuda()
            
            attr_index = sample_index[start_samples_index:end_samples_index]
            batch_info = [node_index, node_label, attr_index]

            # feed the fit() function with new data
            yield inputs, B_params, batch_info
            counter += 1