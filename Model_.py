
import torch
from torch import nn
import pickle as cPickle
from AutoEncoder import AutoEncoder
from Optimizer import OptimizationFunction
from Regularization import RegularizationFunction
from Utils import Util_class
from DataBatchGenerator import DataBatchGenerator
from LossFunction import LossFunction
import numpy as np


class GraphEModel(nn.Module):

    def __init__(self, model_config,):
        """
          model_config : dictionary:
              att_dim : dimension of input and output of attribute/semantical space
              att_layers_list : param layers_list : sequential sort list of semantical network architecture
              att_latent_dim : dimension of embedding/latent semantical space

              net_dim : dimension of input and output of structural/network space
              net_layers_list : param layers_list : sequential sort list of structural network architecture
              net_latent_dim : dimension of embedding/latent structural space

              loss_functions : dictionary key: all,net,att
                  - all : A+N training modality, same loss for both
                  - net : A/N>N/A training modality, set loss for net model
                  - att : A/N>N/A training modality, set loss for att model
                  Each item is a list of vector: [Loss_function function name, param]

              regularization_net : list of dictionary of regularization for structure
                      reg_name : regularization function name,
                      coeff : coeff regularization influence

              regularization_att : list of dictionary of regularization for semantical
                      reg_name : regularization function name,
                      coeff : coeff regularization influence


              model_name : string, name of model

              optimizator_net : dictionary - optimizator config for structure
                      opt_name : optimizator function name,
                      lr_rate :learning rate
                      weight_decay : [OPT - if adam_L2] decay weight param

              optimizator_att : dictionary - optimizator config for semantical
                      opt_name : optimizator function name,
                      lr_rate :learning rate
                      weight_decay : [OPT - if adam_L2] decay weight param

              training_config : string, order to make a training
                  "A>N" : first attribute and then structure
                  "N>A" : first structure and then attribute
                  "A+N" : attribute and structure simultaneously
                  "N+A" : attribute and structure simultaneously
        """
        super(GraphEModel, self).__init__()
        self.epochs_status = dict()

        self.att_dim = model_config["att_dim"]
        self.att_layers_list = model_config["att_layers_list"]
        self.att_latent_dim = model_config["att_latent_dim"]
        self.epochs_status['att'] = 0

        self.net_dim = model_config["net_dim"]
        self.net_layers_list = model_config["net_layers_list"]
        self.net_latent_dim = model_config["net_latent_dim"]
        self.epochs_status['net'] = 0

        self.loss_functions = model_config["loss_functions"]

        self.model_name = model_config["model_name"]

        # Model Auto-encoders Initialization
        self.autoEncoder = dict()
        self.autoEncoder['att'] = AutoEncoder(dim=self.att_dim, layers_list=self.att_layers_list, latent_dim=self.att_latent_dim)
        self.autoEncoder['net'] = AutoEncoder(dim=self.net_dim, layers_list=self.net_layers_list, latent_dim=self.net_latent_dim)

        # Optimization Initialization
        self.optimizatior = dict()
        opt_net_obj = OptimizationFunction(model_config['optimizator_net'])
        opt_att_obj = OptimizationFunction(model_config['optimizator_att'])
        self.optimizatior['net'] = opt_net_obj.get_optimizator(self.autoEncoder['net'])
        self.optimizatior['att'] = opt_att_obj.get_optimizator(self.autoEncoder['att'])

        # Regularization Initialization
        self.regularization = dict()
        regularization_net_obj = RegularizationFunction(model_config['regularization_net'])
        regularization_att_obj = RegularizationFunction(model_config['regularization_att'])
        self.regularization['net'] = regularization_net_obj
        self.regularization['att'] = regularization_att_obj
        self.scheduler = dict()
        self.scheduler['net'] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizatior['net'], mode='min', factor=0.1, patience=5)
        self.scheduler['att'] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizatior['att'], mode='min', factor=0.1, patience=5)
        self.training_config = model_config['training_config']

        self.space_embedded = {'net': dict(), 'att': dict(), 'node_label': dict()}

    def get_Model_semantical(self):
        return self.autoEncoder['att']

    def get_Model_structural(self):
        return self.autoEncoder['net']

    def get_Models(self):
        return {"att": self.get_Model_semantical(), "net": self.get_Model_structural()}

    def save_model(self, epoch, path_file):
        torch.save({
            'NET_model_state_dict': self.autoEncoder['net'].state_dict(),
            'ATT_model_state_dict': self.autoEncoder['att'].state_dict(),

            'NET_optimizer_state_dict': self.optimizatior['net'].state_dict(),
            'ATT_optimizer_state_dict': self.optimizatior['att'].state_dict(),

            'epochs_status': self.epochs_status,
            'space_embedded': self.space_embedded,
          }, path_file)

    def load_model(self, path_file):
        checkpoint = torch.load(path_file)
        self.autoEncoder['net'].load_state_dict(checkpoint['NET_model_state_dict'])
        self.autoEncoder['att'].load_state_dict(checkpoint['ATT_model_state_dict'])

        self.optimizatior['net'].load_state_dict(checkpoint['NET_optimizer_state_dict'])
        self.optimizatior['att'].load_state_dict(checkpoint['ATT_optimizer_state_dict'])

        self.epochs_status = checkpoint['epochs_status']
        self.space_embedded = checkpoint['space_embedded']

    def save_embedding(self):
        raise NotImplementedError('GraphE save_embedding not implemented')

    def model_info(self):
        print("STRUCTURAL Model's state_dict :")
        for param_tensor in self.autoEncoder['net'].state_dict():
            print(param_tensor, "\t", self.autoEncoder['net'].state_dict()[param_tensor].size())

        print("SEMANTICAL Model's state_dict :")
        for param_tensor in self.autoEncoder['att'].state_dict():
            print(param_tensor, "\t", self.autoEncoder['att'].state_dict()[param_tensor].size())

        # Print optimizer's state_dict
        print("STRUCTURAL Optimizer's state_dict:")
        for var_name in self.optimizatior['net'].state_dict():
            print(var_name, "\t", self.optimizatior['net'].state_dict()[var_name])
        print("SEMANTICAL Optimizer's state_dict:")
        for var_name in self.optimizatior['att'].state_dict():
            print(var_name, "\t", self.optimizatior['att'].state_dict()[var_name])

    def models_training(self, datagenerator, epochs, path_embedding="/content/", loss_verbose=False):

        if (self.training_config == "A+N") or (self.training_config == "N+A"):

            if isinstance(epochs, dict):
                res = self.models_training_simultaneously(datagenerator, epochs, path_embedding=path_embedding, loss_verbose=loss_verbose)

            else:
                raise GraphEModel_Exception__TrainingEpochType(epochs, self.training_config, int)

        elif self.training_config == "A>N":
            phases = ["att", "net"]
            if isinstance(epochs, dict):
                epochs_check = Util_class.check_key_in_dict(epochs, phases)
                if epochs_check[0]:
                    res = self.models_training_2phased(phases, datagenerator, epochs, path_embedding= path_embedding, loss_verbose=loss_verbose)
                else:
                    raise GraphEModel_Exception__TrainingEpochItems(epochs, phases, epochs_check[1])
            else:
                raise GraphEModel_Exception__TrainingEpochType(epochs, self.training_config, dict)

        elif self.training_config == "N>A":
            phases = ["net", "att"]
            if isinstance(epochs, dict):
                epochs_check = Util_class.check_key_in_dict(epochs, phases)
                if epochs_check[0]:
                    res = self.models_training_2phased(phases, datagenerator, epochs, path_embedding=path_embedding, loss_verbose=loss_verbose)
                else:
                    raise GraphEModel_Exception__TrainingEpochItems(epochs, phases, epochs_check[1])
            else:
                raise GraphEModel_Exception__TrainingEpochType(epochs, self.training_config, dict)

        elif self.training_config == "multiAttr":
            phases = ["net", "att"]
            if isinstance(epochs, dict):
                epochs_check = Util_class.check_key_in_dict(epochs, phases)
                if epochs_check[0]:
                    res = self.models_training_multiAttr(phases, datagenerator, epochs, path_embedding=path_embedding, loss_verbose=loss_verbose)
                else:
                    raise GraphEModel_Exception__TrainingEpochItems(epochs, phases, epochs_check[1])
            else:
                raise GraphEModel_Exception__TrainingEpochType(epochs, self.training_config, dict)

        else:
            raise GraphEModel_Exception__TrainingModality()
        return res

    def models_training_simultaneously(self, datagenerator, epochs, path_embedding="/content/", loss_verbose=False):
        """
        data : DataBatchGenerator, data
        epochs : int, times re-training process do
        """
        outputs = dict()
        losses = []

        if not isinstance(datagenerator, DataBatchGenerator):
            raise GraphEModel_Exception__notDataBatchGeneratorClass(datagenerator)

        epochs_time = epochs["all"]
        tot_epochs = epochs_time + self.epochs_status['att']
        for epoch in range(1, epochs_time+1):
            loss_epoch = []
            if epoch %2 == 0:
                print("=")
            else:
                print("==")

            node_4batch = list()

            for [input, B_param, batch_info] in datagenerator.generate():
                [net_batch, att_batch, net_batch_adj_tensor] = input
                [B_net, B_att] = B_param
                [node_index, node_labels] = batch_info

                # Output of Auto-encoder
                net_comp = self.autoEncoder['net'].forward(net_batch)
                att_comp = self.autoEncoder['att'].forward(att_batch)

                # Calculate the loss function
                loss_values_matrix = {
                    "net": {
                        "y_true": net_comp["x_input"],
                        "y_late": net_comp["x_latent"],
                        "y_pred": net_comp["x_output"],
                        "B_param": B_net,
                        "y_adj": net_batch_adj_tensor,
                    },
                    "att": {
                        "y_true": att_comp["x_input"],
                        "y_late": att_comp["x_latent"],
                        "y_pred": att_comp["x_output"],
                        "B_param": B_att,
                        "y_adj": None,
                    }
                }

                loss_obj = LossFunction(self.loss_functions['all'], loss_values_matrix)
                loss = loss_obj.loss_computate(loss_verbose)
                if torch.isnan(loss):
                    print(loss_values_matrix)
                    raise NotImplementedError('loss is nan')

                regularization_influence_net = self.regularization['net'].get_regularization(self.autoEncoder['net'])
                regularization_influence_att = self.regularization['att'].get_regularization(self.autoEncoder['att'])
                regularization_loss = regularization_influence_net + regularization_influence_att

                loss += regularization_loss

                self.optimizatior['net'].zero_grad()
                self.optimizatior['att'].zero_grad()

                loss.backward()

                # The gradients are set to zero, and the gradient is computed and stored.
                # .step() performs parameter update
                self.optimizatior['net'].step()
                self.optimizatior['att'].step()

                # Storing the losses in a list for plotting
                loss_epoch.append(loss.item())
                loss_mean_epoch = sum(loss_epoch) / float(len(loss_epoch))

                if epoch == tot_epochs-1:
                    output_dict_net = {
                        "input": net_comp["x_input"],
                        "latent": net_comp["x_latent"],
                        "output": net_comp["x_output"],
                    }
                    output_dict_att = {
                        "input": att_comp["x_input"],
                        "latent": att_comp["x_latent"],
                        "output": att_comp["x_output"],
                    }
                    node_info = {
                        "node_index": node_index,
                        "node_label": node_labels,
                    }

                    output_dict = {
                        "net": output_dict_net,
                        "att": output_dict_att,
                        "node_info": node_info
                    }
                    node_4batch.append(output_dict)

            self.epochs_status['att'] += 1
            self.epochs_status['net'] += 1
            epoch_globaly = self.epochs_status['net']

            self.scheduler['net'].step(loss_mean_epoch)
            self.scheduler['att'].step(loss_mean_epoch)
            losses.append(loss_mean_epoch)
            print("Epoch : ", epoch_globaly, "/", tot_epochs, "\tLoss : ", loss_mean_epoch, "\tlr net: ", self.optimizatior['net'].param_groups[0]['lr'], "\tlr att: ", self.optimizatior['att'].param_groups[0]['lr'])
            outputs[epoch] = node_4batch

        self.set_embedding(encoder_out=outputs, last_epoch= epochs_time - 1, save=True, path=path_embedding, phases=['att', 'net'])
        return {"output": outputs, "losses": losses, "saved_embedding": True}

    def models_training_2phased(self, phases_list, datagenerator, epochs, path_embedding="/content/", loss_verbose=False):
        check_phase = Util_class.same_key_in_dict(phases_list, ['net', 'att'])

        if not check_phase[0]:
            raise GraphEModel_Exception__TrainingPhasesNotSame(check_phase)

        outputs = dict()
        losses = dict()

        if not isinstance(datagenerator, DataBatchGenerator):
            raise GraphEModel_Exception__notDataBatchGeneratorClass(datagenerator)

        for phase in phases_list:
            epochs_time = epochs[phase]
            tot_epochs = epochs_time + self.epochs_status[phase]
            losses[phase] = list()

            if epochs[phase] < 1:
                print(f"No epoch to train for phase: {phase}.")

            for epoch in range(epochs_time):
                loss_epoch = []

                if epoch % 2 == 0:
                    print("=")
                else:
                    print("==")

                node_4batch = list()

                for [input, B_param, batch_info] in datagenerator.generate():
                    [net_batch, att_batch, net_batch_adj_tensor] = input
                    [B_net, B_att] = B_param
                    [node_index, node_labels] = batch_info

                    # Output of Auto-encoder
                    autoencoder_component = dict()
                    if phase == "net":
                        autoencoder_component['net'] = self.autoEncoder['net'].forward(net_batch)
                        loss_values_matrix = {
                            "net": {
                                "y_true": autoencoder_component['net']["x_input"],
                                "y_late": autoencoder_component['net']["x_latent"],
                                "y_pred": autoencoder_component['net']["x_output"],
                                "B_param": B_net,
                                "y_adj": net_batch_adj_tensor,
                            },
                            "att": {
                                "y_true": None,  # autoencoder_component['att']["x_input"],
                                "y_late": None,  # autoencoder_component['att']["x_latent"],
                                "y_pred": None,  # autoencoder_component['att']["x_output"],
                                "B_param": None,  # B_att,
                                "y_adj": None,
                            }
                        }
                    elif phase == "att":
                        autoencoder_component['att'] = self.autoEncoder['att'].forward(att_batch)
                        loss_values_matrix = {
                            "net": {
                                "y_true": None,  # autoencoder_component['net']["x_input"],
                                "y_late": self.get_embedding(nodes_list=node_index, phase='net', type_output='tensor'),   # autoencoder_component['net']["x_latent"],
                                "y_pred": None,  # autoencoder_component['net']["x_output"],
                                "B_param": B_net,
                                "y_adj": net_batch_adj_tensor,
                            },
                            "att": {
                                "y_true": autoencoder_component['att']["x_input"],
                                "y_late": autoencoder_component['att']["x_latent"],
                                "y_pred": autoencoder_component['att']["x_output"],
                                "B_param": B_att,
                                "y_adj": None,
                            }
                        }

                    # Calculate the loss function
                    loss_obj = LossFunction(self.loss_functions[phase], loss_values_matrix)
                    loss = loss_obj.loss_computate(loss_verbose)

                    regularization_influence = self.regularization[phase].get_regularization(self.autoEncoder[phase])
                    regularization_loss = regularization_influence

                    loss += regularization_loss

                    # Reset gradient
                    self.optimizatior[phase].zero_grad()

                    loss.backward(retain_graph=True)
                    # The gradients are set to zero, and the gradient is computed and stored.
                    # .step() performs parameter update
                    self.optimizatior[phase].step()

                    # Storing the losses in a list for plotting
                    loss_epoch.append(loss.item())

                    if epoch == epochs_time-1:
                        output_dict = {
                            "input": autoencoder_component[phase]["x_input"],
                            "latent": autoencoder_component[phase]["x_latent"],
                            "output": autoencoder_component[phase]["x_output"],
                        }
                        node_info = {
                            "node_index": node_index,
                            "node_label": node_labels,
                        }

                        output_dict = {
                            phase: output_dict,
                            "node_info": node_info
                        }
                        node_4batch.append(output_dict)

                loss_mean_epoch = sum(loss_epoch) / float(len(loss_epoch))

                self.epochs_status[phase] += 1
                epoch_globaly = self.epochs_status[phase]

                self.scheduler[phase].step(loss_mean_epoch)
                losses[phase].append(loss_mean_epoch)
                print("Phase : ", phase, "\tEpoch : ", epoch_globaly, "/", tot_epochs, "\tLoss : ", loss_mean_epoch, "\tlr net: ", self.optimizatior['net'].param_groups[0]['lr'], "\tlr att: ", self.optimizatior['att'].param_groups[0]['lr'])
            outputs[phase] = node_4batch
            self.set_embedding(encoder_out=outputs, last_epoch=phase, save=True, path=path_embedding, phases=[phase])
        return {"output": outputs, "losses": losses, "saved_embedding": True}

    def models_training_multiAttr(self, phases_list, datagenerator, epochs, path_embedding="/content/", loss_verbose=False):
        # using multiAttr forces a N>A phase
        # and datagenerator must be a dict contains two seperate datagenerator, 'net' and 'att'.
        outputs = dict()
        losses = dict()

        # network embedding
        phase = phases_list[0]
        epochs_time = epochs[phase]
        tot_epochs = epochs_time + self.epochs_status[phase]
        losses[phase] = list()

        if epochs[phase] < 1:
            print(f"No epoch to train for phase: {phase}.")

        for epoch in range(epochs_time):

            loss_epoch = []

            if epoch % 2 == 0:
                print("=")
            else:
                print("==")

            node_4batch = list()

            for [input, B_param, batch_info] in datagenerator['net'].generate():
                [net_batch, net_batch_adj_tensor] = input
                B_net = B_param
                [node_index, node_labels] = batch_info

                # Output of Auto-encoder
                autoencoder_component = dict()
                autoencoder_component['net'] = self.autoEncoder['net'].forward(net_batch)
                loss_values_matrix = {
                    "net": {
                        "y_true": autoencoder_component['net']["x_input"],
                        "y_late": autoencoder_component['net']["x_latent"],
                        "y_pred": autoencoder_component['net']["x_output"],
                        "B_param": B_net,
                        "y_adj": net_batch_adj_tensor,
                    },
                    "att": {
                        "y_true": None,  # autoencoder_component['att']["x_input"],
                        "y_late": None,  # autoencoder_component['att']["x_latent"],
                        "y_pred": None,  # autoencoder_component['att']["x_output"],
                        "B_param": None,  # B_att,
                        "y_adj": None,
                    }
                }

                loss_obj = LossFunction(self.loss_functions[phase], loss_values_matrix)
                loss = loss_obj.loss_computate(loss_verbose)

                regularization_influence = self.regularization[phase].get_regularization(self.autoEncoder[phase])
                regularization_loss = regularization_influence

                loss += regularization_loss

                # Reset gradient
                self.optimizatior[phase].zero_grad()

                loss.backward(retain_graph=True)
                # The gradients are set to zero, and the gradient is computed and stored.
                # .step() performs parameter update
                self.optimizatior[phase].step()

                # Storing the losses in a list for plotting
                loss_epoch.append(loss.item())

                if epoch == epochs_time-1:
                    output_dict = {
                        "input": autoencoder_component[phase]["x_input"],
                        "latent": autoencoder_component[phase]["x_latent"],
                        "output": autoencoder_component[phase]["x_output"],
                    }
                    node_info = {
                        "node_index": node_index,
                        "node_label": node_labels,
                    }

                    output_dict = {
                        phase: output_dict,
                        "node_info": node_info
                    }
                    node_4batch.append(output_dict)

            loss_mean_epoch = sum(loss_epoch) / float(len(loss_epoch))

            self.epochs_status[phase] += 1
            epoch_globaly = self.epochs_status[phase]

            self.scheduler[phase].step(loss_mean_epoch)
            losses[phase].append(loss_mean_epoch)
            print("Phase : ", phase, "\tEpoch : ", epoch_globaly, "/", tot_epochs, "\tLoss : ", loss_mean_epoch, "\tlr net: ", self.optimizatior['net'].param_groups[0]['lr'], "\tlr att: ", self.optimizatior['att'].param_groups[0]['lr'])
        outputs[phase] = node_4batch
        self.set_embedding(encoder_out=outputs, last_epoch= phase, save=True, path=path_embedding, phases=[phase])

        phase = phases_list[1]
        epochs_time = epochs[phase]
        tot_epochs = epochs_time + self.epochs_status[phase]
        losses[phase] = list()

        if epochs[phase] < 1:
            print(f"No epoch to train for phase: {phase}.")

        allAttrEmbedding = list()
        allAttrIndex = list()
        for epoch in range(epochs_time):
            loss_epoch = []
            if epoch % 2 == 0:
                print("=")
            else:
                print("==")
            node_4batch = list()
            maxIndex = 0
            for [input, B_param, batch_info] in datagenerator['att'].generate():
                [att_batch, net_batch_adj_tensor] = input
                B_att = B_param
                [node_index, node_labels, attr_index] = batch_info

                # Output of Auto-encoder
                autoencoder_component = dict()
                attrLen = int(len(node_labels) / len(attr_index))

                autoencoder_component['att'] = self.autoEncoder['att'].forward(att_batch)
                loss_values_matrix = {
                    "net": {
                        "y_true": None,  # autoencoder_component['net']["x_input"],
                        "y_late": self.get_embedding(nodes_list=node_index, phase='net', type_output='tensor'),  # autoencoder_component['net']["x_latent"],
                        "y_pred": None,  # autoencoder_component['net']["x_output"],
                        "B_param": B_net,
                        "y_adj": net_batch_adj_tensor,
                    },
                    "att": {
                        "y_true": autoencoder_component['att']["x_input"],
                        "y_late": autoencoder_component['att']["x_latent"],
                        "y_pred": autoencoder_component['att']["x_output"],
                        "B_param": B_att,
                        "y_adj": None,
                    }
                }

                loss_obj = LossFunction(self.loss_functions[phase], loss_values_matrix)

                loss = loss_obj.loss_computate(loss_verbose)

                regularization_influence = self.regularization[phase].get_regularization(self.autoEncoder[phase])
                regularization_loss = regularization_influence

                loss += regularization_loss

                self.optimizatior[phase].zero_grad()

                loss.backward(retain_graph=True)
                # The gradients are set to zero, and the gradient is computed and stored.
                # .step() performs parameter update
                self.optimizatior[phase].step()

                # Storing the losses in a list for plotting
                loss_epoch.append(loss.item())

                if epoch == epochs_time-1:
                    output_dict = {
                        "input": autoencoder_component[phase]["x_input"],
                        "latent": autoencoder_component[phase]["x_latent"],
                        "output": autoencoder_component[phase]["x_output"],
                    }
                    node_info = {
                        "node_index": node_index,
                        "node_label": node_labels,
                    }

                    output_dict = {
                        phase: output_dict,
                        "node_info": node_info
                    }
                    node_4batch.append(output_dict)
                    allAttrIndex.append(attr_index)
                    allAttrEmbedding.append(autoencoder_component[phase]["x_latent"])
                    maxIndex_epoch = np.max(attr_index)
                    if maxIndex_epoch > maxIndex:
                        maxIndex = maxIndex_epoch

            loss_mean_epoch = sum(loss_epoch) / float(len(loss_epoch))

            self.epochs_status[phase] += 1
            epoch_globaly = self.epochs_status[phase]

            self.scheduler[phase].step(loss_mean_epoch)
            losses[phase].append(loss_mean_epoch)
            print("Phase : ", phase, "\tEpoch : ", epoch_globaly, "/", tot_epochs, "\tLoss : ", loss_mean_epoch, "\tlr net: ", self.optimizatior['net'].param_groups[0]['lr'], "\tlr att: ", self.optimizatior['att'].param_groups[0]['lr'])

        allAttrEmbedding_out = [None] * (maxIndex+1)
        for ies,indexs in enumerate(allAttrIndex):
            useEmbedding = allAttrEmbedding[ies].detach().cpu().numpy()
            for i, index in enumerate(indexs):
                allAttrEmbedding_out[index] = useEmbedding[i*attrLen:(i+1)*attrLen, :]
           
        outputs[phase] = node_4batch
        self.set_embedding(encoder_out=outputs, last_epoch= phase, save=True, path=path_embedding, phases=[phase])
        return {"output": outputs, "losses": losses, "saved_embedding": True, "allEmbedding": allAttrEmbedding_out}

    def get_embedding(self, nodes_list=None, phase='net', type_output='tensor'):

        if phase not in ['net', 'att', 'node_label']:
            raise GraphEModel_Exception__EmbeddingKeyNotRecoignezed(phase)
        if nodes_list is None or len(nodes_list) == 0:
            nodes_list = []
            for k in self.space_embedded['node_label']:
                nodes_list.append(k)
            if len(nodes_list) > 0:
                return self.get_embedding(nodes_list, phase, type_output)
            else:
                raise GraphEModel_Exception__EmbeddingNodeIdNotFound(-1)
        else:
            embedding_request = None
            for node_id in nodes_list:
                if node_id not in self.space_embedded['node_label']:
                    raise GraphEModel_Exception__EmbeddingNodeIdNotFound(node_id)
                else:
                    if phase == 'node_label':
                        if embedding_request is None:
                            embedding_request = [self.space_embedded[phase][node_id]]
                        else:
                            embedding_request.append(self.space_embedded[phase][node_id])

                    else:
                        if embedding_request is None:
                            embedding_request = self.space_embedded[phase][node_id]
                        else:
                            embedding_request = torch.vstack([embedding_request,self.space_embedded[phase][node_id]])

            if type_output == 'tensor':
                return embedding_request
            elif (type_output == 'np' or type_output == 'numpy') and phase == 'node_label':
                return np.array(embedding_request)
            elif (type_output == 'np' or type_output == 'numpy'):
                return np.array(list(embedding_request.detach().cpu().numpy()))
            else:
                raise GraphEModel_Exception__EmbeddingNodeIdNotFound(node_id)

            return embedding_request

    def set_embedding(self, encoder_out, last_epoch, save=False, path=None, phases=["net", "att"]):
        """
        batches : epoch batches
        epoch : int, epoch to analized embedding

        RETURN set locally embedding space selected and if save=True it is saved in a file
        """

        for phase in phases:
            print("Set embedding for:\t", phase)
            for batch in range(len(encoder_out[last_epoch])):
                for i in range(len(encoder_out[last_epoch][batch]['node_info']['node_index'])):
                    node_key = encoder_out[last_epoch][batch]['node_info']['node_index'][i]
                    self.space_embedded[phase][node_key] = encoder_out[last_epoch][batch][phase]['latent'][i].data.clone()

                    self.space_embedded['node_label'][node_key] = encoder_out[last_epoch][batch]['node_info']['node_label'][i]
            if save:
                if path is None:
                    raise GraphEModel_Exception__notPathEmbeddingParam()
                else:
                    path_embedding_file = "{fpath}embedding_{fmodelname}_{fphase}.ecarbo".format(fpath=path, fmodelname=self.model_name, fphase=phase)
                    with open(path_embedding_file, "wb") as fileEmbedding:
                          cPickle.dump(self.space_embedded, fileEmbedding)
                    print(f"Saved embedding for:\t {phase}\t\t on path:\t{path_embedding_file}")


class GraphEModel_Exception__notDataBatchGeneratorClass(Exception):
    """Exception raised for errors of data input type"""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"{type(self.value)} : type of attribute file format not recognized. It should be a 'DataBatchGenerator' istance."


class GraphEModel_Exception__notPathEmbeddingParam(Exception):
    """Exception raised for errors of path to save embedding is none"""

    def __init__(self):
        self.value = None

    def __str__(self):
        return f"Path where save embedding is None."


class GraphEModel_Exception__EmbeddingKeyNotRecoignezed(Exception):
    """Exception raised for errors of path to save embedding is none"""

    def __init__(self,value):
        self.value = value

    def __str__(self):
        return f"{self.value} is not embedding recognized key. Phase accept are: 'net','att' and 'node_label'."


class GraphEModel_Exception__EmbeddingNodeIdNotFound(Exception):
    """Exception raised for errors of path to save embedding is none"""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Node id '{self.value}' not found."


class GraphEModel_Exception__TrainingModality(Exception):
    """Exception raised for error no training modality recognized"""

    def __init__(self, value):
        self.value = None

    def __str__(self):
        return f"{self.value} is not a modality for training recognized. It should be: 'A+N' or 'N<A' or 'A>N'."


class GraphEModel_Exception__TrainingEpochType(Exception):
    """Exception raised for error no training modality recognized"""

    def __init__(self, value, modality, typeObjRequest):
        self.value = value
        self.modality = modality
        self.typeObjRequest = typeObjRequest

    def __str__(self):
        return f"In modality of training like {self.modality}, epoch value shoud be a {str(self.typeObjRequest)} object but receive an {str(type(self.value))} object."


class GraphEModel_Exception__TrainingEpochItems(Exception):
    """Exception raised for error no training modality recognized"""

    def __init__(self, value, keyRequest, keyMissing):
        self.value = value
        self.keyRequest = keyRequest
        self.keyMissing = (" ").join(keyMissing)

    def __str__(self):
        return f"Epochs array should have {len(self.keyRequest)} items but receive {len(self.value)} items. Key Missin is: {self.keyMissing}"


class GraphEModel_Exception__TrainingPhasesNotSame(Exception):
    """Exception raised for error no training modality recognized"""

    def __init__(self, value, list_check_phases):
        self.value = value
        self.is_same = list_check_phases[0]
        self.key_not_dict = list_check_phases[1]
        self.key_not_list = list_check_phases[2]

    def __str__(self):
        message = "Phase should be same of declaration but:\n"
        if len(self.key_not_dict) > 0:
            _msg = (" ").join(self.key_not_dict)
            message += f"There are input phases key not recognized:\n\t {_msg} \n"

        if len(self.key_not_list) > 0:
            _msg = (" ").join(self.key_not_list)
            message += f"There are  missing phases:\n\t {_msg} \n"

        return message
