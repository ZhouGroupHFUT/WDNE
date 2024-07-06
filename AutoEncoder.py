
import torch
from torch import nn
from CompetitiveKateLayer import KCompetitiveLayer


class AutoEncoder(nn.Module):
    """
    param dim: original dimension
    param layers_list : sequential sort list of dict : Each item  have a value for "type", as:
                                          DENSE -  hidden layers, with: "features" is dimention of features in output, "act_funtion" is the relative activation funcion,'bias' boolean
                                          DROP  -  dropout, with: "prob" is the percentaul of neuro drop
                                          KCOMP -  kcompetitive layer, with "ktop":int #of active neurons at end of computation, "alpha_factor":float coefficent
    param latent_dim : dimension of latent space
    last_isSigm : boolean, True if last activation function of decoder is a sigmoid
    return : auto-encoder model
    """

    def __init__(self, dim, layers_list, latent_dim, last_isSigm= True):

        super().__init__()
        self.activation = {}
        self.encoder_list=[]
        self.decoder_list=[]

        last_dim = dim
        for i,layer in enumerate(layers_list):
            if layer['type'] == "DROP":
                prob = layer['prob']
                if isinstance(prob, float) and 0 <= prob <= 1:
                    self.encoder_list.append(torch.nn.Dropout(p=prob))
                    self.decoder_list.insert(0,torch.nn.Dropout(p=prob))
                else:
                    raise AutoEncoder_Exception_DropoutProb(prob)

            elif layer['type'] == "DENSE":
                self.encoder_list.append(torch.nn.Linear(in_features=last_dim, out_features=layer['features'], bias=layer['bias']))
                if layer['act_funtion'] == "RELU":
                    self.encoder_list.append(torch.nn.ReLU())
                    decoder_layer_funact = torch.nn.ReLU()
                elif layer['act_funtion'] == "SIGM":
                    self.encoder_list.append(torch.nn.Sigmoid())
                    decoder_layer_funact = torch.nn.Sigmoid()
                else:
                    raise AutoEncoder_Exception_ActivationFunction(layer['act_funtion'])

                if i == 0 and last_isSigm:
                  decoder_layer_funact = torch.nn.Sigmoid()
                self.decoder_list.insert(0, decoder_layer_funact)
                self.decoder_list.insert(0, torch.nn.Linear(in_features=layer['features'], out_features=last_dim, bias=layer['bias']))
                last_dim = layer['features']
            elif layer['type'] == "KCOMP":
                competitiveLayers = KCompetitiveLayer(layer['ktop'], layer['alpha_factor'])
                self.encoder_list.append(competitiveLayers)
            else:
                raise AutoEncoder_Exception_Type(layer['type'])

        if last_dim != latent_dim:
            raise AutoEncoder_Exception_LatentSpace(last_dim, latent_dim)
        self.encoder = nn.Sequential(*self.encoder_list).cuda()
        self.decoder = nn.Sequential(*self.decoder_list).cuda()

    def forward(self,x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent)
        return {"x_input": x, "x_latent": x_latent, "x_output": x_hat}


class AutoEncoder_Exception_Type(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f'{self.value}: type layer not recognized: it should be a hidden layer linear (DENSE) or dropout layer (DROP).'


class AutoEncoder_Exception_DropoutProb(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        if isinstance(self.value, float):
            return f'Dropout should have probability param in range 0 to 1, but receive {self.value}.'
        else:
            return f'Dropout should be a float but receive a {type(self.value)}.'


class AutoEncoder_Exception_ActivationFunction(Exception):

        def __init__(self, value):
            self.value = value

        def __str__(self):
            return f'{self.value}: activation function not recognized: it should be a relu function (RELU), a sigmoid funcion (SIGM).'


class AutoEncoder_Exception_LatentSpace(Exception):

    def __init__(self, last_dim, latent_dim):
        self.last_dim = last_dim
        self.latent_dim = latent_dim

    def __str__(self):
        return f'Last layer have {self.last_dim} output dimention but latent space should be {self.latent_dim}.'
