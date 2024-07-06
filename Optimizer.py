
import torch


class OptimizationFunction():

    def __init__(self, opt_config):
        """
          opt_config : list of dictionary:
              opt_name : optimize function name,
              lr_rate :learning rate
              weight_decay : [OPT - if adam_L2] decay weight param
        """
        self.name_opt = opt_config["opt_name"]
        self.lr_rate = opt_config["lr_rate"]
        if self.name_opt not in ["adam", "adam_L2"]:
            raise OptimizationFunction_Exception_OptimizatorNotExist(self.name_opt)

        if self.name_opt == "adam_L2":
            if "weight_decay" not in opt_config:
                raise OptimizationFunction_Exception_OptimizatorParamsMissing(self.name_opt, "weight_decay")
            else:
                self.weight_decay = opt_config["weight_decay"]

    def get_optimizator(self, net_model):
        self.net_params = net_model.parameters()
        if self.name_opt == "adam":
            return torch.optim.Adam(params=self.net_params, lr=self.lr_rate)
        elif self.name_opt == "adam_L2":
            return torch.optim.Adam(params=self.net_params, lr=self.lr_rate, weight_decay=self.weight_decay)
        else:
            raise OptimizationFunction_Exception_OptimizatorNotExist(self.name_opt)


class OptimizationFunction_Exception_OptimizatorParamsMissing(Exception):
    """Exception raised for error if a param is missing"""

    def __init__(self, name_opt, name_param_missing):
        self.name_opt = name_opt
        self.name_param_missing = name_param_missing

    def __str__(self):
        return f'Optimizator "{self.name_opt}" needs "{self.name_param_missing}" param.'


class OptimizationFunction_Exception_OptimizatorNotExist(Exception):
    """Exception raised for error if optimizator not exist"""

    def __init__(self, opt_name):
        self.opt_name = opt_name

    def __str__(self):
        return f'Optimizator "{self.opt_name}" not exist.'
