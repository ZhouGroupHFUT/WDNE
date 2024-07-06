
class RegularizationFunction():

    def __init__(self, reg_config):
        """
          reg_config : list of dictionary:
              reg_name : regularization name,
              coeff : regularization coefficent

        """
        self.regularizations = reg_config

    def get_regularization(self, net_model):

        net_params = net_model.parameters()
        loss_reg = 0
        for _reg in self.regularizations:
            reg_lambda = _reg["coeff"]
            reg_name = _reg["reg_name"]
            if reg_name == "L1":
                reg_norm1 = sum(param.abs(2.0).sum() for param in net_params)
                loss_reg += reg_lambda * reg_norm1
            elif reg_name == "L2":
                reg_norm2 = sum(param.pow(2.0).sum() for param in net_params)
                loss_reg += reg_lambda * reg_norm2
            else:
                raise RegularizationFunction_Exception_RegularizationNotExist(reg_name)
        return loss_reg


class RegularizationFunction_Exception_RegularizationNotExist(Exception):
    """Exception raised for error if optimizer not exist"""

    def __init__(self, opt_name):
        self.opt_name = opt_name

    def __str__(self):
        return f'Regularization "{self.opt_name}" not exist.'
