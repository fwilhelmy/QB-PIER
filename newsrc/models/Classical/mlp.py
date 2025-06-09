import torch
from typing import List, Tuple
from torch import nn
from models.BaseModel import BaseModel

class Linear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
       
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super(Linear, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
    
    def forward(self, input):
        """
            :param input: [bsz, in_features]
            :return result [bsz, out_features]
        """
        return input @ self.weight.t() + self.bias

class MLP(BaseModel):
    SUPPORTED_ACTIVATION_FN = {'relu': torch.relu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid}

    # 20 points
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, activation: str = "relu", **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_size = input_size
        assert input_size > 0, "The size of the input should be positive"
        self.hidden_sizes = hidden_sizes
        assert len(hidden_sizes) > 1, "You should at least have one hidden layer"
        assert all([hidden_size > 0 for hidden_size in hidden_sizes]), "The number of neurons in each layer should be positive"
        self.num_classes = num_classes
        assert num_classes > 1, "The number of classes should be at least 2"
        self.activation = activation
        assert activation in self.SUPPORTED_ACTIVATION_FN.keys(), "Invalid choice of activation"
        self.hidden_layers, self.output_layer = self._build_layers(input_size, hidden_sizes, num_classes)
        
        # Initializaton
        for layer in self.hidden_layers:
            self._initialize_linear_layer(layer)
        self._initialize_linear_layer(self.output_layer)
    
    def _build_layers(self, input_size: int, 
                        hidden_sizes: List[int], 
                        num_classes: int) -> Tuple[nn.ModuleList, nn.Module]:
        """
        Build the layers for MLP. Be ware of handlling corner cases.
        :param input_size: An int
        :param hidden_sizes: A list of ints. E.g., for [32, 32] means two hidden layers with 32 each.
        :param num_classes: An int
        :Return:
            hidden_layers: nn.ModuleList. Within the list, each item has type nn.Module
            output_layer: nn.Module
        """
        
        hidden_layers = nn.ModuleList(
            [Linear(input_size, hidden_sizes[0])] + 
            [Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
        output_layer = Linear(hidden_sizes[-1], num_classes)
        return hidden_layers, output_layer
    
    def activation_fn(self, activation, inputs: torch.Tensor) -> torch.Tensor:
        """ process the inputs through different non-linearity function according to activation name """
        return self.SUPPORTED_ACTIVATION_FN[activation](inputs)
        
    def _initialize_linear_layer(self, module: nn.Linear) -> None:
        """ For bias set to zeros. For weights set to glorot normal """
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.xavier_normal_(module.weight)
      
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward images and compute logits.
        1. The images are first fattened to vectors. 
        2. Forward the result to each layer in the self.hidden_layer with activation_fn
        3. Finally forward the result to the output_layer.
        
        :param images: [batch, channels, width, height]
        :return logits: [batch, num_classes]
        """
        
        x = x.view(x.size(0), -1)
        for layer in self.hidden_layers:
            x = self.activation_fn(self.activation, layer(x))
        return self.output_layer(x)