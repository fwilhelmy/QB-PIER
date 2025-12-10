import torch
from typing import List, Tuple
from torch import nn

from models import BaseModel, ModelFactory

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(Linear, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
    
    def forward(self, input):
        return input @ self.weight.t() + self.bias

@ModelFactory.model("mlp")
class MLP(BaseModel):
    SUPPORTED_ACTIVATION_FN = {'relu': torch.relu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid}

    # 20 points
    def __init__(self, hidden_sizes: List[int], activation: str = "relu", **kwargs) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters("hidden_sizes", "activation")
        channels, img_size, _ = self.input_size
        assert channels > 0 and img_size > 0, "Invalid input_size provided"
        flattened = channels * img_size * img_size
        self.flat_input_size = flattened
        self.hidden_sizes = hidden_sizes
        assert len(hidden_sizes) > 1, "You should at least have one hidden layer"
        assert all([hidden_size > 0 for hidden_size in hidden_sizes]), "The number of neurons in each layer should be positive"
        assert self.output_size > 1, "The number of classes should be at least 2"
        self.activation = activation
        assert activation in self.SUPPORTED_ACTIVATION_FN.keys(), "Invalid choice of activation"
        self.hidden_layers, self.output_layer = self._build_layers(flattened, hidden_sizes, self.output_size)
        
        # Initializaton
        for layer in self.hidden_layers:
            self._initialize_linear_layer(layer)
        self._initialize_linear_layer(self.output_layer)
    
    def _build_layers(self, input_size: int,
                        hidden_sizes: List[int],
                        output_size: int) -> Tuple[nn.ModuleList, nn.Module]:
        hidden_layers = nn.ModuleList(
            [Linear(input_size, hidden_sizes[0])] + 
            [Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
        output_layer = Linear(hidden_sizes[-1], output_size)
        return hidden_layers, output_layer
    
    def activation_fn(self, activation, inputs: torch.Tensor) -> torch.Tensor:
        """ process the inputs through different non-linearity function according to activation name """
        return self.SUPPORTED_ACTIVATION_FN[activation](inputs)
        
    def _initialize_linear_layer(self, module: nn.Linear) -> None:
        """ For bias set to zeros. For weights set to glorot normal """
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.xavier_normal_(module.weight)
      
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        for layer in self.hidden_layers:
            logits = self.activation_fn(self.activation, layer(logits))
        return self.output_layer(logits)