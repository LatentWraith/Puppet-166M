import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from mamba_ssm.utils.generation import InferenceParams


class Params():
    def __init__(self):
        self.SEED = 13
        
        #Architecture
        self.n_layers = 1
        self.model_dim = 512
        self.d_state = 256
        self.mamba_expand = 2
        self.d_conv = 4
        self.matrix_div = 4
        
        self.length = 1024
        
        #Tokeniser
        self.tok_vocab_n = 32000
        self.tok_min_freq = 100


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.l0 = nn.Linear(dim, dim*2)
        self.l1 = nn.Linear(dim, dim*2)
        
        self.l2 = nn.Linear(dim*2, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.l0(x)) * self.l1(x)
        
        return self.l2(x)
    
class WeightGen(nn.Module):
    def __init__(self, params: Params, shuffle: bool):
        super().__init__()
        self.div = params.matrix_div
        self.chunk = params.model_dim//params.matrix_div
        
        self.mlp = nn.Sequential(
            SwiGLU(params.model_dim),
            nn.RMSNorm(params.model_dim),
        )
        
        self.gen = nn.Linear(params.model_dim, (self.chunk**2) * self.div)
        
        self.bias_head = nn.Linear(params.model_dim, params.model_dim)
        
        self.shuffle = nn.Linear(params.model_dim, params.model_dim, bias=False) if shuffle else None
        if self.shuffle is not None:
            nn.init.eye_(self.shuffle.weight)
        
        nn.init.zeros_(self.gen.bias)
        nn.init.zeros_(self.bias_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D  = x.shape
        
        gx = self.mlp(x)
        dense = self.gen(gx).view((B, S, self.div, self.chunk, self.chunk))

        x = x.view((B, S, self.div, self.chunk)).unsqueeze(-2)
        x = (x @ dense).squeeze(-2)
        x = x.view((B, S, D)) + self.bias_head(gx)
        
        if self.shuffle is not None:
            x = self.shuffle(x)
            
        return x
    
class Bundle(nn.Module):
    def __init__(self, params: Params, layer: int, shuffle: bool):
        super().__init__()
        
        self.norm = nn.RMSNorm(params.model_dim)
        self.mamba = Mamba2(
            d_model=params.model_dim,
            d_state=params.d_state,
            expand=params.mamba_expand,
            d_conv=params.d_conv,
            layer_idx=layer,
        )
        
        self.alpha = nn.Parameter(torch.ones((params.model_dim)))
        self.gen_norm = nn.RMSNorm(params.model_dim)
        self.weight_gen = WeightGen(params, shuffle)
        
    def forward(self, x: torch.Tensor, inf_params: InferenceParams | None = None) -> torch.Tensor:
        x = x + self.mamba(self.norm(x), inference_params = inf_params)
        x = x * self.alpha + self.weight_gen(self.gen_norm(x))
        
        return x

class Block(nn.Module):
    def __init__(self, params: Params, idx: int):
        super().__init__()
        self.params = params
        
        self.steps = nn.ModuleList(
            [Bundle(params, b + params.matrix_div * idx, b!=params.matrix_div-1) for b in range(params.matrix_div)]
        )
        
    def forward(self, x: torch.Tensor, inf_params: InferenceParams | None = None) -> torch.Tensor:
        for b in self.steps:
            x = b(x, inf_params)
        
        return x

class NeuralNetwork(nn.Module):
    def __init__(self, params: Params):
        super().__init__()
        self.params = params
        
        self.embedding = nn.Embedding(params.tok_vocab_n, params.model_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        self.emb_norm = nn.RMSNorm(params.model_dim)
        
        self.layers = nn.ModuleList([
            Block(params, b) for b in range(params.n_layers)
        ])
        
        self.final_norm = nn.RMSNorm(params.model_dim)
    
    def forward(self, x: torch.Tensor, inf_params: InferenceParams | None = None) -> torch.Tensor:
        x = self.emb_norm(self.embedding(x))
        
        for layer in self.layers:
            x = layer(x, inf_params=inf_params)
        
        x = self.final_norm(x)
        
        out = F.linear(x, self.embedding.weight)
        return out