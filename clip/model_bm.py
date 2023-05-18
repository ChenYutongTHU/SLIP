from collections import OrderedDict
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
# import bmtrain as bmt
from typing import Optional

class Linear(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = bmt.DistributedParameter(torch.empty(out_features, in_features, dtype=dtype, device="cuda"), init_method=torch.nn.init.xavier_normal_)
        if bias:
            self.bias = bmt.DistributedParameter(torch.empty(out_features, dtype=dtype, device="cuda"), init_method=torch.nn.init.zeros_)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Layernorm_bm(bmt.DistributedModule):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True,
                dtype=None) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = bmt.DistributedParameter(torch.empty(self.normalized_shape, dtype=dtype, device="cuda"), init_method=torch.nn.init.ones_)
            self.bias = bmt.DistributedParameter(torch.empty(self.normalized_shape, dtype=dtype, device="cuda"), init_method=torch.nn.init.zeros_)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

class LayerNorm(Layernorm_bm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(bmt.DistributedModule):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Attention_bm(bmt.DistributedModule):
    def __init__(self, 
            dim_model : int, dim_head : int,
            num_heads : int, add_bias_kv :bool = False, bias : bool = True,
            dtype = None
        ) -> None:
        super().__init__()

        self.project_q = Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
        self.project_k = Linear(dim_model, dim_head * num_heads, bias=add_bias_kv, dtype=dtype)
        self.project_v = Linear(dim_model, dim_head * num_heads, bias=add_bias_kv, dtype=dtype)

        self.project_out = Linear(dim_head * num_heads, dim_model, bias=bias, dtype=dtype)

        self.softmax = torch.nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_model = dim_model
    
    def forward(self, 
            hidden_q : torch.Tensor,        # (batch_size, seq_q, dim_model)
            hidden_kv : torch.Tensor,       # (batch_size, seq_kv, dim_model)
            mask : torch.BoolTensor,        # (batch_size, seq_q, seq_kv)
            position_bias : Optional[torch.Tensor] = None,   # (batch, num_heads, seq_q, seq_kv)
        ) -> torch.Tensor:
        batch_size, seq_q, dim_model = hidden_q.size()
        seq_kv = hidden_kv.size(1)

        h_q : torch.Tensor = self.project_q(hidden_q)
        h_k : torch.Tensor = self.project_k(hidden_kv)
        h_v : torch.Tensor = self.project_v(hidden_kv)

        h_q = h_q.view(batch_size, seq_q, self.num_heads, self.dim_head)
        h_k = h_k.view(batch_size, seq_kv, self.num_heads, self.dim_head)
        h_v = h_v.view(batch_size, seq_kv, self.num_heads, self.dim_head)

        h_q = h_q.permute(0, 2, 1, 3).contiguous()
        h_k = h_k.permute(0, 2, 1, 3).contiguous()
        h_v = h_v.permute(0, 2, 1, 3).contiguous()

        h_q = h_q.view(batch_size * self.num_heads, seq_q, self.dim_head)
        h_k = h_k.view(batch_size * self.num_heads, seq_kv, self.dim_head)
        h_v = h_v.view(batch_size * self.num_heads, seq_kv, self.dim_head)

        score = torch.bmm(
            h_q, h_k.transpose(1, 2)
        )
        score = score / math.sqrt(self.dim_head)

        score = score.view(batch_size, self.num_heads, seq_q, seq_kv)

        if position_bias is not None:
            score = score + position_bias.view(batch_size, self.num_heads, seq_q, seq_kv)
        
        score = torch.where(
            mask.view(batch_size, 1, seq_q, seq_kv),
            score,
            torch.scalar_tensor(float('-inf'), device=score.device, dtype=score.dtype)
        )

        score = torch.where(
            mask.view(batch_size, 1, seq_q, seq_kv),
            self.softmax(score),
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype)
        )

        score = score.view(batch_size * self.num_heads, seq_q, seq_kv)

        h_out = torch.bmm(
            score, h_v
        )
        h_out = h_out.view(batch_size, self.num_heads, seq_q, self.dim_head)
        h_out = h_out.permute(0, 2, 1, 3).contiguous()
        h_out = h_out.view(batch_size, seq_q, self.num_heads * self.dim_head)

        attn_out = self.project_out(h_out)
        return attn_out

class ResidualAttentionBlock(bmt.DistributedModule):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        #self.attn = nn.MultiheadAttention(d_model, n_head)
        self.attn = Attention_bm(dim_model=d_model, dim_head=d_model//n_head, num_heads=n_head)
        self.n_head = n_head
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, attn_mask=None):
        # self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

        if attn_mask==None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
            return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)
        else:
            B, L = attn_mask.shape
            attn_mask = attn_mask.to(dtype=x.dtype, device=x.device) if attn_mask is not None else None
            attn_mask = torch.tile(attn_mask.unsqueeze(1), [self.n_head,L,1]) #b,L -> b*HEAD, L, L
            if self.attn_mask!=None:
                self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None #l,l
                attn_mask = self.attn_mask.unsqueeze(0) + attn_mask #(1,L,L)+(b*head,L,l)           
            y = self.attn(x, x, x, need_weights=True, attn_mask=attn_mask)
            return y
    
    def forward(self, x: torch.Tensor, attn_mask=None):
        # x = x + self.attention(self.ln_1(x))
        # x = x + self.mlp(self.ln_2(x))
        # return x
        attn_output, attn_weights = self.attention(self.ln_1(x), attn_mask=attn_mask) #attn_weight B, H, L, S
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights


class Transformer(bmt.DistributedModule):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.heads = heads
        self.resblocks = bmt.TransformerBlockList(
            [bmt.CheckpointBlock(ResidualAttentionBlock(width, heads, attn_mask)) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask=None):
        #return self.resblocks(x)
        layers_attn_weights = []
        xs = []
        for i in range(self.layers):
            x, attn_weights = self.resblocks[i](x, attn_mask)
            layers_attn_weights.append(attn_weights)
            xs.append(x)
        return xs, layers_attn_weights

class VisionTransformer(bmt.DistributedModule):
    def __init__(self, 
        input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
        return_full_embed=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = bmt.DistributedParameter(scale * torch.randn(width))
        self.positional_embedding = bmt.DistributedParameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = bmt.DistributedParameter(scale * torch.randn(width, output_dim))
        self.return_full_embed = return_full_embed

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attention_weights = self.transformer(x)
        x = x[-1].permute(1, 0, 2)  # LND -> NLD (last layer)

        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj

        return x

class X_Attn_Encoder(bmt.DistributedModule):
    def __init__(self,
        layers, width, heads, embed_dim,
        context_length, causal_attn, order):
        super().__init__()
        self.context_length, self.order = context_length, order
        self.transformer = Transformer(
                width=width, layers=layers, heads=heads,
                attn_mask=self.build_attention_mask() if causal_attn else None) #still causal mask (L,L)
        self.ln_final = LayerNorm(width)
        self.text_projection = bmt.DistributedParameter(
                torch.empty(width, embed_dim))
    
    @property
    def dtype(self):
        return self.text_projection.dtype
    
    def initialize_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def forward(self, zh, en, zh_eot, en_eot):
        def build_attn_mask(txt, eot):
            L, B, D = txt.shape
            loc = torch.tile(torch.arange(L)[None,:],[B,1]).to(eot.device) #B,1
            attn_mask = torch.where(loc<=eot[:,None], 0, float('-inf')) #!!<= include eot
            return attn_mask
        zh_attn_mask = build_attn_mask(zh, zh_eot)
        en_attn_mask = build_attn_mask(en, en_eot)
        if self.order == 'zh_en': #(no need to permute)
            bi_text_features_imme = torch.cat([zh, en], dim=0) #L1+L2,N,D
            bi_text_attn_mask = torch.cat([zh_attn_mask, en_attn_mask], dim=-1) #B,L1+L2
            bi_eot = zh.shape[0]+en_eot #B,
        else:
            bi_text_features_imme = torch.cat([en, zh], dim=0) #L1+L2,N,D
            bi_text_attn_mask = torch.cat([en_attn_mask, zh_attn_mask], dim=-1) #B,L1+L2 
            bi_eot = en.shape[0]+zh_eot #B,
        x, _ = self.transformer(bi_text_features_imme, bi_text_attn_mask)
        x = x[-1].permute(1,0,2) #(permute back N,L,D)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), bi_eot]@self.text_projection
        return x
    
    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
class CLIP_bm(bmt.DistributedModule):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 return_full_embed=False
                 ):
        super().__init__()

        self.context_length = context_length
        self.embed_dim = embed_dim
        self.return_full_embed = return_full_embed
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.transformer_width = transformer_width
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = bmt.DistributedParameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = bmt.DistributedParameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = bmt.DistributedParameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if False and isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            continue
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        #return self.visual.conv1.weight.dtype
        return self.text_projection.dtype


    def encode_image(self, image):
        image_embed = self.visual(image.type(self.dtype))
        if self.return_full_embed==False:
            return image_embed[:,0,:]
        else:
            return image_embed

    def encode_text(self, text, n_layer_1=-1):
        print(text.shape, text[0])
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        xs, _ = self.transformer(x)
        x = xs[-1]
        x_1 = xs[n_layer_1]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        eot_loc = torch.sum(text!=0, dim=-1)-1 #B
        #print('build attn_mask', attn_mask.shape, attn_mask)
        if self.return_full_embed==False:
            x = x[torch.arange(x.shape[0]), eot_loc] @ self.text_projection
        else:
            x = x@self.text_projection
        return x, x_1, eot_loc

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: bmt.DistributedModule):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, return_full_embed = False):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        return_full_embed=return_full_embed
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
