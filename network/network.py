import torch
import math
from torch import nn
from einops import rearrange
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, dim)
        self.act = nn.SiLU()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
 
    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.proj(x)
        return x

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    
    def __init__(self, dim, dim_out, blocks, groups=8):
        super().__init__()
        self.blocks = blocks

        self.block1 = Block(dim, dim_out, groups=groups)
        if blocks:
            self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        if self.blocks:
            h = self.block2(h)
        return h + self.res_conv(x)

class UpSample(nn.Module):
    """
    ## Up-sampling layer
    """
    def __init__(self, channels):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution mapping
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Up-sample by a factor of $2$
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        # Apply convolution
        return self.conv(x)
    
class DownSample(nn.Module):
    """
    ## Down-sampling layer
    """
    def __init__(self, channels):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution with stride length of $2$ to down-sample by a factor of $2$
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Add padding
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        # Apply convolution
        return self.conv(x)

class ImprovedModalAwareEncoder(nn.Module):
    def __init__(self, init_dim=64, num_inputs=4, dim_mults=(1, 2, 4, 8, 10), blocks=True):
        super().__init__()
        self.conv_initial = nn.Conv2d(1, init_dim, 3, padding=1)  # Process one modality at a time
        self.modality_embeddings = nn.Parameter(torch.randn(num_inputs, init_dim, 1, 1))
        self.num_inputs = num_inputs
        
        # Encoder structure
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        # Fusion convolution for each layer
        self.fusion_convs = nn.ModuleList([])
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList([
                    ResnetBlock(dim_in, dim_out, blocks),
                    DownSample(dim_out) if not is_last else nn.Identity(),
                ])
            )
            # Create cross-modal attention fusion module for each layer after the first
            if ind > 0:  # All layers after first
                self.fusion_convs.append(CrossModalityAttention(dim_in))
        
        # Final fusion
        self.final_fusion_conv = nn.Conv2d(dims[-1] * num_inputs, dims[-1], 1)
    
    def forward(self, x):  # x: [B, 4, H, W]
        batch_size = x.shape[0]
        
        # Process intermediate features for each modality separately
        h_per_modality = [[] for _ in range(self.num_inputs)]
        final_features = []
        
        # First layer processing - handle each modality independently
        for i in range(self.num_inputs):
            xi = x[:, i:i+1]  # [B, 1, H, W]
            hi = self.conv_initial(xi)  # [B, dim, H, W]
            
            # Add modality-specific embedding
            hi = hi + self.modality_embeddings[i].expand(batch_size, -1, hi.shape[2], hi.shape[3])
            
            # Save first layer features
            h_per_modality[i].append(hi)
            
            # Continue processing remaining layers
            for j, down in enumerate(self.downs):
                block, downsample = down
                hi = block(hi)
                
                # Save features from each layer
                if j < len(self.downs) - 1:
                    h_per_modality[i].append(hi)
                else:
                    final_features.append(hi)  # Save final layer features separately
                
                hi = downsample(hi)
        
        # Use cross-attention mechanism to fuse features from each layer
        h_fused = []
        for layer_idx in range(len(h_per_modality[0])):
            # Collect features from all modalities at current layer
            layer_features = [h_per_modality[i][layer_idx] for i in range(self.num_inputs)]
            
            if layer_idx == 0:
                # Simple average fusion for first layer
                h_fused.append(torch.mean(torch.stack(layer_features, dim=0), dim=0))
            else:
                # Use cross-attention fusion for subsequent layers
                fused_features = self.fusion_convs[layer_idx-1](layer_features)
                # Take average of fused features as output for this layer
                h_fused.append(torch.mean(torch.stack(fused_features, dim=0), dim=0))
        
        # Fuse final features
        final_feature = self.final_fusion_conv(torch.cat(final_features, dim=1))
        
        return final_feature, h_fused  # Same output format as original encoder 
    
class ModalAwareEncoder(nn.Module):
    def __init__(self, init_dim=64, num_inputs=4, dim_mults=(1, 2, 4, 8, 10), blocks=True):
        super().__init__()
        self.conv_initial = nn.Conv2d(1, init_dim, 3, padding=1)  # Process one modality at a time
        self.modality_embeddings = nn.Parameter(torch.randn(num_inputs, init_dim, 1, 1))
        self.num_inputs = num_inputs
        
        # Other encoder structure remains the same as original
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        # Final fusion convolution to merge all modality features into one
        self.fusion_conv = nn.Conv2d(dim_mults[-1] * init_dim * num_inputs, 
                                    dim_mults[-1] * init_dim, 1)
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList([
                    ResnetBlock(dim_in, dim_out, blocks),
                    DownSample(dim_out) if not is_last else nn.Identity(),
                ])
            )
        
        # Add attention weight calculation
        self.modal_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_mults[-1] * init_dim, num_inputs, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):  # x: [B, 4, H, W]
        batch_size = x.shape[0]
        features = []
        all_h_lists = []
        
        # Process each modality separately
        for i in range(self.num_inputs):
            xi = x[:, i:i+1]  # [B, 1, H, W]
            hi = self.conv_initial(xi)  # [B, dim, H, W]
            
            # Add modality-specific embedding
            hi = hi + self.modality_embeddings[i].expand(batch_size, -1, hi.shape[2], hi.shape[3])
            
            h_list = []
            for down in self.downs:
                block, downsample = down
                hi = block(hi)
                h_list.append(hi)
                hi = downsample(hi)
            
            features.append(hi)
            all_h_lists.append(h_list)
        
        # Calculate attention weights for each modality
        concat_features = torch.cat(features, dim=1)
        x = self.fusion_conv(concat_features)
        
        # Fuse feature maps at different resolutions
        h = []
        for level in range(len(all_h_lists[0])):
            level_features = [h_list[level] for h_list in all_h_lists]
            # Fuse features at the same level
            level_concat = torch.cat(level_features, dim=1)
            level_fused = nn.Conv2d(level_concat.shape[1], all_h_lists[0][level].shape[1], 1).to(x.device)(level_concat)
            h.append(level_fused)
            
        return x, h  # x:(batch_size, 640, 240/2^4, 240/2^4), h: list of feature maps at each resolution

class ModalAwareEncoder(nn.Module):
    def __init__(self, init_dim=64, num_inputs=4, dim_mults=(1, 2, 4, 8, 10), blocks=True):
        super().__init__()
        self.conv_initial = nn.Conv2d(1, init_dim, 3, padding=1)  # Process one modality at a time
        self.modality_embeddings = nn.Parameter(torch.randn(num_inputs, init_dim, 1, 1))
        self.num_inputs = num_inputs
        
        # Other encoder structure remains the same as original
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        # Final fusion convolution to merge all modality features into one
        self.fusion_conv = nn.Conv2d(dim_mults[-1] * init_dim * num_inputs, 
                                    dim_mults[-1] * init_dim, 1)
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList([
                    ResnetBlock(dim_in, dim_out, blocks),
                    DownSample(dim_out) if not is_last else nn.Identity(),
                ])
            )
        
        self.modal_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_mults[-1] * init_dim, num_inputs, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):  # x: [B, 4, H, W]
        batch_size = x.shape[0]
        features = []
        all_h_lists = []
        
        # Add attention weight calculation
        for i in range(self.num_inputs):
            xi = x[:, i:i+1]  # [B, 1, H, W]
            hi = self.conv_initial(xi)  # [B, dim, H, W]
            
            # Add modality-specific embedding
            hi = hi + self.modality_embeddings[i].expand(batch_size, -1, hi.shape[2], hi.shape[3])
            
            h_list = []
            for down in self.downs:
                block, downsample = down
                hi = block(hi)
                h_list.append(hi)
                hi = downsample(hi)
            
            features.append(hi)
            all_h_lists.append(h_list)
        
        # Calculate attention weights for each modality
        concat_features = torch.cat(features, dim=1)
        x = self.fusion_conv(concat_features)
        
        # Fuse feature maps at different resolutions
        h = []
        for level in range(len(all_h_lists[0])):
            level_features = [h_list[level] for h_list in all_h_lists]
            # Fuse features at the same level
            level_concat = torch.cat(level_features, dim=1)
            level_fused = nn.Conv2d(level_concat.shape[1], all_h_lists[0][level].shape[1], 1).to(x.device)(level_concat)
            h.append(level_fused)
            
        return x, h  # x:(batch_size, 640, 240/2^4, 240/2^4), h: list of feature maps at each resolution

class CrossModalityAttention(nn.Module):
    def __init__(self, dim, dim_mults=(1, 2, 4, 8, 10), blocks=True):
        super().__init__()
        self.query_proj = nn.Conv2d(dim, dim, 1)
        self.key_proj = nn.Conv2d(dim, dim, 1)
        self.value_proj = nn.Conv2d(dim, dim, 1)
        self.scale = dim ** -0.5
        self.output_proj = nn.Conv2d(dim, dim, 1)
        self.extract_model = self.extract_model
        # Add gating mechanism
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        
        # Add gating mechanism
        self.pos_embedding = nn.Parameter(torch.randn(1, dim, 1, 1))
    
    def forward(self, modality_features):
        # modality_features: list of tensors [B, C, H, W]
        
        batch_size = modality_features[0].shape[0]
        num_modalities = len(modality_features)
        c, h, w = modality_features[0].shape[1:]
        
        # Fix: Correctly expand position encoding
        pos_embed_expanded = self.pos_embedding.expand(batch_size, -1, h, w)
        
        # Add position encoding
        modality_features = [feat + pos_embed_expanded for feat in modality_features]
        
        # Calculate query, key and value
        queries = [self.query_proj(feat) for feat in modality_features]
        keys = [self.key_proj(feat) for feat in modality_features]
        values = [self.value_proj(feat) for feat in modality_features]
        
        # Perform cross-modal attention calculation
        output_features = []
        
        for i in range(num_modalities):
            q = queries[i].flatten(2).permute(0, 2, 1)  # [B, HW, C]
            
            # Collect context from all other modalities
            context = 0
            attention_weights = []
            
            for j in range(num_modalities):
                k = keys[j].flatten(2).permute(0, 2, 1).transpose(1, 2)  # [B, C, HW]
                v = values[j].flatten(2).permute(0, 2, 1)  # [B, HW, C]
                
                # Calculate attention weights
                attn = torch.bmm(q, k) * self.scale  # [B, HW, HW]
                attn = F.softmax(attn, dim=-1)
                attention_weights.append(attn)
                
                # Weighted values
                attended = torch.bmm(attn, v)  # [B, HW, C]
                context += attended.permute(0, 2, 1).view(batch_size, c, h, w)
            
            # Apply gating mechanism
            gate_value = self.gate(modality_features[i])
            
            # Fuse original features with context information
            fused = gate_value * context + (1 - gate_value) * modality_features[i]
            output = self.output_proj(fused)
            
            output_features.append(output)
        
        return output_features

class ModalityWeighting(nn.Module):
    def __init__(self, num_modalities):
        super().__init__()
        self.modality_weights = nn.Parameter(torch.ones(num_modalities))
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, modality_features):
        weights = self.softmax(self.modality_weights)
        combined = sum(f * w for f, w in zip(modality_features, weights))
        return combined

class Encoder(nn.Module):
    def __init__(self, init_dim=64,H=240,W=240, num_inputs=4, dim_mults=(1, 2, 4, 8, 10), blocks=True):
        super(Encoder, self).__init__()
        self.conv_initial = nn.Conv2d(num_inputs, init_dim, 3, padding=1) # 4*240*240 -> 64*240*240
        self.pos_feature = nn.Parameter(torch.empty(init_dim, H, W))
        nn.init.xavier_normal_(self.pos_feature)
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)] # 64, 64, 128, 256, 512, 640
        in_out = list(zip(dims[:-1], dims[1:])) # [(64, 64), (64, 128), (128, 256), (256, 512), (512, 640)]

        # layers
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, blocks),
                        DownSample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            ) 

    def forward(self, x):
        x = self.conv_initial(x)
        x = x + self.pos_feature.unsqueeze(0) # Add positional encoding, x: [B, C, H, W]
        h = []
        # downsample
        for down in self.downs:
            block, downsample = down
            x = block(x)
            h.append(x)
            x = downsample(x) 

        return x, h # x:(batch_size, 640, 240/2^4, 240/2^4), h: list of feature maps at each resolution:[]


class SingleEncoder(nn.Module):
    def __init__(self, init_dim=64,H=240,W=240, num_inputs=4, dim_mults=(1, 2, 4, 8, 10), blocks=True):
        super(SingleEncoder, self).__init__()
        self.conv_initial = nn.Conv2d(num_inputs, init_dim, 3, padding=1) # 4*240*240 -> 64*240*240
        self.pos_feature = nn.Parameter(torch.empty(init_dim, H, W))
        nn.init.xavier_normal_(self.pos_feature)
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)] # 64, 64, 128, 256, 512, 640
        in_out = list(zip(dims[:-1], dims[1:])) #  [(64, 64), (64, 128), (128, 256), (256, 512), (512, 640)]

        # layers
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, blocks),
                        DownSample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            ) 

    def forward(self, x, target_feature):
        x = self.conv_initial(x)
        x = x + target_feature # Add positional encoding, x: [B, C, H, W]
        h = []
        # downsample
        for down in self.downs:
            block, downsample = down
            x = block(x)
            h.append(x)
            x = downsample(x) 

        return x, h # x:(batch_size, 640, 240/2^4, 240/2^4), h: list of feature maps at each resolution:[]
    
class MultiBaseEncoder(nn.Module):
    def __init__(self, init_dim=64, H = 240, W = 240, num_inputs=4, dim_mults=(1, 2, 4, 8, 10), blocks=True):
        super(MultiBaseEncoder, self).__init__()
        self.conv_initial = nn.Conv2d(num_inputs, init_dim, 3, padding=1) # 4*240*240 -> 64*240*240
        # self.pos_feature = nn.Parameter(torch.randn(init_dim, H, W))
        self.pos_feature = nn.Parameter(torch.empty(init_dim, H, W))
        nn.init.xavier_normal_(self.pos_feature)
        
        
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)] # 64, 64, 128, 256, 512, 640
        in_out = list(zip(dims[:-1], dims[1:])) #  [(64, 64), (64, 128), (128, 256), (256, 512), (512, 640)]

        # layers
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, blocks),
                        DownSample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            ) 

    def forward(self, x):
        # print("hhhhhhhh:",x.shape)
        x = self.conv_initial(x)
        x = x + self.pos_feature.unsqueeze(0)  # Add positional encoding, x: [B, C, H, W]
        h = []
        # downsample
        for down in self.downs:
            block, downsample = down
            x = block(x)
            h.append(x)
            x = downsample(x) 

        return x, h # x:(batch_size, 640, 240/2^4, 240/2^4), h: list of feature maps at each resolution:[]
    
class Decoder(nn.Module):
    def __init__(self, init_dim=64, num_outputs=1, dim_mults=(1, 2, 4, 8, 10), skip=True, blocks=True, skip_multiplier=2):
        super(Decoder, self).__init__()
        self.skip = skip
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)] # (64, 64, 128, 256, 512, 640)
        in_out = list(zip(dims[:-1], dims[1:])) # [(64, 64), (64, 128), (128, 256), (256, 512), (512, 640)]

        # layers
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            if skip:
                dim_skip = int(dim_out*skip_multiplier)
            else:
                dim_skip = dim_out
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_skip, dim_in, blocks),
                        UpSample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )


        self.conv_final = nn.Sequential(
            ResnetBlock(init_dim, init_dim, blocks), nn.GroupNorm(8, init_dim), nn.SiLU(), nn.Conv2d(init_dim, num_outputs, 1)
        )

    def forward(self, x, h):
        # upsample
        for n, up in enumerate(self.ups):
            block, upsample = up
            if self.skip:
                x = torch.cat((x, h[::-1][n]), dim=1)
            x = block(x)
            x = upsample(x)

        return self.conv_final(x)
    
class CHattnblock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid())

    def forward(self, x):
        w = self.attn(x)
        # print(w.shape)
        return w
    
def Sinkhorn_log_exp_sum(C, mu, nu, epsilon,mask=None):
    
    def _log_boltzmann_kernel(u, v, epsilon, C=None):
        kernel = -C + u.unsqueeze(-1) + v.unsqueeze(-2)
        kernel /= epsilon
        return kernel
    
    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)
    thresh = 1e-6
    max_iter = 100
    # if mask is not None:
        # mask.unsqueeze(1)  # mask: (B*H*W, M) -> (B*H*W, M, 1)        
    for i in range(max_iter):
        
        u0 = u  # useful to check the update
        K = _log_boltzmann_kernel(u, v, epsilon, C)
        if mask is not None:
            K = K.masked_fill(~mask.unsqueeze(-1), -1e9)  # mask: (B*H*W, M)

        u_ = torch.log(mu + 1e-8) - torch.logsumexp(K, dim=2)
        u = epsilon * u_ + u
        
        K_t = _log_boltzmann_kernel(u, v, epsilon, C).permute(0, 2, 1).contiguous()
        if mask is not None:
            K_t = K_t.masked_fill(~mask.unsqueeze(-2), -1e9)  # mask: (B*H*W, M) -> (B*H*W, M, 1)

        v_ = torch.log(nu + 1e-8) - torch.logsumexp(K_t, dim=2)
        v = epsilon * v_ + v
        
        err = (u - u0).abs().mean()
        if err.item() < thresh:
            break
    
    K = _log_boltzmann_kernel(u, v, epsilon, C)
    if mask is not None:
        K = K.masked_fill(~mask.unsqueeze(-1), -1e9)
    
    T = torch.exp(K) 
    if mask is not None:
        T = T.masked_fill(~mask.unsqueeze(-1), 0.0)

    return T


class AttentionOT(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.eps = 0.05 

    def forward(self, xq, xk, xv, mask=None):
        Nq, B, C = xq.size() 
        Nv = xv.size()[1]

        xq = self.q(xq)
        xk = self.k(xk)
        v = self.v(xv)
        
        # assign variables
        _, M, _ = xk.shape
        xq = F.normalize(xq, dim=-1, p=2)
        xk = F.normalize(xk, dim=-1, p=2)

        # compute score map 
        sim = torch.einsum('bmc,nbc->bnm', xk, xq)
        sim = sim.permute(0,2,1) 
        sim = sim.contiguous().view(B, M, Nq) 
        wdist = 1.0 - sim

        # optimally transport score map
        if mask is not None:
            # mask: (b*h*w, num_inputs) indicates which key/value positions are valid
            valid_counts = mask.sum(dim=1, keepdim=True).float()  # (b*h*w, 1)
            # input(f"valid_counts: {valid_counts.shape}")
            # Handle cases with no valid positions
            xx = mask.float() / valid_counts.clamp(min=1.0)  # (b*h*w, num_inputs)

            # Handle cases with no valid positions
            no_valid_mask = (valid_counts == 0).squeeze(-1)  # (b*h*w,)
            if no_valid_mask.any():
                xx[no_valid_mask] = 1.0 / M  # Uniform distribution as fallback
        else:
            xx = torch.zeros(B, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        # input(f"xx: {xx.shape}")
        # xx = torch.zeros(B, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        yy = torch.zeros(B, Nq, dtype=sim.dtype, device=sim.device).fill_(1. / Nq)
        T = Sinkhorn_log_exp_sum(wdist, xx,yy, self.eps, mask)
        
        # T * score map
        score_map = (M * Nq * sim * T).view(B, M, Nq) 
        attn_save = score_map.clone().contiguous().sum(dim=-1).squeeze(-1)
        if mask is not None:
            attn_save = attn_save.masked_fill(~mask, 0.0)
        attn = rearrange(T.view(B, M, Nq), 'b m n -> n b m', b = B, n = Nq) 
        # attn = self.attn_drop(attn)

        x = torch.einsum('nbm,bmc->nbc', attn, v)
        x = self.proj(x)
        # x = self.proj_drop(x)
        
        return x, attn_save


class TOEncoder(nn.Module): # Target-Oriented encoder
    def __init__(self, dim=64, num_inputs=4, dim_mults=(1, 2, 4, 8, 10), n_layers=2, blocks=True, n_tokens=0,H=240, W=240,batch_size=6):
        super().__init__()
        self.num_inputs = num_inputs
        self.encoder_multi = MultiBaseEncoder(init_dim=dim,H=H, W=W, num_inputs=num_inputs, dim_mults=dim_mults, blocks=blocks)
        # self.encoder_early = Encoder(dim, num_inputs, dim_mults, blocks)
        self.encoder_single = nn.ModuleList([SingleEncoder(dim,H, W, 1, dim_mults, blocks) for i in range(num_inputs)])
        self.attn_blocks = nn.ModuleList([CHattnblock(dim*dim_mults[-1]) for i in range(num_inputs+1)])
        self.conv1 = nn.Conv2d(dim*dim_mults[-1]*2, dim*dim_mults[-1], 1)
        self.softmax = nn.Softmax(dim=0)
        self.Softmax1 = nn.Softmax(dim=1)
        self.target_features = nn.Parameter(torch.empty(num_inputs, dim, H, W))
        self.AttentionOT = AttentionOT(dim=dim*dim_mults[-1], num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.)
        self.Softmax2 = nn.Softmax(dim=2)
        self.target_features = nn.Parameter(torch.empty(num_inputs, dim, H, W))
        # self.add_information = nn.Parameter(torch.empty(num_inputs,H,W))
        # nn.init.xavier_normal_(self.add_information)
        final_h = H // (2 ** (len(dim_mults)-1))
        final_w = W // (2 ** (len(dim_mults)-1))
        self.querys = nn.Parameter(torch.empty(num_inputs, final_h * final_w,dim*dim_mults[-1]))
        nn.init.xavier_normal_(self.querys)
        
        
    def convert_modals_to_tensor_padded(self, input_modals, num_inputs, device):
        """
        Use pad_sequence method
        """
        from torch.nn.utils.rnn import pad_sequence
        
        batch_size = len(input_modals)
        
        # Convert each modality list to tensor
        modal_lists = [torch.tensor(modals, dtype=torch.long, device=device) for modals in input_modals]

        # Pad to the same length
        padded_modals = pad_sequence(modal_lists, batch_first=True, padding_value=-1)
        
        # Create mask (-1 is padding value)
        mask = (padded_modals != -1)
        
        # Create result tensor
        modal_tensor = torch.zeros((batch_size, num_inputs), dtype=torch.bool, device=device)
        
        # Use scatter operation
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(padded_modals)
        valid_batch_idx = batch_idx[mask]
        valid_modal_idx = padded_modals[mask]
        
        modal_tensor[valid_batch_idx, valid_modal_idx] = True
        
        return modal_tensor
    
    def forward(self, x, input_modals,input_modals_tensor, modalities, train_mode=False):
        final_multi, layers_multi = self.encoder_multi(x) # final_multi: (batch_size, 640, 15, 15), layers_multi: [5 tensors, every tensor is (batch_size, 640, 240, 240)]
        # input(f"layers_multi shape {[layer.shape for layer in layers_multi]}")
        final_single_all = []
        layer_single_all = []
        target_channel_atten = []
        
        batch_target_features = self.target_features[modalities]
        
        for i in range(self.num_inputs):
            final_single, layer_single = self.encoder_single[i](x[:,i:i+1,:], batch_target_features) 
            # final_single: (batch_size, 640, 15, 15), h_middle: [5 tensors, every tensor has different shape]
            # layer_single: every tensor is (batch_size, channels, h, w)
            final_single_all.append(final_single)
            layer_single_all.append(layer_single) # layer_single_all: [[5 tensors, each tensor has different shape]...] num_inputs items 
            target_channel_atten.append(self.attn_blocks[i](final_single)) # self.attn_blocks[i](x_middle) shape: [batch_size, 640, 1, 1]
        target_channel_atten.append(self.attn_blocks[-1](final_multi)) # (num_inputs+1, batch_size, 640, 1, 1)
        
        final_single_all = torch.stack(final_single_all, dim=0).permute(1, 0, 2, 3, 4)  # shape: (batch_size, num_inputs, 640, 15, 15)
        target_channel_atten = torch.stack(target_channel_atten, dim=0).permute(1, 0, 2, 3, 4) # shape: (batch_size, num_inputs+1, 640, 1, 1)
        single_layer_fusion = []
        
        for layer_idx in range(len(layer_single_all[0])):  # len(layer_single_all[0]) is num_inputs
            layer_tensors = [layer_single_all[modal_idx][layer_idx] for modal_idx in range(self.num_inputs)] # layer_tensors: num_inputs items (batch_size, channels, h, w)
            permute_tensors = torch.stack(layer_tensors, dim=0).permute(1, 0, 2, 3, 4) # shape: (batch_size, num_inputs, channels, h, w)
            single_layer_fusion.append(permute_tensors)  # []: layer_size items (batch_size, num_inputs, channels, h, w)
        # modal_position [batch_size, num_inputs] val is True or False, True means the modal is available
        modal_position = self.convert_modals_to_tensor_padded(input_modals, self.num_inputs, x.device) # convert input_modals to tensor padded
        layers_single = []
        mask = modal_position.view(modal_position.shape[0], modal_position.shape[1], 1, 1, 1)
        for layer_idx in range(len(single_layer_fusion)):
            # Use torch.where to directly select
            layer_features = single_layer_fusion[layer_idx]  # (batch_size, num_inputs, channels, h, w)
            selected_features = torch.where(mask, layer_features, torch.zeros_like(layer_features))
            # Sum and calculate average
            fused_layer = selected_features.sum(dim=1)  # (batch_size, channels, h, w)
            modal_counts = modal_position.sum(dim=1, keepdim=True).view(-1, 1, 1, 1) # shape: (batch_size, 1, 1, 1)
            fused_layer = fused_layer / modal_counts 
            layers_single.append(fused_layer)
                    
        feal_fusion_single = final_single_all * target_channel_atten[:,:-1,...] # (batch_size, num_inputs, 640, 15, 15)
        add_fusion_s = final_multi*target_channel_atten[:,-1:,...].squeeze(1)  # (batch_size, 640, 15, 15)
        
        B, num_inputs, C, H, W = feal_fusion_single.shape
        tensor_device = feal_fusion_single.device
        max_modalities_cout = num_inputs - 1
        # Use pad_sequence for unified processing
        padded_indices = pad_sequence(input_modals_tensor, batch_first=True, padding_value=-1)  # (B, max_len)
        # input("padded_indices:", padded_indices.shape)
        # Truncate or pad to max_modalities
        
        if padded_indices.shape[1] > max_modalities_cout:
            print("Error: available modalities count > max modalities count")
            padded_indices = padded_indices[:, :max_modalities_cout]
        elif padded_indices.shape[1] < max_modalities_cout:
            padding = torch.full((B, max_modalities_cout - padded_indices.shape[1]), num_inputs, 
                            dtype=torch.long, device=tensor_device)
            padded_indices = torch.cat([padded_indices, padding], dim=1)
        
        # Create mask
        # mask = (padded_indices < num_inputs)  # (B, max_modalities)
        mask_gather = torch.logical_and(
        padded_indices >= 0,
        padded_indices < num_inputs
    )
        # Handle invalid indices
        indices = torch.clamp(padded_indices, min=0, max=num_inputs-1).long()  # Add max limit
        
        indices_expanded = indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        indices_expanded = indices_expanded.expand(-1, -1, C, H, W)
        # input(indices_expanded)
        gathered = torch.gather(feal_fusion_single, 1, indices_expanded)
        
        mask_expanded = mask_gather.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        mask_expanded = mask_expanded.expand(-1, -1, C, H, W) # (b,)
        
        single_feate_mask = mask_gather.unsqueeze(-1).unsqueeze(-1)  # (B, max_input, 1, 1)
        single_feate_mask = single_feate_mask.expand(B, max_modalities_cout, H, W)  # (B, max_input, H, W)
        
        common_feat_mask = torch.full((B, 1), True, dtype=torch.bool,device=tensor_device)  # (B, 1)
        common_feat_mask_expanded = common_feat_mask.unsqueeze(-1).unsqueeze(-1)
        common_feat_mask_expanded = common_feat_mask_expanded.expand(B, 1, H, W)  # (B, 1, H, W)
        all_feature_mask = torch.concat((common_feat_mask_expanded, single_feate_mask), dim=1)  # (B, 1+max_input, H, W)
        
        gathered_single_modal_feat = torch.where(mask_expanded, gathered, torch.full_like(gathered, 0))        
        
        all_feature = torch.concat((add_fusion_s.unsqueeze(1), gathered_single_modal_feat), dim=1)  # (batch_size, num_inputs, 640, 15, 15)
        
        H_final_feat = all_feature.shape[-2]  
        W_final_feat = all_feature.shape[-1]
        channel_size = all_feature.shape[-3] 
        
        actual_batch_size = all_feature.shape[0]  # Get actual batch_size
        query = self.querys[modalities] # b, 15 * 15, 640
        # print(query.shape)
        # print(self.batch_size)
        
        query = query.view(1, actual_batch_size * H_final_feat * W_final_feat, channel_size)  # (1,batch_size*225, 640)
        key = all_feature.permute(0, 3, 4, 1, 2).reshape(actual_batch_size * H_final_feat * W_final_feat, self.num_inputs, channel_size)
        value = all_feature.permute(0, 3, 4, 1, 2).reshape(actual_batch_size * H_final_feat * W_final_feat, self.num_inputs, channel_size)
        
        mask_expanded_OT = all_feature_mask.view(actual_batch_size * H_final_feat * W_final_feat, self.num_inputs)  # (b*h*w, max_input+1)
        fusion_feat,_ = self.AttentionOT(query, key, value,mask_expanded_OT)
        
        fusion_feat = fusion_feat.squeeze(0).squeeze(0)  # (b*15*15, 640)
        fusion_feat = fusion_feat.view(actual_batch_size, H_final_feat, W_final_feat, channel_size)  # (3, 15, 15, 640)
        fusion_feat = fusion_feat.permute(0, 3, 1, 2)  # (b, 640, 15, 15)
        x_fusion = fusion_feat
        
        if train_mode:
            idx_1ch = x.shape[0] // 2
            x = x_fusion
            layers_multi_and_single = [layer_multi[0:idx_1ch,:] + layer_single[0:idx_1ch,:] for layer_multi, layer_single in zip(layers_multi, layers_single)]
            layers_only_single = [layer_single[idx_1ch:,:] for layer_single in layers_single]
            fusion_layers = [torch.cat([multi_single, only_single], dim=0) for multi_single,only_single  in zip(layers_multi_and_single, layers_only_single)]
        else:
            x = x_fusion
            fusion_layers = []
            for layer_multi, layer_single in zip(layers_multi, layers_single):
                f = []
                for n, modals in enumerate(input_modals):
                    if len(modals) == 1:
                        f.append(layer_single[n:n+1,:])
                    else:
                        f_sum = layer_multi[n:n+1,:] + layer_single[n:n+1,:]
                        f.append(f_sum)
                fusion_layers.append(torch.cat(f, dim=0))
        
        return x, fusion_layers
    

       
# import math

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, dim, patch_size, n_tokens):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(in_channels=dim,
                                       out_channels=dim,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_tokens, dim))
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        #B,C,W,H = x.shape()
        x = self.patch_embeddings(x)
        x = x.flatten(2,3) #B,C,W*H
        h = x.permute(0,2,1) #B,W*H,C

        embeddings = h + self.position_embeddings #B,W*H,C == 1*B,n_tokens,dim
        # embeddings = self.dropout(embeddings)
        return embeddings

class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim*4)
        self.fc2 = nn.Linear(dim*4, dim)
        self.act_fn =  nn.SiLU()
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
# Attention module
class Attention(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super().__init__()

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(0.1)
        self.proj_dropout = nn.Dropout(0.1)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output
    
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size)
        self.attn = Attention(hidden_size, n_heads)

    def forward(self, x):
        h = x

        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

# Position embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        if time.dtype == torch.int64:
            return embeddings
        else:
            return embeddings.type(time.dtype)

class ModalityInfuser(nn.Module):
    def __init__(self, hidden_size, patch_size, n_tokens, n_layers, n_heads, modality_embed):
        super().__init__()
        self.modality_embed = modality_embed
        #n_tokens = int((240/(2**n_downs)/patch_size)**2)
        self.modality_embedding = nn.Sequential(
                SinusoidalPositionEmbeddings(hidden_size), # hidden_size is 640
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        
        self.embedding =Embeddings(hidden_size, patch_size, n_tokens)
        self.layers = nn.ModuleList([])
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        for _ in range(n_layers):
            self.layers.append(
                TransformerBlock(hidden_size, n_heads)
            )

    def forward(self, x, m): # x is common Latent Feature z (b,640,15,15), m is target modality (b,1)
        B,C,W,H =x.shape
        h = self.embedding(x)
        if self.modality_embed:
            m = self.modality_embedding(m) 
            h = rearrange(m, "b c -> b 1 c") + h

        for layer_block in self.layers:
            h = layer_block(h)
        h = self.encoder_norm(h)
        h = h.permute(0,2,1).contiguous().view(B,C,W,H)
        return h

class TOMS(nn.Module):
    def __init__(self, dim, num_inputs, num_outputs, dim_mults, n_layers, skip, blocks, image_size=240,H=240,W=240,batch_size=6):
        super().__init__()
        patch_size=1
        n_tokens = int((image_size/(2**(len(dim_mults)-1))/patch_size)**2)
        self.encoder = TOEncoder(dim, num_inputs, dim_mults, n_layers, blocks, n_tokens=n_tokens, H=H, W=W,batch_size=batch_size)
        self.decoder = Decoder(dim, num_outputs, dim_mults, skip, blocks)
        self.middle = ModalityInfuser(hidden_size=dim*dim_mults[-1],  patch_size=1, n_tokens=n_tokens, n_layers=n_layers, n_heads=16, modality_embed=True)

#PatchGAN Discriminator (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538)
class Discriminator(nn.Module):
    def __init__(self, channels=1, num_filters_last=32, n_layers=3, n_classes=4, ixi=False):
        super(Discriminator, self).__init__()

        layers = [nn.Conv2d(channels, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.GroupNorm(8, num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]
        self.model = nn.Sequential(*layers)
        self.final = nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1, bias=False)
        if ixi:
            self.classifier = nn.Conv2d(num_filters_last * num_filters_mult, n_classes, 31, bias=False)
        else:
            self.classifier = nn.Conv2d(num_filters_last * num_filters_mult, n_classes, 29, bias=False)

    def forward(self, x):
        x = self.model(x)
        logits = self.final(x)
        labels = self.classifier(x)
        return logits, labels.view(labels.size(0), labels.size(1))