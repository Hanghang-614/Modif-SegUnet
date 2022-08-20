class DAttentionBaseline(nn.Module):

    def __init__(
        self,  n_heads, n_head_channels, n_groups, 
    ):

        super().__init__()
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels =self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        
        kk = 4
        self.conv_offset = nn.Sequential(
         nn.Conv3d(self.n_group_channels, self.n_group_channels, kk, 2, kk//2, groups=self.n_group_channels),
         LayerNormProxy(),
         nn.GELU(),
         nn.Conv3d(self.n_group_channels, 3, 1, 1, 0, bias=False)
        )
        self.proj_q = nn.Conv3d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv3d(
            self.n_group_channels, self.n_group_channels,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv3d(
            self.n_group_channels, self.n_group_channels,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv3d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(0.1, inplace=False)
        self.attn_drop = nn.Dropout(0.1, inplace=False)

    
    @torch.no_grad()
    def _get_ref_points(self,D_key, H_key, W_key, B, dtype, device):
        
        ref_y ,ref_x, ref_z = torch.meshgrid( 
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device), 
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            torch.linspace(0.5, D_key - 0.5, D_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x,ref_z), -1)
        ref[..., 2].div_(D_key).mul_(2).sub_(1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1, -1) # B * g D H W 2
        
        return ref

    def forward(self, x):
        x_shape = x.size()
        B, C, D, H, W = x_shape    #x  (b,c,d,h,w)
        dtype, device = x.dtype, x.device 
        
        q = self.proj_q(x)    #q  (b,c,d,h,w)
        q_off = einops.rearrange(q, 'b (g c) d h w -> (b g) c d h w', g=self.n_groups, c=self.n_group_channels)      #q_off  (b*n_groups,n_group_channels,d,h,w)
        offset = self.conv_offset(q_off)        #offset  (b*b_groups,2,D/r,H/r,W/r) 
        Dk, Hk, Wk =offset.size(2), offset.size(3), offset.size(4)
        n_sample = Dk * Hk * Wk
        
            
        offset = einops.rearrange(offset, 'b p d h w -> b d h w p')    #offset  (b*n_groups,D/r,H/r,W/r,2)
        reference = self._get_ref_points(Dk, Hk, Wk, B, dtype, device) #reference (b*n_groups,D/r,H/r,W/r,2)
        reference=einops.rearrange(reference, 'b h w d p -> b d h w p')
            
        pos = (offset + reference).tanh()
        
        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels,D, H, W),   #input (b*n_groups,n_group_channels,d,h,w)
            grid=pos,
            mode='bilinear', align_corners=True) # B * g, Cg, Dg,Hg, Wg
            
        

        q = q.reshape(B * self.n_heads, self.n_head_channels, D * H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        
        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, Ns
        attn = attn.mul(self.scale)
        

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        

        out = out.reshape(B, C, D,  H, W)
        
        y = self.proj_drop(self.proj_out(out))
        
        return y
    def proj_out(self, x):
        x_shape = x.size()
        n, ch, d, h, w = x_shape
        x = rearrange(x, "n c d h w -> n d h w c")
        x = F.layer_norm(x, [ch])
        x = rearrange(x, "n d h w c -> n c d h w")
        return x
class LayerNormProxy(nn.Module):
    def __init__(self):
        
        super().__init__()

    def forward(self, x):
        x_shape = x.size()
        n, ch, d, h, w = x_shape
        x = rearrange(x, "n c d h w -> n d h w c")
        x = F.layer_norm(x, [ch])
        x = rearrange(x, "n d h w c -> n c d h w")
        return x