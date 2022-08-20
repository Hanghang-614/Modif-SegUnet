class MyTransformer(nn.Module):

    def __init__(self,  n_heads, n_head_channels, n_groups,channels, mlp_dim):
        
        super().__init__()
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels =self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.dim1 = channels
        self.dim2 = mlp_dim
        self.linear1 = nn.Conv3d(self.dim1, self.dim2, 1, 1, 0)
        self.drop1 = nn.Dropout(0.1, inplace=True)
        self.act = nn.GELU()
        self.linear2 = nn.Conv3d(self.dim2, self.dim1, 1, 1, 0) 
        self.drop2 = nn.Dropout(0.1, inplace=True)
        self.dwc = nn.Conv3d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)
        self.attention = DAttentionBaseline(n_heads,n_head_channels,n_groups)
        self.mlp = TransformerMLPWithConv(channels,mlp_dim)
    
    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.mlp(x)
        
        return x