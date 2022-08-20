class MyLayer(nn.Module):
    def __init__(
        self,n_heads, n_head_channels, n_groups,channels, mlp_dim
        
    ) -> None:
        super().__init__()
        self.transformer = MyTransformer(n_heads,n_head_channels,n_groups,channels,mlp_dim)
        self.down = MyDownSample(channels)

    def forward(self, x):
        x_shape = x.size()
        x = self.transformer(x)
        x = self.down(x)
        return x