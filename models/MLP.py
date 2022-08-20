class TransformerMLPWithConv(nn.Module):

    def __init__(self, channels, mlp_dim):
        
        super().__init__()
        
        self.dim1 = channels
        self.dim2 = mlp_dim
        self.linear1 = nn.Conv3d(self.dim1, self.dim2, 1, 1, 0)
        self.drop1 = nn.Dropout(0.1, inplace=True)
        self.act = nn.GELU()
        self.linear2 = nn.Conv3d(self.dim2, self.dim1, 1, 1, 0) 
        self.drop2 = nn.Dropout(0.1, inplace=True)
        self.dwc = nn.Conv3d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)
    
    def forward(self, x):
        
        x = self.drop1(self.act(self.dwc(self.linear1(x))))
        x = self.drop2(self.linear2(x))
        
        return x