class MyDownSample(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(channels,channels,4,2,1)
                                 ,nn.BatchNorm3d(channels)
                                 ,nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(channels,channels*2,3,1,1)
                                 ,nn.BatchNorm3d(channels*2)
                                 ,nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x