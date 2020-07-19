import torch

class Discriminator(torch.nn.Module):
    def __init__(self, Label_size, Channel_size, Picture_size):
        super().__init__()
        
        # TO DO : padding should be same
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 3, out_channels= 64, kernel_size=5, stride= 2, padding=2),
            torch.nn.BatchNorm2d(64,momentum= 0.9),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 64, out_channels= 128, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(128,momentum= 0.9),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 128, out_channels= 256, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(256, momentum= 0.9),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 256, out_channels= 512, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(512,momentum= 0.9),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        
        self.flatten = torch.nn.Flatten()
        
        self.fully_connected = torch.nn.Sequential(
            torch.nn.Linear(2048 + Label_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        )
        self.Label_size = Label_size
        
    def forward(self, images, labels):
        
        temp = self.conv1(images)
        temp = self.conv2(temp)
        temp = self.conv3(temp)
        temp = self.conv4(temp)
        
        temp = self.flatten(temp)
        
        # bring label in right form and add as dimention
        temp_labels = torch.nn.functional.one_hot(labels, num_classes = self.Label_size)
        temp = torch.cat([temp, temp_labels.float()], 1)
        
        temp = self.fully_connected(temp)
        return temp
        