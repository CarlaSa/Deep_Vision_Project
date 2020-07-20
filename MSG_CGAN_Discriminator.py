import torch

class Discriminator(torch.nn.Module):
    def __init__(self, Label_size, Channel_size, Picture_size):
        super().__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = Channel_size, out_channels= 64, kernel_size=5, stride= 2, padding=2),
            torch.nn.BatchNorm2d(64,momentum= 0.9),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 67, out_channels= 128, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(128,momentum= 0.9),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 131, out_channels= 256, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(256, momentum= 0.9),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 259, out_channels= 512, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(512,momentum= 0.9),
            torch.nn.LeakyReLU(negative_slope=0.1)
        )
        
        self.flatten = torch.nn.Flatten()
        
        self.fully_connected = torch.nn.Sequential(
            torch.nn.Linear(2058, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        )
        self.Label_size = Label_size
        
    def forward(self, image1, image2, image3, image4, labels): #größtes zuerst
        temp = self.conv1(image1) # [100, 64, 16, 16]
        temp = torch.cat([temp, image2],1) # [100, 67, 16, 16]
        
        temp = self.conv2(temp) # [100, 128, 8, 8]
        temp = torch.cat([temp, image3],1) # [100, 131, 8, 8]
        
        temp = self.conv3(temp) # [100, 256, 4, 4]
        temp = torch.cat([temp, image4],1) # [100, 259, 4, 4]

        temp = self.conv4(temp) # [100, 512, 2, 2]
        temp = self.flatten(temp) # [100, 2048]

        # bring label in right form and add as dimention
        temp_labels = torch.nn.functional.one_hot(labels, num_classes = self.Label_size)
        temp = torch.cat([temp, temp_labels.float()], 1) # [100, 2058]
        out = self.fully_connected(temp)
        
        return out
        
        