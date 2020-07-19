import torch

class Generator(torch.nn.Module):
    """
    This Generator is made up of Blocks of Transposed Convolutional Layes, with BatchNorm and LeakyReLU


    """
    def __init__(self, Noise_size, Label_size, Channel_size, Picture_size):
        super().__init__()
        """

        """
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(in_features = Noise_size + Label_size , out_features= 512 *2*2),
            torch.nn.BatchNorm1d(2048, momentum =  0.9),
            torch.nn.LeakyReLU(0.1)
        )
        self.conv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels= 512 , out_channels= 256, kernel_size=5, stride=2, padding=2, output_padding= 1),
            torch.nn.BatchNorm2d(256,momentum =  0.9),
            torch.nn.LeakyReLU(0.1)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels= 256, out_channels= 128, kernel_size=5, stride=2, padding=2, output_padding= 1),
            torch.nn.BatchNorm2d(128, momentum =  0.9),
            torch.nn.LeakyReLU(0.1)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels= 128, out_channels= 64, kernel_size=5, stride=2, padding=2, output_padding= 1),
            torch.nn.BatchNorm2d(64, momentum =  0.9),
            torch.nn.LeakyReLU(0.1)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels= 64, out_channels= 3, kernel_size=5, stride=2, padding=2, output_padding= 1),
        )
        self.Label_size = Label_size

    def forward(self, noise, labels):
        """
        noise has shape (BATCHSIZE, 1)
        labels has shape (BATCHSIZE,)
        """
        
        labels = torch.nn.functional.one_hot(labels, num_classes = self.Label_size) #shape (BATCH_SIZE, 10)
        temp = torch.cat((noise, labels.float()), 1) #shape (BATCH_SIZE, NOISE_SIZE)
        temp = self.dense(temp) #shape (BATCH_SIZE, 2048)
        
        temp = torch.reshape(temp, (-1, 512,2,2))  # shape (BATCH_SIZE, 512,2,2)
        temp = self.conv1(temp) # shape (BATCH_SIZE, 256, 4, 4)
        temp = self.conv2(temp) # shape (BATCH_SIZE, 128, 8, 8)
        temp = self.conv3(temp) # shape (BATCH_SIZE, 64, 16, 16)
        temp = self.conv4(temp) # shape (BATCH_SIZE, 3, 32, 32)
        temp = torch.tanh(temp) # output between -1, 1

        return 0.5 * (1+ temp) # output between 0, 1
        