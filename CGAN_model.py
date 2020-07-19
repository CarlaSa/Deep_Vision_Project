"""
This is a Pytorch implimentation of the Conditional General Adivisary Network 
as stated in https://github.com/mafda/generative_adversarial_networks_101/blob/master/src/cifar10/03_CGAN_CIFAR10.ipynb
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from CGAN_Generator import Generator
from CGAN_Discriminator import Discriminator
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from tqdm.auto import tqdm


torch.autograd.set_detect_anomaly(True)

class CGAN:
    """
    The Generator gets an input of (Noise_size) of Noise 
    and a Label (in one-hot encoding), and generates a picture of shape
    (Channel_size, Picture_size, Picture_size).

    The Discriminator Takes a Picture with shape (Channel_size, Picture_size, Picture_size) 
    and a label and produces a number representing the "probability of realness" of 
    the picture. 

    """
    def __init__(self, Noise_size, Label_size, Channel_size, Picture_size, Batch_size,\
                epochs_trained = 0, lr = 0.0002 , use_cuda = False, weights_Generator = None, \
                weights_Discriminator = None, smoothness = 0.1):
        self.Batch_size = Batch_size
        self.Noise_size = Noise_size
        self.Generator = Generator(Noise_size, Label_size, Channel_size, Picture_size)
        self.Discriminator = Discriminator(Label_size, Channel_size, Picture_size)
        if weights_Generator is not None:
            self.Generator = Generator.load_state_dict(torch.load(weights_Generator, \
                                                        map_location=torch.device('cpu')))
        if weights_Discriminator is not None:
            self.Discriminator = Discriminator.load_state_dict(torch.load(weights_Discriminator, \
                                                        map_location=torch.device('cpu')))
        if use_cuda:
            self.Generator = self.Generator.cuda()
            self.Discriminator = self.Discriminator.cuda()

        self.opt_gen = torch.optim.Adam(self.Generator.parameters(), \
                lr = lr, betas = (0.5, 0.999))
        self.opt_disc =torch.optim.Adam(self.Generator.parameters(), \
                lr = lr, betas = (0.5, 0.999))
        self.loss = torch.nn.BCELoss()

        self.real_label = 1
        self.false_label = 0
        self.smoothness = smoothness
        self.use_cuda = use_cuda

        self.class_names  = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
    
    def train_step(self, images, categories):
        if self.use_cuda:    
            images = images.cuda()
            categories = categories.cuda()

        #### train Discriminator ---------------------------------------
        self.opt_disc.zero_grad()
        # train with true pictures --------------

        output = self.Discriminator(images, categories)    
        label = torch.full(size = (self.Batch_size,1), fill_value = self.real_label * (1- self.smoothness))
        if self.use_cuda:
            label = label.cuda()
        disc_loss_real = self.loss(output, label)
    
        #train with generated pictures -----------
        #generate input from Generator
        # noise is taken from N(0,1) distribution
        noise = torch.randn(size = (self.Batch_size, self.Noise_size))
        gen_categories = torch.randint(low = 0, high = 10, size = (self.Batch_size,))
        if self.use_cuda:
            noise = noise.cuda()
            gen_categories = gen_categories.cuda()
        inp = self.Generator(noise, gen_categories)
        
        output = self.Discriminator(inp.detach(), gen_categories)
        label = torch.full(size = (self.Batch_size, 1), fill_value = self.false_label)
        if self.use_cuda:
            label = label.cuda()
        disc_loss_false = self.loss(output, label)
        
        disc_loss = 0.5 * (disc_loss_false + disc_loss_real)
        disc_loss.backward()

        #### train Generator ---------------------------------------
        #we don't calculate noise and gen_categories twice, inp stays the same
        # we updated Disc so we calculate it again
        self.opt_gen.zero_grad()
        #inp = self.Generator(noise, gen_categories)
        output = self.Discriminator(inp, gen_categories)  

        label = torch.full(size = (self.Batch_size, 1), fill_value = self.real_label)
        if self.use_cuda:
            label = label.cuda()
        gen_loss = self.loss(output, label)
        gen_loss.backward()

        self.opt_disc.step()
        self.opt_gen.step()
        return disc_loss_real.item(), disc_loss_false.item(), gen_loss.item()
    
    def save_model(self, model, name, epoch):
        torch.save(model.state_dict(), f'./weights/{name}_e{epoch}.ckpt')

    def train(self, num_epochs, dataloader, feedback_freq = 5, save_freq = 10):
        gen_loss_list = [list() for i in range(num_epochs)]
        disc_loss_real_list = [list() for i in range(num_epochs)]
        disc_loss_fake_list = [list() for i in range(num_epochs)]
        for epoch in tqdm(range(num_epochs)):
            for images, categories in tqdm(dataloader):
                disc_loss_real, disc_loss_fake, gen_loss = self.train_step(images, categories)
                gen_loss_list[epoch].append(gen_loss)
                disc_loss_real_list[epoch].append(disc_loss_real)
                disc_loss_fake_list[epoch].append(disc_loss_fake)
            
            
            if epoch % feedback_freq == 0 or epoch == num_epochs - 1 : 
                # always in last epoch
                print("epoch: " + str(epoch))
                
                #example picture
                inp, cat = self.generate_example(2)
                if self.use_cuda:
                    inp = inp.cpu()
                Img = F.to_pil_image(inp[0])
                plt.imshow(Img)
                plt.show()
                print("label: " + self.class_names[cat[0].item()])

                # print loss graph
                #x = range(len(gen_loss_list[epoch]))
                plt.figure(num = 4, figsize=(8, 5))
                plt.plot(gen_loss_list[epoch], color = "orange", label = "Loss Generator")
                plt.plot(disc_loss_real_list[epoch], color = "green", label = "Loss Discriminator on Real")
                plt.plot(disc_loss_fake_list[epoch], color = "blue",  label = "Loss Discriminator on Generated")
                plt.legend(loc = "upper right")
                plt.show()

            if epoch % save_freq == 0 or epoch == num_epochs -1:
                self.save_model(self.Generator, "CGAN_gen", epoch)
                self.save_model(self.Discriminator, "CGAN_disc", epoch)

        
            
    def generate_example(self, n):
        noise = torch.randn(size = (n, self.Noise_size))
        gen_categories = torch.randint(low = 0, high = 10, size = (n,))
        if self.use_cuda:
            noise = noise.cuda()
            gen_categories = gen_categories.cuda()
        inp = self.Generator(noise, gen_categories) 
        return inp , gen_categories





