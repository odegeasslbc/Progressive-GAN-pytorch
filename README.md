# Progressive-GAN-pytorch
A pytorch implementation of Progressive-GAN that is actually works, readable and simple to customize

## Description
I simplify the code of training a Progressive-GAN, making it easier to read and customize, for the purpose of research.  
This implementation is portable with minimal library dependency (only torch and torchvision) and just 2 code modules. In the code, you can easily modeify the training-schema, the loss function, and the network structure, etc.  
Enjoy the benefit of the progressive-growing infrastructure and port it to your own research and product!

## How to train
To train the code, just run:
'''
python train.py --path /path/to/image-folder
'''
