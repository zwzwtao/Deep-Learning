# Anime Character Generator


<img src="https://github.com/AidenOdas/Deep-Learning/blob/master/Tensorflow%202.0/GAN/pictures/IMG_2259.GIF" width=470 height=470>


trained on 50 thousand anime character pictures


GAN
* The learning rate for generator and discriminator is 0.005(currently the most stable learning rate)

* There may be a mode collapse if using Adam optimizer for discriminator, after switch to RMSProp, haven't found this problem again(still testing).

WGAN
* GP factor for WGAN is set to 10

* Learning rate for discriminator is set to 0.005, and 0.002 for generator.

* Training strategy: the generator learns each time after discriminator being trained for 5 epochs.
