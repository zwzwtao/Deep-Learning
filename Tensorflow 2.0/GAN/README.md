# Anime Character Generator

Training progress:<br>
<img src="https://github.com/AidenOdas/Deep-Learning/blob/master/Tensorflow%202.0/GAN/pictures/IMG_2259.GIF" width=300 height=300>


trained on 50 thousand anime character pictures


GAN
* The learning rate for generator and discriminator is 0.005(currently the most stable learning rate)

* There may be a mode collapse if using Adam optimizer for discriminator, after switch to RMSProp, haven't found this problem again(still testing).

WGAN
* GP factor for WGAN is set to 10

* Variaty of generated pictures is bigger than which of GAN.
