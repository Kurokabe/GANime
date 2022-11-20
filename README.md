# GANime: Video generation of anime content conditioned on two frames

[Paper](./assets/GANime_paper.pdf) | [Presentation](https://docs.google.com/presentation/d/1KtN6-LmA6fbbY3wG6_Hz75HbOFL3J7vC/edit?usp=sharing&ouid=116500441313574364877&rtpof=true&sd=true)

**tl;dr** This is GANime, a model capable to generate video of anime content based on the first and last frame. This model is trained on a custom dataset based on the Kimetsu no Yaiba anime. It is composed of two model, a VQ-GAN for image generation, and a GPT2 transformer to generate the video frame by frame.

<p align="center">
    <img src=./assets/opu.jpg height="100" />
    <img src=./assets/omu.png height="100" />
    <img src=./assets/mse.png height="100" />
    <img src=./assets/hes-so.jpg height="100" />
</p>

This project is a Master thesis realised by Farid Abdalla at HES-SO in partnership with Osaka Prefecture University (now renamed to Osaka Metropolitan University) in Japan. A PyTorch implementation is available on [this repository](https://github.com/Kurokabe/TorchGANime).

All implementation details are available in the [paper pdf](./assets/GANime_paper.pdf)

## Good results
For each pair of rows, the first row is the generated result and the second row is the ground truth.
![](./assets/good_results/good_results.gif)


## Surprising results
For each pair of rows, the first row is the generated result and the second row is the ground truth.
![](./assets/interesting_results/surprising_results.gif)

Some results are quite surprising. For instance when the first and last frame are identical, the model is capable to generate some animations. For instance, some characters seems to be breathing even though the ground truth is still. When something appears suddenly (upper right video), the model made it appear with a fading effect.

The lower left picture with Zenitsu is interesting: it seems that the VQ-GAN learned that when generating an eye, it must put a pupil inside it, so generating a white eye did not make sense for the model.

For the clock (bottom-middle), the generated video moves the clock arms even though the first and last pictures are identical.


## Dataset
### Image
| Dataset  | Link |
|---------|-----|
| Kimetsu no Yaiba | [link](https://drive.google.com/file/d/1Wm-MVUZTtkcXiQPDLVe4SMj8hUq-XoWy/view?usp=sharing) |

### Video
| Dataset  | Link |
|---------|-----|
| Kimetsu no Yaiba | [link](https://drive.google.com/file/d/1XpSycjeOqRhtuG5E9o4gRFq7jEDbw2VZ/view?usp=sharing) |
### Image
| Dataset  | Link |
|---------|-----|
| Kimetsu no Yaiba | [link](https://drive.google.com/file/d/1Wm-MVUZTtkcXiQPDLVe4SMj8hUq-XoWy/view?usp=sharing)  |

## Pretrained model
### VQ-GAN
| Dataset | Model | Link |
|---------|-------|-----|
| MovingMNIST | moving_mnist_image.yaml | [link](https://drive.google.com/file/d/1uKdwvjTbAz_T8eHa19VEaJu3Ql68YmKk/view?usp=sharing) |
| Kimetsu no Yaiba | kny_image_full_vgg19.yaml | [link](https://drive.google.com/file/d/1rPNdljYa2cO5INtKefWzeeVIg2_W7i4w/view?usp=sharing) |

### Transformer
| Dataset | Model | Link |
|---------|-------|-----|
| Kimetsu no Yaiba | kny_video_gpt2_medium.yaml.yaml | [link](https://drive.google.com/file/d/19a1t3ZE0bXx3gRhflWP6gJAF6PvoJNLz/view?usp=sharing) |

## VGG Weights
[link](https://drive.google.com/file/d/1AcAKt_bzXmmILGRHj8cN_pH56JSEVsY5/view?usp=sharing) 


Instructions on how to train / generate will come later
