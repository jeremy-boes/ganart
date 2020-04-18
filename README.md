#  GANART: generative adversalial networks to paint Magic: the Gathering art

Dataset not provided for legal reasons. Example in this file are results of GANs trained on MtG lands illustrations.

[Check out this blog post](https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f) for an introduction to Generative Networks. 

Selected results from unrolled GAN trained on all lands
<img src=".images/all_selection_unrolled.png" width="1136">

<img src=".images/conv.gif" width="240">

## Lin-GANART
Simple GAN composed of two perceptrons.

<img src=".images/five-linear.png" width="1300">

## GANART (wip)
Unrolled DCGAN

Sample result trained on all lands

<img src=".images/small_sample.png" width="600">

### Previous versions selected results

Standard DCGAN trained on basic lands only
<img src=".images/z9-all.png" width="814">

Standard DCGAN trained on all lands
<img src=".images/all_selection.png" width="1136">


## GANART-load
Load a pretrained Generator and generates art.
