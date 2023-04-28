# Group 54 Reproducibility Blog Post 
Isa Rethans, 4965809, i.g.m.rethans@student.tudelft.nl \
Paul Frolke, 4888693, p.r.frolkee@student.tudelft.nl \
Gijs Admiraal, 4781669, g.j.admiraal@student.tudelft.nl

## Table of contents

<!-- - [Table of contents](#table-of-contents)
- [Sources](#sources)
- [Introduction](#introduction)
- [Problem statement](#problem-statement)
- [Cross Encoder Architecture](#cross-encoder-architecture)
- [The Dataset](#the-dataset)
- [Method](#method)
  - [Data processing](#data-processing)
  - [Dataset creation](#dataset-creation)
- [Approach](#approach)
- [Training procedure](#training-procedure)
- [Results](#results)
- [Discussion / Challenges / problems](#discussion--challenges--problems)
- [Conclusion](#conclusion)
- [Project approach + division of work](#project-approach--division-of-work) -->

[TOC]

## Sources

**Original Paper**
* Paper: <https://openaccess.thecvf.com/content/ICCV2021/papers/Sun_Cross-Encoder_for_Unsupervised_Gaze_Representation_Learning_ICCV_2021_paper.pdf>
* Data: <https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/>
* GitHub: <https://github.com/sunyunjia96/Cross-Encoder>

**Our Reproduction**
* GitHub: <https://github.com/pfrolke/CS4240-DL-Reproducibility>


## Introduction

Gaze estimation is a rapidly growing area of research that involves predicting where a person is looking based on the orientation of their eyes. A persons gaze can help understand the persons desires, intent and state of mind. This technology has important applications in fields such as human-computer interaction, virtual reality, autonomous driving and cognitive neuroscience. 

Accurate gaze estimation can enable more natural and intuitive interfaces for human-computer interaction, such as eye-controlled devices, and can also provide insights into how people perceive and process visual information. Moreover, gaze estimation can be used in autonomous driving to detect driver distractions.

## Problem statement
In this work, we attempt to reproduce the results of the paper "Cross-Encoder for Unsupervised Gaze Representation Learning" by Yunjia Sun, et al. The paper introduces an auto-encoder-like framework, called a Cross-Encoder, for unsupervised gaze representation learning. The authors highlight that learning a good representation for gaze is non-trivial due to the fact that features of gaze direction features are always intertwined with features of the eye's physical appearance. With that in mind, the authors designed a framework that learns to disentangle the gaze and eye features. In the paper, the performance of the Cross-Encoder is evaluated using experiments on various public datasets and with various values for the hyperparameters $d_g$ and $d_e$ representing the the dimension of the gaze and eye and the feature respectively. Our tasks was to reproduce one specific number: the angular error on the Columbia Gaze dataset with $d_g=15$ and $d_e=32$ which is listed in Table 1 of the paper.

## Cross Encoder Architecture
The Cross-Encoder is a modification of an auto-encoder. A conventional auto-encoder is a feedforward network that tries to reproduce its input at the output layer. It consists of an encoder which encodes the input to a low-dimensional embedding and a decoder which tries to reconstruct the image based on the embedding. To incorporate the goal of separating the gaze and eye features, Cross Encoder splits the embedding into two parts, the so called shared feature and specific feature. Furthermore, the input consists of pairs of images instead of just one image. After being encoded to a specific and shared feature, the two images are reconstructed according to its specific feature and the other's shared feature.

### Input
The Cross-Encoder is trained using two types of image pairs. An eye similar pair is a pair that has the same eye but a different gaze. A gaze similar pair, is a pair of images with the same gaze but with different eyes. When the eye similar pair is fed into the network, the shared feature will be the eye feature and the specific feature will be the gaze feature. So these images will be reconstructed based on its gaze feature and the other's eye feature. For the gaze similar pair this is the other way around. 
The paper provides Figure 1 to give an overview of the architecture.

<center>
<img src="[architecture.png](https://github.com/pfrolke/CS4240-DL-Reproducibility/blob/main/blog/imgs/architecture.png?raw=true)" style="width:70%" />
</center>
<p align="center">
  <em>Figure 1: the achitecture of the Cross Encoder as presented in the paper. Here e<sub>J</sub> is the eye feature and g<sub>J</sub> the gaze feature.</em>
</p>

### Loss Function
For one image pair the loss is determined by the following function.

$$
\begin{aligned}
L=\sum||I_i-\hat{I_i}||_1 + ||I_j -\hat{I_j}||_1 + \alpha R
\end{aligned}
$$

where $(I_i, I_j)$ and $(\hat{I_i}, \hat{I_j})$ denote the training images, and the reconstructions respectively. The first two terms are there to compare the reconstructions with the input and the $R = ||(\hat{I_i} - \hat{I_i}) - (\hat{I_j} - \hat{I_j})||_1$ is there to regularize the differences between the losses of the two training images. The parameters of the encoder and decoder are updated using two types of input pairs simultaneously. Therefore the full loss becomes

$$
\begin{aligned}
L_g + \beta L_e
\end{aligned}
$$
where $L_g$ and $L_e$ are the loss for the gaze and eye pair balanced by the hyper parameter $\beta$.

### Gaze estimator
To perform the actual gaze estimation task, the Cross Encoders' encoder component is extracted and extended with two additional linear layers. This network is then trained once more and this time in a supervised manner.

## The Dataset
The Columbia Gaze dataset consists of 5880 images of 56 subjects. It includes 105 images per subject, where each image has different gaze directions and head poses. Additionally, a label for the gaze and landmarks indicating the coordinates of both the left and right eye is available for each image. The label is a 2D vector, specifying the pitch and the yaw.

## Method

### Data processing
To process the data we followed all the steps described in the paper. First, the images are histogram equalized and grey-scaled. Both actions are to eliminate changes in appearance due to variations in the light direction, intensity and color. Then, we used the landmarks to extract the left and the right eye for each image. However, after inspecting the results we noticed something was wrong. To find the issue we plotted the rectangle landmarks on the images and identified a flip in the x-axis of the images. Incorporation of this knowledge, resulted in accurately cropped images. Finally, it was required to have the same width and height for each eye to be able to feed it into the Cross Encoder. To achieve this, we determined the maximum width and height among all the eyes and changed the croppings of each eye accordingly. We ensured that the original cropping was centered relative to the new cropping.

### Dataset creation
After processing the data we created two datasets. One for training the Cross Encoder and one to train the gaze estimator. In the Cross Encoder dataset, one sample contains one eye pair and one gaze pair which boils down to a tuple of length four. For each of the participants in the dataset we created 105 gaze and eye pairs. To form the gaze pairs, we took the left and right eye of the same image, just like the authors did. The eye pairs were constructed by randomly combining two left or two right eyes from different images but from the same person.
The gaze estimator dataset comprises all of the eyes along with their corresponding labels.

### Training procedure

The cross-encoder is trained on an $80/20$ training/test split on the Columbia dataset. In the original paper they used a 5-fold cross-validation on the Columbia dataset but due to time constraints we only do one split. Training is done for 200 epochs with a learning rate of 0.0001 using the Adam optimizer with default hyperparameters, following the original paper. The batch size is set to 16, each training sample consists of one eye pair and the one gaze pair. To reproduce the angular error of $6.4\pm0.1$ we set the gaze vector dimension to 15 and the eye vector dimension to 32, again the same as the original paper. We use the Google Cloud environment with a NVIDIA Tesla T4 GPU to train our model.

After the cross-encoder is trained, the gaze estimator will be trained. The training will run for 90 epochs with a learning rate of 0.01 on 100 shots using the Adam optimizer with default hyperparameters. Thi

* Welke hyper parameters
* Train test split
* (Evt graph van met loss)

## Results

### Angular error on Columbia Gaze
For the assessment of a gaze-estimation technique, typically the angular error is reported. Angular error is a measure of the deviation in degrees between the predicted and target angles of the gaze for a sample.

Our trained Cross-Encoder achieves an angular error of $9.3\pm4.2$ on the test set. This is a significant deviation from the $6.4\pm0.1$ angular error reported by the authors. The error reported by the authors is the mean of 5 split cross-validation training, and we only trained and tested the model on one split. However, the small standard deviation reported by the authors makes it unlikely that this discrepancy is caused by an unfortunate split.

During inspection of the code shared by the authors we noticed an issue in the training procedure they applied. In the file [2_regress_to_vector.py](https://github.com/sunyunjia96/Cross-Encoder/blob/master/2_regress_to_vector.py) the following code snippet can be found:

```python
if error < best:
  torch.save(reg.state_dict(), 'regressor.pth.tar')
  best = error
```

It appears as if the model is only saved for the lowest test error. Because of this code we believe that the authors effectively used the test set as a validation set during training. This is bad practice, as it results in an incorrect view of the model's generalization performance. When we evaluated our model in the same way, it achieved an angular error of $7.9\pm3.8$ on the test set. This is considerably better than the original $9.3\pm4.2$, but still far from the reported $6.4\pm0.1$. 

### Cross-Encoder evaluation
<img src="https://github.com/pfrolke/CS4240-DL-Reproducibility/blob/main/blog/imgs/planning-2.jpg/model_output.png?raw=true" width=200>

The figure above shows a sample of input images from the test set (left) together with their respective output images when reconstructed by the Cross-Encoder (right). The quality of the output images is an indicator that the encoder-decoder training is successfully mapping an input image to a latent space without losing information about the gaze. Therefore, it likely is not the cause of the performance drop.

- miss voorbeeld van slechte prediction
- plaatje van decoder output

## Discussion / Challenges / problems

One significant challenge was that the available code was, in our opinion, convoluted and badly annotated. This made it hard for us to interpret and reproduce the paper, especially since we are not familiar with how Deep Learning or more specifically Gaze estimation projects are structured. With the help of our supervisor did the code become understandable, even though he had his own remarks about the original code.

When looking through the original code we found that the output of the encoder was not the combined size of the gaze and eye feature vector. They multiply the gaze feature by three, which thus results in an unfair representation of gaze feature. In the paper they do not mention this and it is not clear from the code why this is done. Thus for our own implementation we decided not to multiply the gaze dimension by three. This might result our worse performance or explain why their results were so good.

One other explanation why our results are not the same is that we did not perform a 5-fold cross validation. The Columbia dataset is a relatively small dataset, thus using cross-validation one could better utilize the short comings of using a small dataset.


## Conclusion


## Project approach + division of work

At the beginning we made a project plan for the entire course of the project. We decided that we would try to pair program as much as possible during the project since it was hard to divide the task of programming. The planning can be found in the image below.

![Project plan](https://github.com/pfrolke/CS4240-DL-Reproducibility/blob/main/blog/imgs/planning-2.jpg?raw=true)

We ended up having weekly meetings with our supervisor Lingyu. During these meetings we discussed our progress and asked for help when things were unclear. Unfortunately, we ended up meeting Alex, our TA, only once. This was mostly due to miscommunication, as well as that our group contact with Lingyu went through Teams. Alex was in this Teams channel but never replied.

During the project we found out that it was possible to separate the programming tasks a bit more. Gijs mainly worked on implementing the loading and processing the data as well as helping to debug the cross encoder and loss implementations.

When we began writing the report we made a division on who would write which parts of the project. Everyone wrote some sections individually and we then later proofread each others sections to make sure there were no mistakes and that it was a coherent blog.

* Planning die we de eerste week hadden gemaakt?
* Ik zag in een van de voorbeelden een tabel met taken en wie er aan hadden gewerkt. Dat kunnen we ook doen.
