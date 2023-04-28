# Group 54 Reproducibility Blog Post 
Isa Rethans, 4965809, i.g.m.rethans@student.tudelft.nl \
Paul Frolke, 4888693, p.r.frolkee@student.tudelft.nl \
Gijs Admiraal, 4781669, g.j.admiraal@student.tudelft.nl

## Table of contents

- [Table of contents](#table-of-contents)
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
- [Project approach + division of work](#project-approach--division-of-work)



## Sources

**Original Paper**
* Paper: <https://openaccess.thecvf.com/content/ICCV2021/papers/Sun_Cross-Encoder_for_Unsupervised_Gaze_Representation_Learning_ICCV_2021_paper.pdf>
* Data: <https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild>
<https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/>
* GitHub: <https://github.com/sunyunjia96/Cross-Encoder>

**Our Reproduction**
* GitHub: <https://github.com/pfrolke/CS4240-DL-Reproducibility>


## Introduction

Gaze estimation is a rapidly growing area of research that involves predicting where a person is looking based on the orientation of their eyes. A persons gaze can help understand the persons desires, intent and state of mind. This technology has important applications in fields such as human-computer interaction, virtual reality, autonomous driving and cognitive neuroscience. 

Accurate gaze estimation can enable more natural and intuitive interfaces for human-computer interaction, such as eye-controlled devices, and can also provide insights into how people perceive and process visual information. Moreover, gaze estimation can be used in autonomous driving to detect driver distractions.

## Problem statement
In this work, we are trying to reproduce the results of the paper "Cross-Encoder for Unsupervised Gaze Representation Learning" by Yunjia Sun, et al. The paper introduces an auto-encoder-like framework, called a Cross-Encoder, for unsupervised gaze representation learning. The authors highlight that learning a good representation for gaze is non-trivial due to the fact that features of gaze direction features are always interwined with features of the eye's physical appearance. With that in mind, the authors designed a framework that learns to disentangle the gaze and eye features. In the paper, the performance of the Cross-Encoder is evaluated using experiments on various public datasets and with various values for the hyperparemeters $d_g$ and $d_e$ representing the the dimension of the gaze and eye and the feature respectively. Our tasks was to reproduce one specific number: the angular error on the Columbia Gaze dataset with $d_g=15$ and $d_e=32$ which is listed in Table 1 of the paper.

## Cross Encoder Architecture
The Cross-Encoder is a modification of a conventional auto-encoder, a feedforward network that tries to reproduce its input at the output layer. To incorporate the goal of separating the gaze and eye features, Cross Encoder processes paired images together. The Cross-Encoder then encodes each image into two features, known as the shared feature and the specific feature. Both images are reconstructed according to their respective features, 
<!-- Todo -->

## The Dataset
The Columbia Gaze dataset consists of 5880 images from 56 subjects. This means 105 images per subject, where each image has different gaze directions and head poses. Additionally, a label for the gaze and landmarks indicating the coordinates of both the left and right eye is available for each image. The label is a 2D vector, specifying the pitch and the yaw.

## Method

### Data processing
To process the data we followed all the steps discribed in the paper. First, the images are histogram equalised and grey scaled. Both actions are to eliminate changes in appearance due to variations in the light direction, intensity and colour. Then, we used the landmarks to extract the left and the right eye for each image. However, after inspecting the results we noticed something was wrong. To find the issue we plotted the rectangle landmarks on the images and identified a flip in the x-axis of the images. Incorporation of this knowledge, resulted in accurately cropped images. Finally, it was required to have the same width and height for each eye to be able to feed it into the Cross Encoder. To achieve this, we determined the maximum width and height among all the eyes and changed the croppings of each eye accordingly. We ensured that the original cropping was centered relative to the new cropping.

### Dataset creation

## Approach
- Eerst supervised
- Toen cross encoder
- Toen trainen

## Training procedure
- Welke hyper parameters
- Train test split
- (Evt graph van met loss)

## Results

## Discussion / Challenges / problems

## Conclusion

## Project approach + division of work
- Planning die we de eerste week hadden gemaakt?
- Ik zag in een van de voorbeelden een tabel met taken en wie er aan hadden gewerkt. Dat kunnen we ook doen. 