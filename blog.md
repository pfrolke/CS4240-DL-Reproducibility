Group 54 Reproducibility Blog Post
===

## Students

Isa Rethans 4965809
Paul Frolke 4888693
Gijs Admiraal 4781669

## Table of contents

[TOC]

## Sources

Data
<https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild>
<https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/>

GitHub
<https://github.com/sunyunjia96/Cross-Encoder>

Paper
<https://openaccess.thecvf.com/content/ICCV2021/papers/Sun_Cross-Encoder_for_Unsupervised_Gaze_Representation_Learning_ICCV_2021_paper.pdf>

## Introduction

Gaze estimation is a rapidly growing area of research that involves predicting where a person is looking based on the orientation of their eyes. A persons gaze can help understand the persons desires, intent and state of mind. This technology has important applications in fields such as human-computer interaction, virtual reality, autonomous driving and cognitive neuroscience.

Accurate gaze estimation can enable more natural and intuitive interfaces for human-computer interaction, such as eye-controlled devices, and can also provide insights into how people perceive and process visual information. Moreover, gaze estimation can be used in autonomous driving to detect driver distractions.

## Problem statement
In this work, we are trying to reproduce the results of the paper "Cross-Encoder for Unsupervised Gaze Representation Learning" by Yunjia Sun, et al. The paper introduces an auto-encoder-like framework, called a Cross-Encoder, for unsupervised gaze representation learning. This framework learns to disentangle gaze and eye features from eye images.