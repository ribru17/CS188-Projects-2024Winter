---
layout: post
comments: true
title: Deep Learning for Prostate Segmentation
authors: Pratosh Menon, Bulent Yesilyurt, Hayden D'Souza, Riley Bruins
date: 2024-03-22
---

> We aim to analyze how we can use deep learning technique for prostate image
> segmentation. Prostate cancer is the second most common form of cancer for men
> worldwide and the fifth leading cause of death for men globally. However, this
> is a statistic that can be considerably changed with early stage detection. In
> fact, the cancer is completely curable within 5 years if we catch it early. To
> this end, we explore how we can use existing deep learning architectures to
> help with prostate image segmentation to catch early prostate cancer in
> patients.

<!--more-->
{: class="table-of-content"}
{:toc}

## History

Classical approaches to 3d image generation of inner tissues and organs involved
manual delineation/contouring which were incredibly time-consuming, expensive
and had rigid calculations for irregular shapes which led to inaccurate
measurements. Traditional statistical models such as K-Means, SVMs and Random
Forest were just as inaccurate since they depended on handcrafted features and
required significantly more preprocessing.

## Enter Deep Learning

We can think of prostate segmentation as classifying voxels as either part or
not part of a tumor. We then use prior training examples to understand how model
boundaries of prostate in a "noisy advantage". This has a dual advantage of
performing image classification and segmentation simultaneously, reducing
overhead, leading to faster diagnoses. The lowered costs and increased speeds
such methods provide also increases accessibility to prostate cancer detection
methods in emerging economies where medical infrastructure is substandard.

## Models

| Model Name | Trainable Parameters | Non-Trainable Parameters | Size on Disk | Inference Time/Dataset (CPU) | Inference Time/Dataset (GPU) |
|            | -------------------- | ------------------------ | ------------ | ---------------------------- | ---------------------------- |
| ENet       | 362,992              | 8,352                    | 5.8 MB       | 6.17 s                       | 1.07 s                       |
| ERFNet     | 2,056,440            | 0                        | 25.3 MB      | 8.59 s                       | 1.03 s                       |
| UNet       | 5,403,874            | 0                        | 65.0 MB      | 42.02 s                      | 1.57 s                       |

### ENet

### ERFNet

### UNet

<!-- deno-fmt-ignore-start -->
![UNet Architecture]({{ '/assets/images/20/unet_arch.png' | relative_url }})
{: style="width: 900px; max-width: 100%;"}
*Fig 1. UNET Architecture* [1].
<!-- deno-fmt-ignore-end -->

## Conclusion

## References


## History
Classical approaches to 3d image generation of inner tissues and organs involved manual delienation/contouring which were incredibly time-consuming, expensive and had rigid calculations for irregular shapes which led to inaccurate measurements. Traditional statistical models such as K-Means, SVMs and Random Forest were just as inaccurate since they depended on handcrafted features and required significantly more preprocessing.  



## Enter Deep Learning
We can think of prostate segmentation as classifying voxels as either part or not part of a tumor. We then use prior training examples to understand how model boundaries of prostate in a "noisy advantage". This has a dual advantage of performing image classification and segmentation simultaneously, reducing overhead, leading to faster diagnoses. The lowered costs and increased speeds such methods provide also increases accessibility to prostate cancer detection methods in emerging economies where medical infrastructure is substandard. 

## Models

>>>>>>> main
| Model Name | Trainable Parameters | Non-Trainable Parameters | Size on Disk | Inference Time/Dataset (CPU) | Inference Time/Dataset (GPU) |
|:---        |        :---:         |          :---:           |     :---:    |            :----:            |                          ---:|
| ENet       | 362,992              | 8,352                    | 5.8 MB       | 6.17 s                       | 1.07 s                       |
| UNet       | 5,403,874            | 0                        | 65.0 MB      | 42.02 s                      | 1.57 s                       |


### ENet
ENet is a fast and compact Encoder-Decoder network. The vanilla ENet assumes an input size of $$512 \times 512$$. The initial ENet block has a convolution operation ($$3 \times 3$$, stride 2), maxpooling, then a concatenation. The convolution has 13 filters, which produces 16 feature maps after concatenation. Then comes the bottleneck module. Here, convolution is either regular, dilated or full with $$3 \times 3$$
filters, or a $$5\times5$$ convolution into asymetric ones. Then using a skip connection, merge back with element-wise addition. There is also Batch Normalization and PReLU between all convolutions.
The below Figure 1 highlights the overall model architecture.

<!-- ignore -->
![ENet]({{ '/assets/images/team20/ENetArchitecture.png' | relative_url }})
{: style="width: 400px; max-width: 100%; display: block;"}
*Fig 1.ENet: An object Segmentation Method* [2].
<!-- end-->
### UNet

### Comparing Models

### Conclusion 

### Image

Please create a folder with the name of your team id under /assets/images/, put
all your images into the folder and reference the images in your main content.

<!-- deno-fmt-ignore-start -->
You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].
<!-- deno-fmt-ignore-end -->

## Reference

Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object
detection." _Proceedings of the IEEE conference on computer vision and pattern
recognition_. 2016.

--
=======
[1] Zhang, Jeremy. "UNet -- Line by Line Explanation"
_https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5/_.
2019.

---

<!-- vim: set spell: -->

