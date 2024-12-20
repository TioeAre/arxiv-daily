<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#MegaPairs:-Massive-Data-Synthesis-For-Universal-Multimodal-Retrieval>MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#keypoint-detection>Keypoint Detection</a></li>
      <ul>
        <li><a href=#Corn-Ear-Detection-and-Orientation-Estimation-Using-Deep-Learning>Corn Ear Detection and Orientation Estimation Using Deep Learning</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#GSRender:-Deduplicated-Occupancy-Prediction-via-Weakly-Supervised-3D-Gaussian-Splatting>GSRender: Deduplicated Occupancy Prediction via Weakly Supervised 3D Gaussian Splatting</a></li>
        <li><a href=#Bright-NeRF:Brightening-Neural-Radiance-Field-with-Color-Restoration-from-Low-light-Raw-Images>Bright-NeRF:Brightening Neural Radiance Field with Color Restoration from Low-light Raw Images</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval](http://arxiv.org/abs/2412.14475)  
Junjie Zhou, Zheng Liu, Ze Liu, Shitao Xiao, Yueze Wang, Bo Zhao, Chen Jason Zhang, Defu Lian, Yongping Xiong  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Despite the rapidly growing demand for multimodal retrieval, progress in this field remains severely constrained by a lack of training data. In this paper, we introduce MegaPairs, a novel data synthesis method that leverages vision language models (VLMs) and open-domain images, together with a massive synthetic dataset generated from this method. Our empirical analysis shows that MegaPairs generates high-quality data, enabling the multimodal retriever to significantly outperform the baseline model trained on 70 $\times$ more data from existing datasets. Moreover, since MegaPairs solely relies on general image corpora and open-source VLMs, it can be easily scaled up, enabling continuous improvements in retrieval performance. In this stage, we produced more than 26 million training instances and trained several models of varying sizes using this data. These new models achieve state-of-the-art zero-shot performance across 4 popular composed image retrieval (CIR) benchmarks and the highest overall performance on the 36 datasets provided by MMEB. They also demonstrate notable performance improvements with additional downstream fine-tuning. Our produced dataset, well-trained models, and data synthesis pipeline will be made publicly available to facilitate the future development of this field.  
  </ol>  
</details>  
  
  



## Keypoint Detection  

### [Corn Ear Detection and Orientation Estimation Using Deep Learning](http://arxiv.org/abs/2412.14954)  
Nathan Sprague, John Evans, Michael Mardikes  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Monitoring growth behavior of maize plants such as the development of ears can give key insights into the plant's health and development. Traditionally, the measurement of the angle of ears is performed manually, which can be time-consuming and prone to human error. To address these challenges, this paper presents a computer vision-based system for detecting and tracking ears of corn in an image sequence. The proposed system could accurately detect, track, and predict the ear's orientation, which can be useful in monitoring their growth behavior. This can significantly save time compared to manual measurement and enables additional areas of ear orientation research and potential increase in efficiencies for maize production. Using an object detector with keypoint detection, the algorithm proposed could detect 90 percent of all ears. The cardinal estimation had a mean absolute error (MAE) of 18 degrees, compared to a mean 15 degree difference between two people measuring by hand. These results demonstrate the feasibility of using computer vision techniques for monitoring maize growth and can lead to further research in this area.  
  </ol>  
</details>  
**comments**: 22 pages;15 figures  
  
  



## NeRF  

### [GSRender: Deduplicated Occupancy Prediction via Weakly Supervised 3D Gaussian Splatting](http://arxiv.org/abs/2412.14579)  
Qianpu Sun, Changyong Shu, Sifan Zhou, Zichen Yu, Yan Chen, Dawei Yang, Yuan Chun  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    3D occupancy perception is gaining increasing attention due to its capability to offer detailed and precise environment representations. Previous weakly-supervised NeRF methods balance efficiency and accuracy, with mIoU varying by 5-10 points due to sampling count along camera rays. Recently, real-time Gaussian splatting has gained widespread popularity in 3D reconstruction, and the occupancy prediction task can also be viewed as a reconstruction task. Consequently, we propose GSRender, which naturally employs 3D Gaussian Splatting for occupancy prediction, simplifying the sampling process. In addition, the limitations of 2D supervision result in duplicate predictions along the same camera ray. We implemented the Ray Compensation (RC) module, which mitigates this issue by compensating for features from adjacent frames. Finally, we redesigned the loss to eliminate the impact of dynamic objects from adjacent frames. Extensive experiments demonstrate that our approach achieves SOTA (state-of-the-art) results in RayIoU (+6.0), while narrowing the gap with 3D supervision methods. Our code will be released soon.  
  </ol>  
</details>  
  
### [Bright-NeRF:Brightening Neural Radiance Field with Color Restoration from Low-light Raw Images](http://arxiv.org/abs/2412.14547)  
Min Wang, Xin Huang, Guoqing Zhou, Qifeng Guo, Qing Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRFs) have demonstrated prominent performance in novel view synthesis. However, their input heavily relies on image acquisition under normal light conditions, making it challenging to learn accurate scene representation in low-light environments where images typically exhibit significant noise and severe color distortion. To address these challenges, we propose a novel approach, Bright-NeRF, which learns enhanced and high-quality radiance fields from multi-view low-light raw images in an unsupervised manner. Our method simultaneously achieves color restoration, denoising, and enhanced novel view synthesis. Specifically, we leverage a physically-inspired model of the sensor's response to illumination and introduce a chromatic adaptation loss to constrain the learning of response, enabling consistent color perception of objects regardless of lighting conditions. We further utilize the raw data's properties to expose the scene's intensity automatically. Additionally, we have collected a multi-view low-light raw image dataset to advance research in this field. Experimental results demonstrate that our proposed method significantly outperforms existing 2D and 3D approaches. Our code and dataset will be made publicly available.  
  </ol>  
</details>  
**comments**: Accepted by AAAI2025  
  
  



