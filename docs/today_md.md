<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#FFF:-Fixing-Flawed-Foundations-in-contrastive-pre-training-results-in-very-strong-Vision-Language-models>FFF: Fixing Flawed Foundations in contrastive pre-training results in very strong Vision-Language models</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#When-LLMs-step-into-the-3D-World:-A-Survey-and-Meta-Analysis-of-3D-Tasks-via-Multi-modal-Large-Language-Models>When LLMs step into the 3D World: A Survey and Meta-Analysis of 3D Tasks via Multi-modal Large Language Models</a></li>
        <li><a href=#From-NeRFs-to-Gaussian-Splats,-and-Back>From NeRFs to Gaussian Splats, and Back</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [FFF: Fixing Flawed Foundations in contrastive pre-training results in very strong Vision-Language models](http://arxiv.org/abs/2405.10286)  
Adrian Bulat, Yassine Ouali, Georgios Tzimiropoulos  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Despite noise and caption quality having been acknowledged as important factors impacting vision-language contrastive pre-training, in this paper, we show that the full potential of improving the training process by addressing such issues is yet to be realized. Specifically, we firstly study and analyze two issues affecting training: incorrect assignment of negative pairs, and low caption quality and diversity. Then, we devise effective solutions for addressing both problems, which essentially require training with multiple true positive pairs. Finally, we propose training with sigmoid loss to address such a requirement. We show very large gains over the current state-of-the-art for both image recognition ( $\sim +6\%$ on average over 11 datasets) and image retrieval ($\sim +19\%$ on Flickr30k and $\sim +15\%$ on MSCOCO).  
  </ol>  
</details>  
**comments**: Accepted at CVPR 2024  
  
  



## NeRF  

### [When LLMs step into the 3D World: A Survey and Meta-Analysis of 3D Tasks via Multi-modal Large Language Models](http://arxiv.org/abs/2405.10255)  
Xianzheng Ma, Yash Bhalgat, Brandon Smart, Shuai Chen, Xinghui Li, Jian Ding, Jindong Gu, Dave Zhenyu Chen, Songyou Peng, Jia-Wang Bian, Philip H Torr, Marc Pollefeys, Matthias Nießner, Ian D Reid, Angel X. Chang, Iro Laina, Victor Adrian Prisacariu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    As large language models (LLMs) evolve, their integration with 3D spatial data (3D-LLMs) has seen rapid progress, offering unprecedented capabilities for understanding and interacting with physical spaces. This survey provides a comprehensive overview of the methodologies enabling LLMs to process, understand, and generate 3D data. Highlighting the unique advantages of LLMs, such as in-context learning, step-by-step reasoning, open-vocabulary capabilities, and extensive world knowledge, we underscore their potential to significantly advance spatial comprehension and interaction within embodied Artificial Intelligence (AI) systems. Our investigation spans various 3D data representations, from point clouds to Neural Radiance Fields (NeRFs). It examines their integration with LLMs for tasks such as 3D scene understanding, captioning, question-answering, and dialogue, as well as LLM-based agents for spatial reasoning, planning, and navigation. The paper also includes a brief review of other methods that integrate 3D and language. The meta-analysis presented in this paper reveals significant progress yet underscores the necessity for novel approaches to harness the full potential of 3D-LLMs. Hence, with this paper, we aim to chart a course for future research that explores and expands the capabilities of 3D-LLMs in understanding and interacting with the complex 3D world. To support this survey, we have established a project page where papers related to our topic are organized and listed: https://github.com/ActiveVisionLab/Awesome-LLM-3D.  
  </ol>  
</details>  
  
### [From NeRFs to Gaussian Splats, and Back](http://arxiv.org/abs/2405.09717)  
[[code](https://github.com/grasp-lyrl/nerftogsandback)]  
Siming He, Zach Osman, Pratik Chaudhari  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    For robotics applications where there is a limited number of (typically ego-centric) views, parametric representations such as neural radiance fields (NeRFs) generalize better than non-parametric ones such as Gaussian splatting (GS) to views that are very different from those in the training data; GS however can render much faster than NeRFs. We develop a procedure to convert back and forth between the two. Our approach achieves the best of both NeRFs (superior PSNR, SSIM, and LPIPS on dissimilar views, and a compact representation) and GS (real-time rendering and ability for easily modifying the representation); the computational cost of these conversions is minor compared to training the two from scratch.  
  </ol>  
</details>  
  
  



