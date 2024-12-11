<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#sfm>SFM</a></li>
      <ul>
        <li><a href=#Deep-Non-rigid-Structure-from-Motion-Revisited:-Canonicalization-and-Sequence-Modeling>Deep Non-rigid Structure-from-Motion Revisited: Canonicalization and Sequence Modeling</a></li>
      </ul>
    </li>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#On-Motion-Blur-and-Deblurring-in-Visual-Place-Recognition>On Motion Blur and Deblurring in Visual Place Recognition</a></li>
        <li><a href=#Image-Retrieval-with-Intra-Sweep-Representation-Learning-for-Neck-Ultrasound-Scanning-Guidance>Image Retrieval with Intra-Sweep Representation Learning for Neck Ultrasound Scanning Guidance</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#EventSplat:-3D-Gaussian-Splatting-from-Moving-Event-Cameras-for-Real-time-Rendering>EventSplat: 3D Gaussian Splatting from Moving Event Cameras for Real-time Rendering</a></li>
        <li><a href=#Diffusing-Differentiable-Representations>Diffusing Differentiable Representations</a></li>
      </ul>
    </li>
  </ol>
</details>

## SFM  

### [Deep Non-rigid Structure-from-Motion Revisited: Canonicalization and Sequence Modeling](http://arxiv.org/abs/2412.07230)  
Hui Deng, Jiawei Shi, Zhen Qin, Yiran Zhong, Yuchao Dai  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Non-Rigid Structure-from-Motion (NRSfM) is a classic 3D vision problem, where a 2D sequence is taken as input to estimate the corresponding 3D sequence. Recently, the deep neural networks have greatly advanced the task of NRSfM. However, existing deep NRSfM methods still have limitations in handling the inherent sequence property and motion ambiguity associated with the NRSfM problem. In this paper, we revisit deep NRSfM from two perspectives to address the limitations of current deep NRSfM methods : (1) canonicalization and (2) sequence modeling. We propose an easy-to-implement per-sequence canonicalization method as opposed to the previous per-dataset canonicalization approaches. With this in mind, we propose a sequence modeling method that combines temporal information and subspace constraint. As a result, we have achieved a more optimal NRSfM reconstruction pipeline compared to previous efforts. The effectiveness of our method is verified by testing the sequence-to-sequence deep NRSfM pipeline with corresponding regularization modules on several commonly used datasets.  
  </ol>  
</details>  
**comments**: 9 pages main text, 7 pages appendix  
  
  



## Visual Localization  

### [On Motion Blur and Deblurring in Visual Place Recognition](http://arxiv.org/abs/2412.07751)  
Timur Ismagilov, Bruno Ferrarini, Michael Milford, Tan Viet Tuyen Nguyen, SD Ramchurn, Shoaib Ehsan  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Place Recognition (VPR) in mobile robotics enables robots to localize themselves by recognizing previously visited locations using visual data. While the reliability of VPR methods has been extensively studied under conditions such as changes in illumination, season, weather and viewpoint, the impact of motion blur is relatively unexplored despite its relevance not only in rapid motion scenarios but also in low-light conditions where longer exposure times are necessary. Similarly, the role of image deblurring in enhancing VPR performance under motion blur has received limited attention so far. This paper bridges these gaps by introducing a new benchmark designed to evaluate VPR performance under the influence of motion blur and image deblurring. The benchmark includes three datasets that encompass a wide range of motion blur intensities, providing a comprehensive platform for analysis. Experimental results with several well-established VPR and image deblurring methods provide new insights into the effects of motion blur and the potential improvements achieved through deblurring. Building on these findings, the paper proposes adaptive deblurring strategies for VPR, designed to effectively manage motion blur in dynamic, real-world scenarios.  
  </ol>  
</details>  
  
### [Image Retrieval with Intra-Sweep Representation Learning for Neck Ultrasound Scanning Guidance](http://arxiv.org/abs/2412.07741)  
Wanwen Chen, Adam Schmidt, Eitan Prisman, Septimiu E. Salcudean  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Purpose: Intraoperative ultrasound (US) can enhance real-time visualization in transoral robotic surgery. The surgeon creates a mental map with a pre-operative scan. Then, a surgical assistant performs freehand US scanning during the surgery while the surgeon operates at the remote surgical console. Communicating the target scanning plane in the surgeon's mental map is difficult. Automatic image retrieval can help match intraoperative images to preoperative scans, guiding the assistant to adjust the US probe toward the target plane. Methods: We propose a self-supervised contrastive learning approach to match intraoperative US views to a preoperative image database. We introduce a novel contrastive learning strategy that leverages intra-sweep similarity and US probe location to improve feature encoding. Additionally, our model incorporates a flexible threshold to reject unsatisfactory matches. Results: Our method achieves 92.30% retrieval accuracy on simulated data and outperforms state-of-the-art temporal-based contrastive learning approaches. Our ablation study demonstrates that using probe location in the optimization goal improves image representation, suggesting that semantic information can be extracted from probe location. We also present our approach on real patient data to show the feasibility of the proposed US probe localization system despite tissue deformation from tongue retraction. Conclusion: Our contrastive learning method, which utilizes intra-sweep similarity and US probe location, enhances US image representation learning. We also demonstrate the feasibility of using our image retrieval method to provide neck US localization on real patient US after tongue retraction.  
  </ol>  
</details>  
**comments**: 12 pages, 5 figures  
  
  



## NeRF  

### [EventSplat: 3D Gaussian Splatting from Moving Event Cameras for Real-time Rendering](http://arxiv.org/abs/2412.07293)  
Toshiya Yura, Ashkan Mirzaei, Igor Gilitschenski  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce a method for using event camera data in novel view synthesis via Gaussian Splatting. Event cameras offer exceptional temporal resolution and a high dynamic range. Leveraging these capabilities allows us to effectively address the novel view synthesis challenge in the presence of fast camera motion. For initialization of the optimization process, our approach uses prior knowledge encoded in an event-to-video model. We also use spline interpolation for obtaining high quality poses along the event camera trajectory. This enhances the reconstruction quality from fast-moving cameras while overcoming the computational limitations traditionally associated with event-based Neural Radiance Field (NeRF) methods. Our experimental evaluation demonstrates that our results achieve higher visual fidelity and better performance than existing event-based NeRF approaches while being an order of magnitude faster to render.  
  </ol>  
</details>  
  
### [Diffusing Differentiable Representations](http://arxiv.org/abs/2412.06981)  
Yash Savani, Marc Finzi, J. Zico Kolter  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce a novel, training-free method for sampling differentiable representations (diffreps) using pretrained diffusion models. Rather than merely mode-seeking, our method achieves sampling by "pulling back" the dynamics of the reverse-time process--from the image space to the diffrep parameter space--and updating the parameters according to this pulled-back process. We identify an implicit constraint on the samples induced by the diffrep and demonstrate that addressing this constraint significantly improves the consistency and detail of the generated objects. Our method yields diffreps with substantially improved quality and diversity for images, panoramas, and 3D NeRFs compared to existing techniques. Our approach is a general-purpose method for sampling diffreps, expanding the scope of problems that diffusion models can tackle.  
  </ol>  
</details>  
**comments**: Published at NeurIPS 2024  
  
  



