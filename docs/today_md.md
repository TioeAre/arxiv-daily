<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#To-Match-or-Not-to-Match:-Revisiting-Image-Matching-for-Reliable-Visual-Place-Recognition>To Match or Not to Match: Revisiting Image Matching for Reliable Visual Place Recognition</a></li>
      </ul>
    </li>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#To-Match-or-Not-to-Match:-Revisiting-Image-Matching-for-Reliable-Visual-Place-Recognition>To Match or Not to Match: Revisiting Image Matching for Reliable Visual Place Recognition</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Meta-Continual-Learning-of-Neural-Fields>Meta-Continual Learning of Neural Fields</a></li>
        <li><a href=#SE4Lip:-Speech-Lip-Encoder-for-Talking-Head-Synthesis-to-Solve-Phoneme-Viseme-Alignment-Ambiguity>SE4Lip: Speech-Lip Encoder for Talking Head Synthesis to Solve Phoneme-Viseme Alignment Ambiguity</a></li>
        <li><a href=#InvNeRF-Seg:-Fine-Tuning-a-Pre-Trained-NeRF-for-3D-Object-Segmentation>InvNeRF-Seg: Fine-Tuning a Pre-Trained NeRF for 3D Object Segmentation</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [To Match or Not to Match: Revisiting Image Matching for Reliable Visual Place Recognition](http://arxiv.org/abs/2504.06116)  
Davide Sferrazza, Gabriele Berton, Gabriele Trivigno, Carlo Masone  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Place Recognition (VPR) is a critical task in computer vision, traditionally enhanced by re-ranking retrieval results with image matching. However, recent advancements in VPR methods have significantly improved performance, challenging the necessity of re-ranking. In this work, we show that modern retrieval systems often reach a point where re-ranking can degrade results, as current VPR datasets are largely saturated. We propose using image matching as a verification step to assess retrieval confidence, demonstrating that inlier counts can reliably predict when re-ranking is beneficial. Our findings shift the paradigm of retrieval pipelines, offering insights for more robust and adaptive VPR systems.  
  </ol>  
</details>  
**comments**: CVPRW 2025  
  
  



## Image Matching  

### [To Match or Not to Match: Revisiting Image Matching for Reliable Visual Place Recognition](http://arxiv.org/abs/2504.06116)  
Davide Sferrazza, Gabriele Berton, Gabriele Trivigno, Carlo Masone  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual Place Recognition (VPR) is a critical task in computer vision, traditionally enhanced by re-ranking retrieval results with image matching. However, recent advancements in VPR methods have significantly improved performance, challenging the necessity of re-ranking. In this work, we show that modern retrieval systems often reach a point where re-ranking can degrade results, as current VPR datasets are largely saturated. We propose using image matching as a verification step to assess retrieval confidence, demonstrating that inlier counts can reliably predict when re-ranking is beneficial. Our findings shift the paradigm of retrieval pipelines, offering insights for more robust and adaptive VPR systems.  
  </ol>  
</details>  
**comments**: CVPRW 2025  
  
  



## NeRF  

### [Meta-Continual Learning of Neural Fields](http://arxiv.org/abs/2504.05806)  
Seungyoon Woo, Junhyeog Yun, Gunhee Kim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Fields (NF) have gained prominence as a versatile framework for complex data representation. This work unveils a new problem setting termed \emph{Meta-Continual Learning of Neural Fields} (MCL-NF) and introduces a novel strategy that employs a modular architecture combined with optimization-based meta-learning. Focused on overcoming the limitations of existing methods for continual learning of neural fields, such as catastrophic forgetting and slow convergence, our strategy achieves high-quality reconstruction with significantly improved learning speed. We further introduce Fisher Information Maximization loss for neural radiance fields (FIM-NeRF), which maximizes information gains at the sample level to enhance learning generalization, with proved convergence guarantee and generalization bound. We perform extensive evaluations across image, audio, video reconstruction, and view synthesis tasks on six diverse datasets, demonstrating our method's superiority in reconstruction quality and speed over existing MCL and CL-NF approaches. Notably, our approach attains rapid adaptation of neural fields for city-scale NeRF rendering with reduced parameter requirement.  
  </ol>  
</details>  
  
### [SE4Lip: Speech-Lip Encoder for Talking Head Synthesis to Solve Phoneme-Viseme Alignment Ambiguity](http://arxiv.org/abs/2504.05803)  
Yihuan Huang, Jiajun Liu, Yanzhen Ren, Wuyang Liu, Juhua Tang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Speech-driven talking head synthesis tasks commonly use general acoustic features (such as HuBERT and DeepSpeech) as guided speech features. However, we discovered that these features suffer from phoneme-viseme alignment ambiguity, which refers to the uncertainty and imprecision in matching phonemes (speech) with visemes (lip). To address this issue, we propose the Speech Encoder for Lip (SE4Lip) to encode lip features from speech directly, aligning speech and lip features in the joint embedding space by a cross-modal alignment framework. The STFT spectrogram with the GRU-based model is designed in SE4Lip to preserve the fine-grained speech features. Experimental results show that SE4Lip achieves state-of-the-art performance in both NeRF and 3DGS rendering models. Its lip sync accuracy improves by 13.7% and 14.2% compared to the best baseline and produces results close to the ground truth videos.  
  </ol>  
</details>  
  
### [InvNeRF-Seg: Fine-Tuning a Pre-Trained NeRF for 3D Object Segmentation](http://arxiv.org/abs/2504.05751)  
Jiangsan Zhao, Jakob Geipel, Krzysztof Kusnierek, Xuean Cui  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) have been widely adopted for reconstructing high quality 3D point clouds from 2D RGB images. However, the segmentation of these reconstructed 3D scenes is more essential for downstream tasks such as object counting, size estimation, and scene understanding. While segmentation on raw 3D point clouds using deep learning requires labor intensive and time-consuming manual annotation, directly training NeRF on binary masks also fails due to the absence of color and shading cues essential for geometry learning. We propose Invariant NeRF for Segmentation (InvNeRFSeg), a two step, zero change fine tuning strategy for 3D segmentation. We first train a standard NeRF on RGB images and then fine tune it using 2D segmentation masks without altering either the model architecture or loss function. This approach produces higher quality, cleaner segmented point clouds directly from the refined radiance field with minimal computational overhead or complexity. Field density analysis reveals consistent semantic refinement: densities of object regions increase while background densities are suppressed, ensuring clean and interpretable segmentations. We demonstrate InvNeRFSegs superior performance over both SA3D and FruitNeRF on both synthetic fruit and real world soybean datasets. This approach effectively extends 2D segmentation to high quality 3D segmentation.  
  </ol>  
</details>  
  
  



