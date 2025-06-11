<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Robust-Visual-Localization-via-Semantic-Guided-Multi-Scale-Transformer>Robust Visual Localization via Semantic-Guided Multi-Scale Transformer</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#A-Probability-guided-Sampler-for-Neural-Implicit-Surface-Rendering>A Probability-guided Sampler for Neural Implicit Surface Rendering</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Robust Visual Localization via Semantic-Guided Multi-Scale Transformer](http://arxiv.org/abs/2506.08526)  
Zhongtao Tian, Wenhao Huang, Zhidong Chen, Xiao Wei Sun  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Visual localization remains challenging in dynamic environments where fluctuating lighting, adverse weather, and moving objects disrupt appearance cues. Despite advances in feature representation, current absolute pose regression methods struggle to maintain consistency under varying conditions. To address this challenge, we propose a framework that synergistically combines multi-scale feature learning with semantic scene understanding. Our approach employs a hierarchical Transformer with cross-scale attention to fuse geometric details and contextual cues, preserving spatial precision while adapting to environmental changes. We improve the performance of this architecture with semantic supervision via neural scene representation during training, guiding the network to learn view-invariant features that encode persistent structural information while suppressing complex environmental interference. Experiments on TartanAir demonstrate that our approach outperforms existing pose regression methods in challenging scenarios with dynamic objects, illumination changes, and occlusions. Our findings show that integrating multi-scale processing with semantic guidance offers a promising strategy for robust visual localization in real-world dynamic environments.  
  </ol>  
</details>  
  
  



## NeRF  

### [A Probability-guided Sampler for Neural Implicit Surface Rendering](http://arxiv.org/abs/2506.08619)  
Gonçalo Dias Pais, Valter Piedade, Moitreya Chatterjee, Marcus Greiff, Pedro Miraldo  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Several variants of Neural Radiance Fields (NeRFs) have significantly improved the accuracy of synthesized images and surface reconstruction of 3D scenes/objects. In all of these methods, a key characteristic is that none can train the neural network with every possible input data, specifically, every pixel and potential 3D point along the projection rays due to scalability issues. While vanilla NeRFs uniformly sample both the image pixels and 3D points along the projection rays, some variants focus only on guiding the sampling of the 3D points along the projection rays. In this paper, we leverage the implicit surface representation of the foreground scene and model a probability density function in a 3D image projection space to achieve a more targeted sampling of the rays toward regions of interest, resulting in improved rendering. Additionally, a new surface reconstruction loss is proposed for improved performance. This new loss fully explores the proposed 3D image projection space model and incorporates near-to-surface and empty space components. By integrating our novel sampling strategy and novel loss into current state-of-the-art neural implicit surface renderers, we achieve more accurate and detailed 3D reconstructions and improved image rendering, especially for the regions of interest in any given scene.  
  </ol>  
</details>  
**comments**: Accepted in ECCV 2024  
  
  



