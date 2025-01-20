<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#FLORA:-Formal-Language-Model-Enables-Robust-Training-free-Zero-shot-Object-Referring-Analysis>FLORA: Formal Language Model Enables Robust Training-free Zero-shot Object Referring Analysis</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Surface-SOS:-Self-Supervised-Object-Segmentation-via-Neural-Surface-Representation>Surface-SOS: Self-Supervised Object Segmentation via Neural Surface Representation</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [FLORA: Formal Language Model Enables Robust Training-free Zero-shot Object Referring Analysis](http://arxiv.org/abs/2501.09887)  
Zhe Chen, Zijing Chen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Object Referring Analysis (ORA), commonly known as referring expression comprehension, requires the identification and localization of specific objects in an image based on natural descriptions. Unlike generic object detection, ORA requires both accurate language understanding and precise visual localization, making it inherently more complex. Although recent pre-trained large visual grounding detectors have achieved significant progress, they heavily rely on extensively labeled data and time-consuming learning. To address these, we introduce a novel, training-free framework for zero-shot ORA, termed FLORA (Formal Language for Object Referring and Analysis). FLORA harnesses the inherent reasoning capabilities of large language models (LLMs) and integrates a formal language model - a logical framework that regulates language within structured, rule-based descriptions - to provide effective zero-shot ORA. More specifically, our formal language model (FLM) enables an effective, logic-driven interpretation of object descriptions without necessitating any training processes. Built upon FLM-regulated LLM outputs, we further devise a Bayesian inference framework and employ appropriate off-the-shelf interpretive models to finalize the reasoning, delivering favorable robustness against LLM hallucinations and compelling ORA performance in a training-free manner. In practice, our FLORA boosts the zero-shot performance of existing pretrained grounding detectors by up to around 45%. Our comprehensive evaluation across different challenging datasets also confirms that FLORA consistently surpasses current state-of-the-art zero-shot methods in both detection and segmentation tasks associated with zero-shot ORA. We believe our probabilistic parsing and reasoning of the LLM outputs elevate the reliability and interpretability of zero-shot ORA. We shall release codes upon publication.  
  </ol>  
</details>  
  
  



## NeRF  

### [Surface-SOS: Self-Supervised Object Segmentation via Neural Surface Representation](http://arxiv.org/abs/2501.09947)  
[[code](https://github.com/zhengxyun/surface-sos)]  
Xiaoyun Zheng, Liwei Liao, Jianbo Jiao, Feng Gao, Ronggang Wang  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Self-supervised Object Segmentation (SOS) aims to segment objects without any annotations. Under conditions of multi-camera inputs, the structural, textural and geometrical consistency among each view can be leveraged to achieve fine-grained object segmentation. To make better use of the above information, we propose Surface representation based Self-supervised Object Segmentation (Surface-SOS), a new framework to segment objects for each view by 3D surface representation from multi-view images of a scene. To model high-quality geometry surfaces for complex scenes, we design a novel scene representation scheme, which decomposes the scene into two complementary neural representation modules respectively with a Signed Distance Function (SDF). Moreover, Surface-SOS is able to refine single-view segmentation with multi-view unlabeled images, by introducing coarse segmentation masks as additional input. To the best of our knowledge, Surface-SOS is the first self-supervised approach that leverages neural surface representation to break the dependence on large amounts of annotated data and strong constraints. These constraints typically involve observing target objects against a static background or relying on temporal supervision in videos. Extensive experiments on standard benchmarks including LLFF, CO3D, BlendedMVS, TUM and several real-world scenes show that Surface-SOS always yields finer object masks than its NeRF-based counterparts and surpasses supervised single-view baselines remarkably. Code is available at: https://github.com/zhengxyun/Surface-SOS.  
  </ol>  
</details>  
**comments**: Accepted by TIP  
  
  



