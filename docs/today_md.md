<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Normal-NeRF:-Ambiguity-Robust-Normal-Estimation-for-Highly-Reflective-Scenes>Normal-NeRF: Ambiguity-Robust Normal Estimation for Highly Reflective Scenes</a></li>
      </ul>
    </li>
  </ol>
</details>

## NeRF  

### [Normal-NeRF: Ambiguity-Robust Normal Estimation for Highly Reflective Scenes](http://arxiv.org/abs/2501.09460)  
[[code](https://github.com/sjj118/normal-nerf)]  
Ji Shi, Xianghua Ying, Ruohao Guo, Bowei Xing, Wenzhen Yue  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) often struggle with reconstructing and rendering highly reflective scenes. Recent advancements have developed various reflection-aware appearance models to enhance NeRF's capability to render specular reflections. However, the robust reconstruction of highly reflective scenes is still hindered by the inherent shape ambiguity on specular surfaces. Existing methods typically rely on additional geometry priors to regularize the shape prediction, but this can lead to oversmoothed geometry in complex scenes. Observing the critical role of surface normals in parameterizing reflections, we introduce a transmittance-gradient-based normal estimation technique that remains robust even under ambiguous shape conditions. Furthermore, we propose a dual activated densities module that effectively bridges the gap between smooth surface normals and sharp object boundaries. Combined with a reflection-aware appearance model, our proposed method achieves robust reconstruction and high-fidelity rendering of scenes featuring both highly specular reflections and intricate geometric structures. Extensive experiments demonstrate that our method outperforms existing state-of-the-art methods on various datasets.  
  </ol>  
</details>  
**comments**: AAAI 2025, code available at https://github.com/sjj118/Normal-NeRF  
  
  



