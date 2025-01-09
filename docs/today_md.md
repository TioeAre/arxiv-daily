<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#NeRFs-are-Mirror-Detectors:-Using-Structural-Similarity-for-Multi-View-Mirror-Scene-Reconstruction-with-3D-Surface-Primitives>NeRFs are Mirror Detectors: Using Structural Similarity for Multi-View Mirror Scene Reconstruction with 3D Surface Primitives</a></li>
      </ul>
    </li>
  </ol>
</details>

## NeRF  

### [NeRFs are Mirror Detectors: Using Structural Similarity for Multi-View Mirror Scene Reconstruction with 3D Surface Primitives](http://arxiv.org/abs/2501.04074)  
Leif Van Holland, Michael Weinmann, Jan U. Müller, Patrick Stotko, Reinhard Klein  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    While neural radiance fields (NeRF) led to a breakthrough in photorealistic novel view synthesis, handling mirroring surfaces still denotes a particular challenge as they introduce severe inconsistencies in the scene representation. Previous attempts either focus on reconstructing single reflective objects or rely on strong supervision guidance in terms of additional user-provided annotations of visible image regions of the mirrors, thereby limiting the practical usability. In contrast, in this paper, we present NeRF-MD, a method which shows that NeRFs can be considered as mirror detectors and which is capable of reconstructing neural radiance fields of scenes containing mirroring surfaces without the need for prior annotations. To this end, we first compute an initial estimate of the scene geometry by training a standard NeRF using a depth reprojection loss. Our key insight lies in the fact that parts of the scene corresponding to a mirroring surface will still exhibit a significant photometric inconsistency, whereas the remaining parts are already reconstructed in a plausible manner. This allows us to detect mirror surfaces by fitting geometric primitives to such inconsistent regions in this initial stage of the training. Using this information, we then jointly optimize the radiance field and mirror geometry in a second training stage to refine their quality. We demonstrate the capability of our method to allow the faithful detection of mirrors in the scene as well as the reconstruction of a single consistent scene representation, and demonstrate its potential in comparison to baseline and mirror-aware approaches.  
  </ol>  
</details>  
  
  



