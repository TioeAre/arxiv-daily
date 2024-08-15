<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#image-matching>Image Matching</a></li>
      <ul>
        <li><a href=#RSD-DOG-:-A-New-Image-Descriptor-based-on-Second-Order-Derivatives>RSD-DOG : A New Image Descriptor based on Second Order Derivatives</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Rethinking-Open-Vocabulary-Segmentation-of-Radiance-Fields-in-3D-Space>Rethinking Open-Vocabulary Segmentation of Radiance Fields in 3D Space</a></li>
      </ul>
    </li>
  </ol>
</details>

## Image Matching  

### [RSD-DOG : A New Image Descriptor based on Second Order Derivatives](http://arxiv.org/abs/2408.07687)  
Darshan Venkatrayappa, Philippe Montesinos, Daniel Diep, Baptiste Magnier  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    This paper introduces the new and powerful image patch descriptor based on second order image statistics/derivatives. Here, the image patch is treated as a 3D surface with intensity being the 3rd dimension. The considered 3D surface has a rich set of second order features/statistics such as ridges, valleys, cliffs and so on, that can be easily captured by using the difference of rotating semi Gaussian filters. The originality of this method is based on successfully combining the response of the directional filters with that of the Difference of Gaussian (DOG) approach. The obtained descriptor shows a good discriminative power when dealing with the variations in illumination, scale, rotation, blur, viewpoint and compression. The experiments on image matching, demonstrates the advantage of the obtained descriptor when compared to its first order counterparts such as SIFT, DAISY, GLOH, GIST and LIDRIC.  
  </ol>  
</details>  
  
  



## NeRF  

### [Rethinking Open-Vocabulary Segmentation of Radiance Fields in 3D Space](http://arxiv.org/abs/2408.07416)  
Hyunjee Lee, Youngsik Yun, Jeongmin Bae, Seoha Kim, Youngjung Uh  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Understanding the 3D semantics of a scene is a fundamental problem for various scenarios such as embodied agents. While NeRFs and 3DGS excel at novel-view synthesis, previous methods for understanding their semantics have been limited to incomplete 3D understanding: their segmentation results are 2D masks and their supervision is anchored at 2D pixels. This paper revisits the problem set to pursue a better 3D understanding of a scene modeled by NeRFs and 3DGS as follows. 1) We directly supervise the 3D points to train the language embedding field. It achieves state-of-the-art accuracy without relying on multi-scale language embeddings. 2) We transfer the pre-trained language field to 3DGS, achieving the first real-time rendering speed without sacrificing training time or accuracy. 3) We introduce a 3D querying and evaluation protocol for assessing the reconstructed geometry and semantics together. Code, checkpoints, and annotations will be available online. Project page: https://hyunji12.github.io/Open3DRF  
  </ol>  
</details>  
**comments**: Project page: https://hyunji12.github.io/Open3DRF  
  
  



