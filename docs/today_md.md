<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Neural-Radiance-Fields-for-the-Real-World:-A-Survey>Neural Radiance Fields for the Real World: A Survey</a></li>
        <li><a href=#DWTNeRF:-Boosting-Few-shot-Neural-Radiance-Fields-via-Discrete-Wavelet-Transform>DWTNeRF: Boosting Few-shot Neural Radiance Fields via Discrete Wavelet Transform</a></li>
      </ul>
    </li>
  </ol>
</details>

## NeRF  

### [Neural Radiance Fields for the Real World: A Survey](http://arxiv.org/abs/2501.13104)  
Wenhui Xiao, Remi Chierchia, Rodrigo Santa Cruz, Xuesong Li, David Ahmedt-Aristizabal, Olivier Salvado, Clinton Fookes, Leo Lebrat  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRFs) have remodeled 3D scene representation since release. NeRFs can effectively reconstruct complex 3D scenes from 2D images, advancing different fields and applications such as scene understanding, 3D content generation, and robotics. Despite significant research progress, a thorough review of recent innovations, applications, and challenges is lacking. This survey compiles key theoretical advancements and alternative representations and investigates emerging challenges. It further explores applications on reconstruction, highlights NeRFs' impact on computer vision and robotics, and reviews essential datasets and toolkits. By identifying gaps in the literature, this survey discusses open challenges and offers directions for future research.  
  </ol>  
</details>  
  
### [DWTNeRF: Boosting Few-shot Neural Radiance Fields via Discrete Wavelet Transform](http://arxiv.org/abs/2501.12637)  
Hung Nguyen, Blark Runfa Li, Truong Nguyen  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Neural Radiance Fields (NeRF) has achieved superior performance in novel view synthesis and 3D scene representation, but its practical applications are hindered by slow convergence and reliance on dense training views. To this end, we present DWTNeRF, a unified framework based on Instant-NGP's fast-training hash encoding. It is coupled with regularization terms designed for few-shot NeRF, which operates on sparse training views. Our DWTNeRF includes a novel Discrete Wavelet loss that allows explicit prioritization of low frequencies directly in the training objective, reducing few-shot NeRF's overfitting on high frequencies in earlier training stages. We additionally introduce a model-based approach, based on multi-head attention, that is compatible with INGP-based models, which are sensitive to architectural changes. On the 3-shot LLFF benchmark, DWTNeRF outperforms Vanilla NeRF by 15.07% in PSNR, 24.45% in SSIM and 36.30% in LPIPS. Our approach encourages a re-thinking of current few-shot approaches for INGP-based models.  
  </ol>  
</details>  
**comments**: 10 pages, 6 figures  
  
  



