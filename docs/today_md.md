<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#Light-Transport-aware-Diffusion-Posterior-Sampling-for-Single-View-Reconstruction-of-3D-Volumes>Light Transport-aware Diffusion Posterior Sampling for Single-View Reconstruction of 3D Volumes</a></li>
      </ul>
    </li>
  </ol>
</details>

## NeRF  

### [Light Transport-aware Diffusion Posterior Sampling for Single-View Reconstruction of 3D Volumes](http://arxiv.org/abs/2501.05226)  
Ludwic Leonard, Nils Thuerey, Ruediger Westermann  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    We introduce a single-view reconstruction technique of volumetric fields in which multiple light scattering effects are omnipresent, such as in clouds. We model the unknown distribution of volumetric fields using an unconditional diffusion model trained on a novel benchmark dataset comprising 1,000 synthetically simulated volumetric density fields. The neural diffusion model is trained on the latent codes of a novel, diffusion-friendly, monoplanar representation. The generative model is used to incorporate a tailored parametric diffusion posterior sampling technique into different reconstruction tasks. A physically-based differentiable volume renderer is employed to provide gradients with respect to light transport in the latent space. This stands in contrast to classic NeRF approaches and makes the reconstructions better aligned with observed data. Through various experiments, we demonstrate single-view reconstruction of volumetric clouds at a previously unattainable quality.  
  </ol>  
</details>  
  
  



