<details>
  <summary><b>TOC</b></summary>
  <ol>
    <li><a href=#visual-localization>Visual Localization</a></li>
      <ul>
        <li><a href=#Visual-Delta-Generator-with-Large-Multi-modal-Models-for-Semi-supervised-Composed-Image-Retrieval>Visual Delta Generator with Large Multi-modal Models for Semi-supervised Composed Image Retrieval</a></li>
      </ul>
    </li>
    <li><a href=#nerf>NeRF</a></li>
      <ul>
        <li><a href=#DreamCraft:-Text-Guided-Generation-of-Functional-3D-Environments-in-Minecraft>DreamCraft: Text-Guided Generation of Functional 3D Environments in Minecraft</a></li>
      </ul>
    </li>
  </ol>
</details>

## Visual Localization  

### [Visual Delta Generator with Large Multi-modal Models for Semi-supervised Composed Image Retrieval](http://arxiv.org/abs/2404.15516)  
Young Kyun Jang, Donghyun Kim, Zihang Meng, Dat Huynh, Ser-Nam Lim  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Composed Image Retrieval (CIR) is a task that retrieves images similar to a query, based on a provided textual modification. Current techniques rely on supervised learning for CIR models using labeled triplets of the reference image, text, target image. These specific triplets are not as commonly available as simple image-text pairs, limiting the widespread use of CIR and its scalability. On the other hand, zero-shot CIR can be relatively easily trained with image-caption pairs without considering the image-to-image relation, but this approach tends to yield lower accuracy. We propose a new semi-supervised CIR approach where we search for a reference and its related target images in auxiliary data and learn our large language model-based Visual Delta Generator (VDG) to generate text describing the visual difference (i.e., visual delta) between the two. VDG, equipped with fluent language knowledge and being model agnostic, can generate pseudo triplets to boost the performance of CIR models. Our approach significantly improves the existing supervised learning approaches and achieves state-of-the-art results on the CIR benchmarks.  
  </ol>  
</details>  
**comments**: 15 pages  
  
  



## NeRF  

### [DreamCraft: Text-Guided Generation of Functional 3D Environments in Minecraft](http://arxiv.org/abs/2404.15538)  
Sam Earle, Filippos Kokkinos, Yuhe Nie, Julian Togelius, Roberta Raileanu  
<details>  
  <summary>Abstract</summary>  
  <ol>  
    Procedural Content Generation (PCG) algorithms enable the automatic generation of complex and diverse artifacts. However, they don't provide high-level control over the generated content and typically require domain expertise. In contrast, text-to-3D methods allow users to specify desired characteristics in natural language, offering a high amount of flexibility and expressivity. But unlike PCG, such approaches cannot guarantee functionality, which is crucial for certain applications like game design. In this paper, we present a method for generating functional 3D artifacts from free-form text prompts in the open-world game Minecraft. Our method, DreamCraft, trains quantized Neural Radiance Fields (NeRFs) to represent artifacts that, when viewed in-game, match given text descriptions. We find that DreamCraft produces more aligned in-game artifacts than a baseline that post-processes the output of an unconstrained NeRF. Thanks to the quantized representation of the environment, functional constraints can be integrated using specialized loss terms. We show how this can be leveraged to generate 3D structures that match a target distribution or obey certain adjacency rules over the block types. DreamCraft inherits a high degree of expressivity and controllability from the NeRF, while still being able to incorporate functional constraints through domain-specific objectives.  
  </ol>  
</details>  
**comments**: 16 pages, 9 figures, accepted to Foundation of Digital Games 2024  
  
  



