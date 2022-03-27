# Synthesize Distorted Image and Its Control Points
<div align="center">
  <img width="850" src="https://github.com/gwxie/Synthesize-Distorted-Image-and-Its-Control-Points/blob/main/dataset.jpg">
  
  <p>We uniformly sample a set of (b) reference points on (a) scanned document image, and then perform geometric deformation on them to get (c) distorted image and (d) control points. (e) Synthetic data consists of distorted image, reference points and control points.</p>
  
</div>

See [“Document Dewarping with Control Points”](https://arxiv.org/pdf/2203.10543.pdf) for more information.

# Quick Start

```bash
python perturbed_images_generation_multiProcess.py            --path=./scan/new/ --bg_path=./background/ --output_path=./output/`
       perturbed_images_generation_multiProcess_addition1.py 
       perturbed_images_generation_multiProcess_addition2.py 
       perturbed_images_generation_multiProcess_addition3.py 
```

# Requirements
<p>python >=3.7</p>
<p>opencv-python</p>
<p>scipy</p>
<p>math</p>
<p>pickle</p>
