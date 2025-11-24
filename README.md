# DA3-blender
Blender addon for Depth-Anything-3 3D reconstruction

Input an image folder which contains single or multiple images, then you will get point could geometry nodes with material.

This blender addon is based on [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3). 

## Usage
1. Download Depth-Anything-3 model from operation panel.
2. select an image folder.
3. Generate.



https://github.com/user-attachments/assets/6eeff6d0-a89f-4c2c-970b-47fe2b5475d3


## Installation (only the first time)
Current implementation download and use DA3-LARGE model (My GPU spec is not enough for DA3-GIANT). If you want to use other models, change this line of code in `operators.py`.
```python
_URL = "https://huggingface.co/depth-anything/DA3-LARGE/resolve/main/model.safetensors"
```
1. Download Zip from this github repo.
2. Toggle System Console for installation logs tracking.
3. Install addon in blender preference with "Install from Disk" and select downloaded zip.
4. Wait for Depth-Anything-3 git clone and python dependencies installation.
5. After addon activated, download Depth-Anything-3 model from operation panel.



## Tested on
- Win11
- Blender 4.2
- cuda 12.6

also tested on Ubuntu 25.10, Blender 5.0, CUDA 13.0 https://github.com/xy-gao/DA3-blender/issues/1#issue-3652866452

