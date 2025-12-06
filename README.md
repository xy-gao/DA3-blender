# DA3-blender
Blender addon for Depth-Anything-3 3D reconstruction

Input an image folder which contains single or multiple images, then you will get point cloud geometry nodes with material.

This blender addon is based on [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3). 

## Usage
1. Download Depth-Anything-3 model from operation panel (press N to toggle Sidebar, and click DA3 tab).
2. select an image folder.
3. Generate.

https://github.com/user-attachments/assets/6eeff6d0-a89f-4c2c-970b-47fe2b5475d3


## Installation (only the first time)
1. Download Zip from this github repo (but don't extract it).
2. In Blender, toggle System Console (Window > Toggle System Console) for installation logs tracking.
3. Install addon in blender preference (Edit > Preferences > Add-ons) with "Install from Disk" (v button in top right corner) and select downloaded zip.
4. Wait for Depth-Anything-3 git clone and python dependencies installation.
5. After addon activated, download Depth-Anything-3 model from operation panel (press N to toggle Sidebar, and click DA3 tab).

<img width="246" height="162" alt="517991309-15df7535-7177-4d9f-9a25-3dc3d6990ee4" src="https://github.com/user-attachments/assets/fd7233d9-ceb4-485c-8f36-5dfb35ef8b94" />


<img width="246" height="138" alt="517991330-436125db-a8ee-4c7f-a84b-b5b18ad6ef86" src="https://github.com/user-attachments/assets/24d50d1e-4dd6-4584-8203-6f99115a60ab" />

## Advanced usage
- There are several **Model**s to choose from. **DA3 Large** is recommended unless you have a lot of VRAM.
- The scale will be very small by default. Check **Use Metric** to use the DA3 Metric model to help scale it to approximately life-size. This is twice as slow. You will have to click the button to download the Metric model if you haven't already.
- The default resolution is only 504x504 (or less for non-square images). You can change the resolution of the longest side in the **Process Resolution** box, but it must be a multiple of 14. If you don't know your 14 times tables, Blender supports typing maths in the box, eg. `504+14`. Higher resolutions use a lot more VRAM and will fail.
- If you want to specify the resolution of the shortest side instead of the longest side, select **Lower Bound Resize** from the drop-down box. That will massively increase VRAM usage and is not recommended.
- There is a limit to how many images the model can process at once without crashing, based on VRAM. For 4GB of VRAM, at 504x280, the limit is 10. Set the **Batch Size** to whatever the limit is for your graphics card (by experimenting). Set the **Batch Mode** to how you want to handle more than that many images. **Skip Frames** will choose 10 evenly spaced images from the folder for a single batch, and is the only mode with good alignment. **Last Frame Overlap** will do it in batches using one frame of overlap to align the batches. **First-Last Overlap** will use two frames of overlap for better alignment (in theory). And **Ignore Batch Size** will try to do all the images at once and risk crashing.
- **Use Ray-based Pose** will use a slower more precise method of aligning cameras within a batch, but doesn't help alignment between batches.
- **Use Half Precision** will run the model in mixed precision, mostly 16-bit, reducing the VRAM used to run the model (but the model itself still uses the same VRAM), and making it faster, with only a slight loss of precision.
- Click **Unload Model** after you have finished to free VRAM for other things, otherwise the model will stay in VRAM.
- To view the confidence of each point in the point cloud, select the point cloud then click on the **Shading** tab at the top of the screen. In the node editor, change the **Factor** of the yellow **Mix** node to `1.0` (or something between 0 and 1) to show the confidence of each point instead of the colour.
- To change the size of each point, select the point cloud then click on the **Geometry Nodes** tab at the top of the screen. In the node editor, change the **Radius** of the green **Mesh to Points** node to the desired size.
- To hide points below a certain confidence level, select the point cloud, then click on the blue spanner icon in the bottom right column of icons, and set **Threshold** to a value between 1 and about 30. Setting it to `2.0` will filter out almost all the noise, but also some of the background.
- **Generate Meshes** will create a separate textured mesh for each image instead of a single point cloud. The meshes use the original full-resolution image as a texture.
- To view the scene from one of the cameras, select the camera, move the mouse over the 3D View, and press Ctrl+Numpad0

## Tested on
- Win10, Win11
- Blender 4.2, 4.4
- cuda 12.6
- 4GB VRAM GTX 970

also tested on Ubuntu 25.10, Blender 5.0, CUDA 13.0 https://github.com/xy-gao/DA3-blender/issues/1#issue-3652866452

