# DA3-blender 1.1
Blender addon for Depth-Anything-3 3D reconstruction

Input an image folder which contains single or multiple images, then you will get point cloud geometry nodes with material.

This blender addon is based on [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3). 

## Version Changes for 1.1
- Asynchronous operation, with progress bars (Esc or right-click to cancel)
- DA3 Streaming batch mode
- New 1.1 models which fixed a bug in their training
- Model can now be loaded in 16-bit (hacky and experimental)
- Frame Stride option
- Reference View Strategy option
- Bug fixes (and new bugs)
- Installation now requires an extra step to install dependencies

## Usage
1. Download Depth-Anything-3 model from operation panel (press N to toggle Sidebar, and click DA3 tab).
2. select an image folder.
3. Generate.

https://github.com/user-attachments/assets/6eeff6d0-a89f-4c2c-970b-47fe2b5475d3


## Installation (only the first time)
1. Download Zip from this github repo (but don't extract it).
2. In Blender, toggle System Console (Window > Toggle System Console) for installation logs tracking.
3. Install addon in blender preference (Edit > Preferences > Add-ons) with "Install from Disk" (v button in top right corner) and select downloaded zip.
4. Expand the > DA3 Addon panel, then click on the Install Dependencies button.
5. Wait for Depth-Anything-3 git clone and python dependencies installation.
(optional) Type a path into the Model Folder box if you want to store DA3 models in a different folder.
6. Uncheck [ ] DA3 Addon, then check it again to restart the addon (or exit Blender and restart)
7. Close Preferences, then click in the 3D view and press N to toggle the Sidebar, click the DA3 tab.
8. Choose a model in the Model box, I recommend DA3 Large 1.1. Then click the download button and wait.

<img width="246" height="162" alt="517991309-15df7535-7177-4d9f-9a25-3dc3d6990ee4" src="https://github.com/user-attachments/assets/fd7233d9-ceb4-485c-8f36-5dfb35ef8b94" />


<img width="246" height="138" alt="517991330-436125db-a8ee-4c7f-a84b-b5b18ad6ef86" src="https://github.com/user-attachments/assets/24d50d1e-4dd6-4584-8203-6f99115a60ab" />

## Advanced usage
- There are several **Model**s to choose from. **DA3 Large** is recommended unless you have a lot of VRAM. The 1.1 versions are the latest.
- **Use FP16 Weights (experimental)** will load and run the model in 16-bit, reducing the VRAM used and making it faster, with a slight loss of precision.
- The scale will be very small by default. Check **Use Metric** to use the DA3 Metric model to help scale it to approximately life-size. This is twice as slow. You will have to click the button to download the Metric model if you haven't already.
- The default resolution is only 504x504 (or less for non-square images). You can change the resolution of the longest side in the **Process Resolution** box, but it must be a multiple of 14. If you don't know your 14 times tables, Blender supports typing maths in the box, eg. `504+14`. Higher resolutions use a lot more VRAM and will fail.
- If you want to specify the resolution of the shortest side instead of the longest side, select **Lower Bound Resize** from the drop-down box. That will massively increase VRAM usage and is not recommended.
- There is a limit to how many images the model can process at once without crashing, based on VRAM. For 4GB of VRAM (and enough normal RAM), at 504x280, the limit is 10. Set the **Batch Size** to whatever the limit is for your graphics card (by experimenting). If you have lots of VRAM, the examples on the DA3 website used a batch size of 120. Set the **Batch Mode** to how you want to handle more than that many images. **Skip Frames** will choose 10 evenly spaced images from the folder for a single batch, and is the mode with best alignment. **DA3 Streaming** (NEW) uses the official code for handling long sequences of images, half the batch will be overlap for better alignment. **Last Frame Overlap** will do it in batches using one frame of overlap to align the batches. **First-Last Overlap** will use two frames of overlap for better alignment (in theory). **Ignore Batch Size** will try to do all the images at once and risk crashing.
- **Frame Stride** allows you to skip frames and only use every *n*th frame, if you have too many images.
- **Ref View Strategy** controls how the model chooses the primary frame (in each batch) which other frames try to fit. **Saddle Balanced** picks the one that shares most features with other frames.
- **Advanced Streaming Options** is for the DA3 Streaming batch mode. You can specify an **Output** folder to store the temporary files and PLY files, otherwise it uses the input image folder. **Enable Loop Closure** will try to detect when the images return to place they've already been and link it up. **Alignment Library** should be set to **Torch** unless you know what you're doing. **Depth Threshold** should be set to the maximum depth you want to keep, and will be in metres if using Metric, or random units. The official DA3 Streaming codes uses 15 metres. **Confidence Threshold Coef** controls the fraction (between 0 and 1) of confidence required to keep pixels, the official code uses 0.7 (ie. only keep the best 30% of pixels). **Separate Chunk Collections** will put each batch into a separate point cloud so you can align the batches manually.
- **Use Ray-based Pose** will use a slower more precise method of aligning cameras within a batch, but doesn't help alignment between batches.
- **Filter Edges** removes pixels around sharp transitions in depth, because the model antialiases the depths of those pixels causing streamers of noise behind objects. Filter Edges works better than Min Confidence because it doesn't remove the background.
- **Min Confidence** removes pixels with a confidence below this. The model returns values between 1 and about 30. Setting it to `2.0` will filter out almost all the noise, but also some of the background. The default 0.5 only filters out pixels whose confidence is set to 0 by this addon.
- **Detect Motion** will detect moving objects that are present in one frame but absent in another where they should be visible. It then puts the moving objects into their own point clouds and animates them. Press Play in the animation panel to watch. Static geometry from all frames will always be visible. You may need to manually increase the length of the scene's animation. Detect Motion doesn't work well on feet or objects that are near other objects. It isn't optimised, so it will have problems with large numbers of frames.
- **Use Segmentation** will use a segmentation model to separate out objects into their own point clouds. **YOLOE Large PF** detects the most objects (badly). **YOLO11 X-Large** is the most reliable. If people are detected, their confidence will be dropped to 1 and they won't be used for alignment. The segmentation models will be downloaded automatically.
- **Generate Meshes** will create a separate textured mesh for each image instead of a single point cloud. The meshes use the original full-resolution image as a texture. You will have many meshes layered on top of each other that you need to clean up manually if you want to use them. It makes no attempt to combine meshes into a single mesh yet.
- Click **Unload Model** after you have finished to free VRAM for other things, otherwise the model will stay in VRAM.
- To view the confidence of each point in the point cloud, select the point cloud then click on the **Shading** tab at the top of the screen. In the node editor, change the **Factor** of the yellow **Mix** node to `1.0` (or something between 0 and 1) to show the confidence of each point instead of the colour.
- To change the size of each point, select the point cloud then click on the **Geometry Nodes** tab at the top of the screen. In the node editor, change the **Radius** of the green **Mesh to Points** node to the desired size.
- To hide points below a certain confidence level, select the point cloud, then click on the blue spanner icon in the bottom right column of icons, and set **Threshold** to a value between 1 and about 30. Setting it to `2.0` will filter out almost all the noise, but also some of the background.
- To view the scene from one of the cameras, select the camera, move the mouse over the 3D View, and press Ctrl+Numpad0

## Sample images and folder location
The addon is installed to the `%appdata%\Blender Foundation\Blender\5.0\scripts\addons\DA3-blender-main` folder (or whatever version number you are using). Inside is a sample_images folder, and the models folder where models are downloaded by default.

## Tested on
- Win10, Win11
- Blender 4.2, 4.4, 5.0
- cuda 12.6, 13.0
- 4GB VRAM GTX 970