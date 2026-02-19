# 360Cam-PGM-3DGS-Tools

A consolidated toolkit for turning 360° video captures into photogrammetry (PGM) datasets and optimized point clouds that seed 3D Gaussian Splatting (3DGS) projects.
The scripts cover every stage from frame extraction to PLY refinement, and the desktop GUI orchestrates the full flow for multiple photogrammetry and 3DGS apps.

> **Project status:** actively edited and expanded. Expect interface polish, richer presets, and additional documentation updates in upcoming revisions.
> The code in this repository was generated with the assistance of OpenAI Codex.

---

## Installation & Environment Setup

### Requirements
- Python **3.7** or newer
- `pip` for installing Python packages
- Python dependencies listed in `requirements.txt`
- [FFmpeg](https://ffmpeg.org/) available on your `PATH`
- PyTorch (`torch` **1.10+** and matching `torchvision` build) for the human masking tool
- GPU/CUDA is optional for masking, but accelerates large batches.


### Setup Steps
1. **Clone the repository**
   ```bash
   git clone https://github.com/Mistral-Yu/360Cam-PGM-3DGS-Tools.git
   cd 360Cam-PGM-3DGS-Tools
   ```
2. **Create and activate a Conda environment**
   ```bash
   conda create -n gs360 python=3.7
   conda activate gs360
   ```
   > **Tip:** pick any supported Python version (3.7+) that matches your GPU drivers/CUDA toolkits.
3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   > **Note:** PyTorch wheels are platform-specific. If `pip` cannot find a build for your OS/Python combo, install `torch`/`torchvision` manually inside the Conda environment before rerunning the command.
4. **Verify FFmpeg**
   ```bash
   ffmpeg -version
   ```

---

## Workflow at a Glance
GUI: `gs360_360GUI.py` (Tabs: `Video2Frames`, `FrameSelector`, `360PerspCut`, `SegmentationMaskTool`, `PlyOptimizer`, `MS360xmlToPerspCams`).
1. `Video2Frames` tab: Sample equirectangular frames from a 360° video.
2. `FrameSelector` tab: Score and retain the sharpest frames for Structure-from-Motion (SfM).
3. `360PerspCut` tab: Convert panoramas into perspective views.
4. `SegmentationMaskTool` tab: (Optional) Remove or isolate bystanders with `SegmentationMaskTool` to avoid reconstruction ghosts.
5. Use exported views to align in your photogrammetry software (RealityScan, Metashape, etc.) and export the PLY point cloud plus camera metadata.
6. `PlyOptimizer` tab: (Optional) Downsample/merge the PLY for your 3DGS tool (PostShot, gsplat, etc.) with `PlyOptimizer`. Delete sky point clouds generated with photogrammetry software, add Sky point clouds.
7. `MS360xmlToPerspCams` tab: Convert Metashape camera XML (spherical) into perspective camera parameters → Colmap, Metashape xml, transform.json, RealityScan xmp.

### Workflows (GUI)
#### Rapid
1. Launch `gs360_360GUI.py`.
2. In the `360PerspCut` tab, click **Browse Video** and select a video file.
3. Choose a preset.
4. Under **Video (direct export)**, set the FPS.
5. Click **Run Export** to write the images.
6. **RealityScan**: launch the app, import images, then select all.
   Set Prior Calibration -> Prior to **Fixed** or Prior to Approximate, and change **Focal Length** to the value shown in the `360PerspCut` log
   (Preset default: 12 mm, fisheyelike: 17mm, full360coverage: 14 mm).
   Set Prior Lens Distortion -> Prior to **Fixed**.
7. Or **Metashape**: launch the app, import images, then go to Tools -> **Camera Calibration (Initial tab)**.
   Set Type to **Precalibration** and update **f** to the value shown in the `360PerspCut` log
   (Preset default: 533.33333, fisheyelike: 755.55556, full360coverage: 622.22222).
   Next, click the **Fixed parameters** Select button and **Check all** or check distortion parameters except f.
8. Bring the RealityScan or Metashape alignment results into a 3DGS tool such as PostShot.

#### Faster but low quality
1. Launch `gs360_360GUI.py`.
2. In the `Video2Frames` tab, extract 360 frames by setting the FPS and running the export.
3. In **Metashape**, import the frames, then go to **Tools -> Camera Calibration** and set **Camera Type = Spherical**.
4. Align the cameras.
5. Export from Metashape: **File -> Export -> Export Cameras** (Agisoft XML) and **File -> Export -> Point Cloud** (PLY).
6. Back in `gs360_360GUI.py`, open the **MS360xmlToPerspCams** tab and set **Input XML** to the exported XML.
7. Set **Format = transforms**, enable **PerspCut**, set **PerspCut input** to the 360 image folder, and set **Points PLY** to the exported PLY. Run the tool.
8. Use **PerspCut out**, `transforms.json`, and the rotated PLY in PostShot (or similar 3DGS tools).


### Recommended Workflow 
**TODO**
- Metashape, Multi-Cameras-System
- RealityCapture(Metashape), Hybrid scan combining mirrorless camera and 360-degree camera.
---

## License

This project is released under the [MIT License](LICENSE). Copyright (c) 2025 Yu.

---

## TODO / Ideas
- Flesh out a full GUI walkthrough (tab descriptions, launch parameters, screenshot gallery).
