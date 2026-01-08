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
GUI: `gs360_360GUI.py` (Tabs: `Video2Frames`, `FrameSelector`, `360PerspCut`, `HumanMaskTool`, `PlyOptimizer`).
1. `Video2Frames` tab: Sample equirectangular frames from a 360° video.
2. `FrameSelector` tab: Score and retain the sharpest frames for Structure-from-Motion (SfM).
3. `360PerspCut` tab: Convert panoramas into perspective views.
4. `HumanMaskTool` tab: (Optional) Remove or isolate bystanders with `HumanMaskTool` to avoid reconstruction ghosts.
5. Use exported views to align in your photogrammetry software (RealityScan, Metashape, etc.) and export the PLY point cloud plus camera metadata.
6. `PlyOptimizer` tab: (Optional) Downsample/merge the PLY for your 3DGS tool (PostShot, gsplat, etc.) with `PlyOptimizer`.

### Example Photogrammetry → 3DGS Flow (RealityScan + PostShot)
- After RealityScan completes **Reconstruction** and **Colorize**, export the colorized mesh as a PLY and the camera **CSV**; that pairing feeds best into the `PlyOptimizer` tab.
- Start with a conservative `--target-points` value around **100,000** to keep PostShot responsive; rerun the optimizer with higher counts if needed.
- Other photogrammetry suites (Metashape, RealityCapture, etc.) work similarly. Use the **focal-length** lines printed by the `360PerspCut` tab as a starting point for camera intrinsics.

---

---

## License

This project is released under the [MIT License](LICENSE). Copyright (c) 2025 Yu.

---

## TODO / Ideas
- Flesh out a full GUI walkthrough (tab descriptions, launch parameters, screenshot gallery).
