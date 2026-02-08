# Attribution and Acknowledgments

This ComfyUI custom node combines functionality from two excellent projects:

## FlashVSR
- **Original Repository**: [JunhaoZhuang/FlashVSR](https://github.com/JunhaoZhuang/FlashVSR)
- **ComfyUI Node**: [flashvsr_ultra_fast](https://github.com/ExponentialML/ComfyUI_Native_FlashVSR)
- **License**: GNU General Public License v3.0 (GPL-3.0)
- **Purpose**: Video Super Resolution using diffusion models
- **Copyright**: Original FlashVSR authors and contributors

FlashVSR provides state-of-the-art video super-resolution capabilities with efficient streaming processing.

## GIMM-VFI
- **Original Repository**: [GIMM-VFI Research Project](https://github.com/baiqi96/GIMM-VFI)
- **ComfyUI Node**: [Kijai/ComfyUI-GIMM-VFI](https://github.com/Kijai/ComfyUI-GIMM-VFI)
- **License**: S-Lab License 1.0 (Non-commercial use)
- **Purpose**: Generalizable Implicit Neural Representation for Video Frame Interpolation
- **Copyright**: S-Lab and original GIMM-VFI authors

GIMM-VFI provides advanced frame interpolation using implicit neural representations.

## VSRFI (This Project)
- **Purpose**: Combines FlashVSR and GIMM-VFI into a single unified node
- **License**: GNU General Public License v3.0 (GPL-3.0) - inherited from FlashVSR
- **Additional Restrictions**: Non-commercial use restriction from GIMM-VFI applies to interpolation functionality
- **Author**: Neil (with assistance from Claude)

## License Compliance

This project is licensed under **GPL v3.0** (the most restrictive license among the combined components).

**Important Notes**:
1. The GIMM-VFI components retain their **non-commercial use restriction** under the S-Lab License
2. If you use the interpolation features (which use GIMM-VFI), you must comply with the non-commercial terms
3. All source code is provided under GPL v3.0, which requires:
   - Source code availability for derivative works
   - Same GPL v3.0 license for derivative works
   - Attribution to original authors

## Third-Party Components

This project includes code from:
- **flashvsr_src/**: FlashVSR model implementation (GPL-3.0)
- **gimmvfi/**: GIMM-VFI model implementation (S-Lab License 1.0)

Original licenses for these components are preserved in the `licenses/` directory.

## Model Weights

Model weights are automatically downloaded from HuggingFace:
- FlashVSR models: [JunhaoZhuang/FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1)
- GIMM-VFI models: [Kijai/GIMM-VFI_safetensors](https://huggingface.co/Kijai/GIMM-VFI_safetensors)

Please refer to the respective model repositories for their licensing terms.

## Contact

For questions about commercial use of the interpolation features, please contact the GIMM-VFI authors as specified in the S-Lab License.
