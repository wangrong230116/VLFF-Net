# VLFF-Net

Official review-stage code release for **VLFF-Net: Text-Guided Vision-Language Feature Fusion for Insulator Defect Instance Segmentation**.

VLFF-Net conditions multi-scale visual features on defect-category text. This repository currently exposes the self-contained visual-conditioned text enhancement (VCTE) blocks, a model-graph preview, dataset-format documentation, and tensor-shape tests.

> **Staged release notice**
>
> This public repository is intentionally a partial release while the manuscript is under review. The complete training and evaluation pipeline, dataset split and annotations, trained weights, all ablation scripts, and Jetson/TensorRT deployment assets will be released after paper acceptance, subject to data-distribution permission.

## Currently available

- `vlff_net/vcte.py`: text adapter, visual adapter, visual-conditioned text enhancer, and text-conditioned neck modulator.
- `configs/vlff_net_preview.yaml`: readable architecture preview for the six-class instance-segmentation model.
- `tests/test_vcte_shapes.py`: CPU smoke tests for all released modules.
- `docs/DATASET.md`: expected normalized-polygon instance-segmentation format, without images or labels.

## Withheld until acceptance

- the 3,800-image augmented dataset and the original/augmented split manifests;
- full end-to-end model integration and training runtime;
- pretrained and fine-tuned checkpoints;
- full training, evaluation, ablation, Grad-CAM, and t-SNE pipelines;
- Jetson export, TensorRT engines, calibration data, and deployment benchmarks.

## Quick check

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

The released components are framework modules, not an end-to-end training package. The preview YAML documents integration points whose runtime implementations remain withheld during review.

## Defect categories

The study uses six text prompts: `breakage`, `contamination`, `crack`, `dirt`, `missing`, and `shelter`.

## Citation

If this preview is useful, please cite the repository using `CITATION.cff`. Final bibliographic metadata will be updated after acceptance.

## License

The released code is provided under the GNU Affero General Public License v3.0. Third-party components remain subject to their respective licenses.
