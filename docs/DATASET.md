# Dataset interface (data not included)

The complete dataset is withheld during manuscript review. The eventual release will use a normalized-polygon instance-segmentation directory layout:

```text
dataset/
  images/{train,val,test}/
  labels/{train,val,test}/
  dataset.yaml
```

Each object is stored as one normalized polygon row:

```text
class_id x1 y1 x2 y2 ... xn yn
```

Class IDs are zero-based and follow this order:

| ID | Text prompt |
|---:|---|
| 0 | breakage |
| 1 | contamination |
| 2 | crack |
| 3 | dirt |
| 4 | missing |
| 5 | shelter |

No image, annotation, split manifest, augmented sample, or private filesystem path is included in this repository.
