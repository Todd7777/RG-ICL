# Data Directory

Place datasets here following this structure:

```
data/
├── lag/
│   ├── reference/
│   │   ├── non_glaucoma/
│   │   └── glaucoma/
│   ├── test/
│   │   ├── non_glaucoma/
│   │   └── glaucoma/
│   └── manifest.json
├── ddr/
│   ├── reference/
│   │   ├── no_dr/
│   │   ├── mild_npdr/
│   │   ├── moderate_npdr/
│   │   ├── severe_npdr/
│   │   ├── proliferative_dr/
│   │   └── ungradable/
│   ├── test/
│   └── manifest.json
├── chexpert/
│   ├── reference/
│   │   ├── images/
│   │   └── labels.csv
│   ├── test/
│   │   ├── images/
│   │   └── labels.csv
│   └── manifest.json
├── breakhis/
│   ├── reference/
│   │   ├── adenosis/
│   │   ├── fibroadenoma/
│   │   ├── phyllodes_tumor/
│   │   ├── tubular_adenoma/
│   │   ├── ductal_carcinoma/
│   │   ├── lobular_carcinoma/
│   │   ├── mucinous_carcinoma/
│   │   └── papillary_carcinoma/
│   ├── test/
│   └── manifest.json
├── medical_cxr_vqa/
│   ├── reference/
│   │   ├── images/
│   │   └── qa.json
│   ├── test/
│   │   ├── images/
│   │   └── qa.json
│   └── manifest.json
├── vqa_rad/
│   ├── reference/
│   │   ├── images/
│   │   └── qa.json
│   ├── test/
│   └── manifest.json
├── pathvqa/
│   ├── reference/
│   │   ├── images/
│   │   └── qa.json
│   ├── test/
│   └── manifest.json
└── pmc_vqa/
    ├── reference/
    │   ├── images/
    │   └── qa.json
    ├── test/
    └── manifest.json
```
