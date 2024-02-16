# Head and Gaze Joint Representation Learning

## Run the following after cd to LIMU_BERT_PUBLIC,

### Gaze Multi-Modal Architecture
`python hg_pretrain.py v3 hgbd 2 -s limu_v1 -g 0 -mt gaze_mm`

### Gaze Single-Modal Architecture
`python hg_pretrain.py v3 hgbd 2 -s limu_v1 -g 0 -mt gaze`

### Scipy Interpolation
`python baseline.py v3 hgbd 2 -s limu_v1 -g 0`

