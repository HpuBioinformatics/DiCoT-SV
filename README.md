# DiCoT-SV

# Overview
Hi‑C‑based structural variation detection using a directional convolu-tion and color‑attention transformer

## Installation

### Step 1: Clone the repository

```
git clone https://github.com/XXXXXXXX/DiCoT-SV.git
cd DiCoT-SV
```

### Step 2: Create and activate virtual environment

```
conda create -n DiCoT-SV python=3.9
conda activate DiCoT-SV
```

### Step 3: Install dependencies

```
pip install -r requirements.txt
```

### Preprocessing and transformation into subgraphs 
Perform significance processing on the original Hi-C contact matrix, and then convert the processed Hi-C contact matrix into a subgraph.

Handle inter-chromosomal contact matrix image
```
python Enhance_inter_img.py --cool_file /path_to_mcool_file --folder_path /output_path
```
Handle intra-chromosomal contact matrix image
```
python Enhance_intra_img.py --cool_file /path_to_mcool_file --folder_path /output_path
```

### Use DiCoT-SV to detect the candidate SVs
In this step, the image obtained from the previous step is inspected, using the 'detection_model_weight.pth' file in the weight folder.
Both within the chromosomes and between the chromosomes, the detection is carried out using the following instructions.

```
python detection_SV_region.py --resume /path_trained_model --dataset /path_inter_or_intra_contact_matrix_image --output /output_path
```

### Determine the breakpoint
In this step, PCA is used to determine the breakpoints, and the surrounding sub-matrices are saved based on the breakpoints.

Handle inter-chromosomal candidate SVs
```
python PCA_inter.py --cool_file /path_to_mcool_file --candicate_inter_list /path_to_inter_candicate_SV_file --output /output_path
```
Handle intra-chromosomal candidate SVs
```
python PCA_intra.py --cool_file /path_to_mcool_file --candicate_intra_list /path_to_intra_candicate_SV_file --output /output_path
```

### Classify the candidate SVs
In this step, the sub-matrices extracted in the previous step are classified using the classification model, and the Classification_model_weigth.pth file in the weight folder is used.

Handle inter-chromosomal sub-matrices
```
python Classification_model/predict_inter.py --model /path_trained_model --candicate /path_to_sv_txt_inter --output /ouput_path
```
Handle intra-chromosomal sub-matrices
```
python Classification_model/predict_intra.py --model /path_trained_model --candicate /path_to_sv_txt_intra --output /ouput_path
```
The final SV list will be saved in the files /output_path/Inter_SV_list.txt and /output_path/Intra_SV_list.txt.


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or issues, please open an issue on GitHub or contact [your.email@example.com].
