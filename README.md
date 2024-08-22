### Self-Supervised Few-Shot Learning for Music Symbol Classification in Historical Manuscripts

This project is based on code developed by Alfaro-Contreras et al. 2023 in **Few-Shot Symbol Classification via Self-Supervised Learning and Nearest Neighbor**
The codebase that this was developed from can be found here - https://github.com/mariaalfaroc/ssl-symbol-classification

#### Running the code

This project was developed on python version `3.10.13`. Required dependencies can be found in the `requirements.txt` file.

In order to run the code the required data must be added to the `data` directory.

The main file to run all experiments is `start.py`. Seperate `run_{MODEL}_classifier.py` files can be used to run specific experiments with a pretrained encoder.