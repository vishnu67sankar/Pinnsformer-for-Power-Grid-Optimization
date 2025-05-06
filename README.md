# DL Project
This team project, conducted by Vishnu Sankar, Ilias Baali, Cole Johnson, and Priyanshu Mehta, leverages a Transformer architecture integrated with PINNs to speed up power grid simulations while adhering to physics constraints.

## Steps to run the transformer model
1. Before cloning this repo, and make sure you have the LIPS framework installed in **developer** mode. This can be installed from https://github.com/IRT-SystemX/LIPS/tree/main. Also make sure you have the `ml4physim_startingkit_powergrid` repo from here https://github.com/IRT-SystemX/ml4physim_startingkit_powergrid.git. 

2. Now clone this repo, and copy the `transformer.py` file from transformer_model in this repo to -> `lips\augmented_simulators\tensorflow_models` located inside your LIPS-main directory. This is a baseline model implemented by Nina Inalgad, https://github.com/Ninalgad/powergrid-self-attention in tensorflow. We have adapted, and modified this model for our project. 

3. Copy the `baseline_transformer.py` and place it in the  `ml4physim_startingkit_powergrid` directory

4. Run the `baseline_transformer.py` (epoch = 1), make sure it works fine

5. Run `pinnsformer.py` to train the Physics Informed Transformer model. The code should automatically start downloading the input data if it is not present. Incase the code raises data not found error, then kindly install the data first from https://github.com/IRT-SystemX/ml4physim_startingkit_powergrid/blob/main/2_Datasets.ipynb