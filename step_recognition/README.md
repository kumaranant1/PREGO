## Step Recognition (MiniROAD)

This module is different from original implementation of MiniRoad from the perspective of resource use and management. This implementation utilizes low RAM uses by avoiding loading the whole data into the RAM at once (which the original implementation did). Hence this uses HDF5 format to load the Data when required into the RAM. Hence,  dataset.py is changed to use h5py efficiently. 

This implmentation works perfectly in colab free version.

Before running the main.py script please change/run the following :- 

- follow the main instructions of PREGO or MiniROAD to download the dataset and make sure the cofing points to the correct `root_path`

- change the `output_file` in `CvtFrames_to_h5.py`

- run this script `CvtFrames_to_h5.py` inside this folder 

- change the `h5_path` to generated h5 file path from previous script in config at `PREGO\step_recognition\configs\miniroad_assembly101-O.yaml`

- now you can continue running `main.py` script to train the model.