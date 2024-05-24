# HOW TO EXECUTE THE PROGRAM
Initial precautions:
    1. Set the file 'configuration.yaml'.

First Script: model_pc.py
    1. Change the path to 'configuration.yaml'.
    2. Run from the terminal (after connecting the sensor): python model_pc.py

Second Script: model_pc_align.py
    1. Change the path to 'configuration.yaml'.
    2. Make sure you have correctly set 'input_folder' and have .ply files inside it
    2. Run from terminal: python model_pc_align.py

Third Script: icp_align.py
    1. Change the path to "configuration.yaml".
    2. Ensure that the model path is correctly set to 'configuration.yaml'.
    2. Run from terminal (after connecting the sensor): python model_pc_align.py


PS: For correct functioning, it is MANDATORY to execute the steps exactly in the order they are given (also th order of the scripts).
