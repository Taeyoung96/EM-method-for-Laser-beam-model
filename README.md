# EM-method-for-Laser-beam-model
Implementation "learn intrinsic parameter" pseudocode in Probabilistic robotics (Chapter 6.3.2)

This is an assignment for PROBABILISTIC ROBOTICS 2 (EEE7761) in Yonsei Univ.  
`sensor_data.py` is provided by Prof. [Euntai Kim](https://cilab.yonsei.ac.kr/).  

## Pseudocode  

<p align="center"><img src="https://user-images.githubusercontent.com/41863759/187023954-9f557026-8436-41e5-a828-24daa4902aee.JPG" width = "700" ></p>  


## Requirement  
- numpy  
- matplotlib  

## How to run  
1. `python sensor_data.py` : Generate laser beam data randomly and save in .npz format.  
2. `python EM_method.py` : Load `sensors.npz` to calculate the intrinsic parameter in the beam-based sensor model.  

## Result  

<p align="center"><img src="https://user-images.githubusercontent.com/41863759/187024175-2e6aedc0-88ce-41d9-a382-9c3dabc56bb6.JPG" width = "600" ></p>  
