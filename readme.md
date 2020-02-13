## Aim

Build a formal framework that estimates the authorship probability for a given pair (user, photo).  
To illustrate the proposal, we use data gathered from TripAdvisor containing the reviews (with photos) of restaurants in six cities of different sizes.

## Setup

### Enviroment

* python3.6  
* Install `requirements.txt`

### Run pre-trained models

In order to run pre-trained models you have to:
 1. Download the pretrained models from: [URL]
 2. Unzip the compressed files under `models/` path.
 3. Run the `Main.py` file with `stage='test'` or `stage='stats'` (view parameters section). 

### Train the model with pre-generated data

In this case, you need to follow these steps:
 1. Download a city data from: https://dx.doi.org/10.34740/kaggle/dsv/944945
 2. Create a new folder under the existing "data" folder with the name of the city.
 2. Unzip the files under `data/<city>/` path.
 2. Run the `Main.py` file with `stage='grid'` or `stage='train'` and  `city='<city>'` (view parameters section). 

## Parameters

You can configure:

* **lrates:** = List of leaning rate values (if you want to try different values)
* **dpouts:** = List of dropout values (if you want to try different values)
* **epochs:** = Epoch number
* **seed:** = Random state
* **city:** = City to work with


## Citation

Please cite the following paper:

Jorge Díez, Pablo Pérez-Núñez, Oscar Luaces, Beatriz Remeseiro and Antonio Bahamonde: Towards Explainable Personalized Recommendations by Learning from Users’ Photos. Information Sciences, in press. 2020.
https://doi.org/10.1016/j.ins.2020.02.018