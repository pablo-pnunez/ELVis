## Aim

Build a formal framework that estimates the authorship probability for a given pair (user, photo).  
To illustrate the proposal, we use data gathered from TripAdvisor containing the reviews (with photos) of restaurants in six cities of different sizes.

## Setup

### Environment

* `conda env create -f environment.yml`


### Option 1. Run pre-trained models

In order to run pre-trained models you have to:
 1. Download city data: 
    * [Gijon](https://www.aic.uniovi.es/downloadables/ELVis/gijon.zip)
    * [Madrid](https://www.aic.uniovi.es/downloadables/ELVis/madrid.zip)
    * [Barcelona](https://www.aic.uniovi.es/downloadables/ELVis/barcelona.zip)
    * [Paris](https://www.aic.uniovi.es/downloadables/ELVis/paris.zip)
    * [New York City](https://www.aic.uniovi.es/downloadables/ELVis/newyorkcity.zip)
    * [London](https://www.aic.uniovi.es/downloadables/ELVis/london.zip)
 2. Create a new folder called "data" (If it does not exist) and inside, another with the name of the city.
 3. Unzip the files under `data/<city>/` path.
 4. Download the pre-trained models from [here](https://www.aic.uniovi.es/downloadables/ELVis/models.zip).
 5. Create a folder called "models" (If it does not exist).
 6. Unzip the compressed files under `models/` path.
 7. Run the `Main.py` file with `stage='test'` or `stage='stats'` and  `city='<city>'` (view parameters section). 

### Option 2. Train the model with pre-generated data

In this case, you need to follow these steps:
 1. Download city data as in Option 1.
 2. Run the `Main.py` file with `stage='grid'` or `stage='train'` and  `city='<city>'` (view parameters section). 

## Parameters

You can configure:

* **stage**
    * **"stats"**: If you want to obtain stats about the dataset.
    * **"grid"**: To train a model testing different configuration values.
    * **"train"**: If you want to train the final model.
    * **"test"**: To evaluate the model.

* **city:** City to work with
* **lrates:** List of leaning rate values (if you want to try different values)
* **dpouts:** List of dropout values (if you want to try different values)
* **epochs:** Epoch number
* **seed:** Random state

## Raw data
The data from all the cities without preprocessing can be downloaded from [here](https://zenodo.org/record/5644892).

## Citation

Please cite the following paper:

Jorge Díez, Pablo Pérez-Núñez, Oscar Luaces, Beatriz Remeseiro and Antonio Bahamonde: Towards Explainable Personalized Recommendations by Learning from Users’ Photos. Information Sciences, in press. 2020.
https://doi.org/10.1016/j.ins.2020.02.018