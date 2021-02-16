## Spatio Temporal Contrastive Film: Pytorch (wip)
A Pytorch implementation of Spatio Temporal Contrastive Video Representation Learning for film trailer analysis. 
This repo is designed for unsupervised analysis of clips extracted from movie trailers. 

Check out the paper this implementation is based on here: https://arxiv.org/abs/2008.03800

### Data Pre-Processing - This method works but is convoluted at present
First you will need to split your trailers into scenes. `PySceneDetect` works well for this. 
The `create_trans_data_frame` function in `dataprocessing.py` will split your scenes into two transformed chunks of shape `1 x 16 x 112 x 112`. 
To use this function pass a dataframe with the columns `Filepath, Name, Scene, Genre` and save the resulting pkl file. 

### Training the model
An example config is given in `main.py`. Pass the config to main to train the model. 
#
## Evaluation
Simply run the code with `Train=False` 
The model will post all the data via a `SummaryWriter` to a directory specified in the config object. You can also create T-SNE and K-Means plots using the `Visualisation` class provided. 

If setup etc is too complicated rn check back in a couple of weeks as this repo is being updated daily.

