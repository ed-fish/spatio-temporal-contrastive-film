## Spatio Temporal Contrastive Film: Pytorch (wip)
A Pytorch implementation of Spatio Temporal Contrastive Video Representation Learning which has been adapted for film trailer analysis. 
This repo is designed for unsupervised analysis of clips extracted from movie trailers. 

Check out the paper this implementation is based on here: https://arxiv.org/abs/2008.03800
#
### Data Pre-Processing
First you will need to split your trailers into scenes. `PySceneDetect` works well for this. 
The `create_trans_data_frame` function in `dataprocessing.py` will split your scenes into two transformed chunks of shape `1 x 16 x 112 x 112`. 
To use this function pass a dataframe with the columns `Filepath, Name, Scene, Genre` and save the resulting pkl file. 

### Training the model
An example config is given in `main.py`. Pass the config to main to train the model. 

### Evaluation
Run the code with `Train=False` 
The model will post all the data via a `SummaryWriter` to a directory specified in the config object. This will also contain a Tensorboard T-SNE Plot - for kmeans use the `kmeans = True` flag.

If setup etc is too complicated rn check back in a couple of weeks as this repo is being updated daily.

