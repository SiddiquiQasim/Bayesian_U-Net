# Installation
```
git clone <Repo>
```
```
pip install -r requirements.txt
```
```
python setup.py install
```

# Usage
`cd semantic_uncertainty`
Edit `train.py`, `ìnference.py` and `visualization.py` as per requirement

## Training

```
if __name__ == '__main__':

    '''
    img_shape : shape of the input/label image
    mcd=True : if estimating uncertainty with Monte-Carlo Dropout
    bayesian=True : if estimating uncertainty with Bayesian U-Net
    mcd=False, bayesain=False : if estimating uncertainty with Deep Ensemble
    data_dir='data/slices/' : location of images/labels
    num_of_epochs=100 : number of epochs to train
    idx=0 : model index id if using Deep Ensemble
    '''
    train(img_shape=(320,320), mcd=False, bayesian=False, data_dir='data/slices/', num_of_epochs=100, idx=0)
```

`python train.py` to run model training

## Inference

```
if __name__ == '__main__':

    '''
    img_shape : shape of the input/label image
    mcd=True : if estimating uncertainty with Monte-Carlo Dropout
    bayesian=True : if estimating uncertainty with Bayesian U-Net
    mcd=False, bayesain=False : if estimating uncertainty with Deep Ensemble
    data_dir='data/slices/' : location of images/labels
    fpass=10 : number of sample to estimate uncertainty
    '''
    inference(img_shape=(320,320), mcd=False, bayesian=False, data_dir='data/slices/', fpass=10)
```

`python inference.py` to run inference to estimate uncertainty
## Uncertainty Visualization
```
if __name__ == '__main__':

    '''
    img_shape : shape of the input/label image
    mcd=True : if estimating uncertainty with Monte-Carlo Dropout
    bayesian=True : if estimating uncertainty with Bayesian U-Net
    mcd=False, bayesain=False : if estimating uncertainty with Deep Ensemble
    data_dir='data/slices/' : location of images/labels
    idx=0 : model index id if using Deep Ensemble
    '''
    visualize(img_shape=(320,320), mcd=False, bayesian=False, data_dir='data/slices/', idx=0)
```

`python visualization.py` to visualize uncertainty