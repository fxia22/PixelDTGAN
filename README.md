# Pixel Level Domain Transfer
A torch implementation of "Pixel-Level Domain Transfer". based on [dcgan.torch](https://github.com/soumith/dcgan.torch). 



# Dataset
The dataset used is "LookBook", from [Donggeun Yoo](https://dgyoo.github.io). 


# Training

To train the model, put the LOOKBOOK dataset under repository, resize images to 64*64. Prepare the dataset using `prepare_data.ipynb`.
Then run 
```
th main.lua
```

You can tune the parameters, such as number of filters, optimizer, etc.

# Example results

Example results on LOOKBOOK dataset(top), left is input, right is generated clothes. Results on a similar dataset (bottom). 
More results will be added soon.

![Results](https://github.com/fxia22/pldtgan/blob/master/gan.jpg)
