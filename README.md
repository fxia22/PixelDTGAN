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

# Monitor the performance


- Install display package with: `luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec`
- Start the server with: `th -ldisplay.start`
- Open this URL in your browser: `http://localhost:8000`

Below shows the results after 7 epochs, each 3*1 block is generated cloth, true cloth, input image. Errors of G, D, and A network will be plotted.
 
![epoch 7](https://github.com/fxia22/pldtgan/blob/master/epoch7.jpg)
