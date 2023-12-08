---
layout: archive
title: "Canopy Cover Segmentation From Aerial Imagery Using Deep Learning"
permalink: /ML205/
---



## Introduction

  Urban trees positively impact the lives of urban residents. Many benefits are tangible and noticeable in everyday life, such as reducing the negative impacts from the urban heat island effect and saving on energy bills by alleviating extreme temperatures (Livesly et al., 2016; California Urban Forestry Act of 1978). In addition to these tangible benefits urban trees provide less tangible, but still important services such as sequestering carbon, reducing particulate air pollution, increasing property values, supporting job creation and business growth, and improving urban residents’ mental health (Livesly et al., 2016; Nowak et al., 2006; Kleerekoper et al., 2012; California Urban Forestry Act of 1978). Measuring and managing urban forests to understand and maximize their benefits is an expensive and time intensive process which is often done through fieldwork inventories of city trees by arborists. One way to measure an urban forest without fieldwork is by mapping urban canopy cover.  The California Urban Forestry Act created a goal to achieve a statewide "10-percent increase of tree canopy cover in urban areas by 2035" (California Urban Forestry Act of 1978). To achieve this goal, the state needs an accurate baseline map of canopy cover for California's urban areas from which to measure change.

  In urban environments, change in canopy cover growth happens slowly. Larger trees maintain their shape over time because they are trimmed to a static size. Trees that are planted start smaller than is detectable by most satellites and take many years to increase in size. Because of this, it is difficult to detect overall change in canopy growth. On the other hand, large trees in urban environments can quickly be removed, so loss may be easier to detect. Many studies have created tree cover canopy maps for small areas at high resolutions (using imagery and/or Lidar data from unpiloted aerial vehicles or planes (Braga et al., 2020; Codemo et al., 2022; Miraki et al., 2021; Zhao et al., 2023)), large areas with a combination of high-resolution data sources taken at a compilation of dates and times from many different sensors, making it difficult to create consistent repeat datasets to examine changes over time (Guo et al., 2023), or large repeatable areas at low resolutions (using imagery from satellites such as Landsat or Sentinel (Dewitz & U.S. Geological Survey, 2019). Our study hopes to use high resolution imagery taken every three years by the National Agricultural Imagery Program (NAIP) to create a canopy cover map for urban California that can be repeated every time a new NAIP dataset is created. We plan to develop several models trained specifically for each of the six climate zones across California, so that we can accurately map trees from cities with very different landscapes from Los Angeles, in a mediterranean climate, to Truckee, at high elevation city in the Sierra Nevada, to dry desert cities like Palm Springs (Mcpherson, 2010). The first step in this study is to develop robust methods for creating those models, which can then be trained with imagery from each specific climate zone. For this project, we started with developing a neural network that segments tree canopy from 2020 NAIP imagery in urban areas in the Southern California Coast climate zone, which spans from Point Conception to San Diego along California’s coast.  


## Methods
#### Data

  Our training data consists of 448x448 of NAIP imagery in a raster format (.tifs). NAIP. These rasters were clipped from NAIP imagery downloaded and mosaicked that covered four cities in the Southern California Coast climate zone: Riverside, Claremont, Long Beach, and Santa Monica. These cities were chosen because they had LiDAR data available, are major population centers, and represent slightly varying environments from the coast to the most inland parts of the climate zone. 
 The imagery has four bands: red, blue, green, and near-infrared and is 60 cm in resolution. We added a fifth band which is an established vegetation index called a Normalized Difference Vegetation Index (NDVI). NDVIs can help balance the negative effects of shadows in imagery which sometimes cover canopies, and make vegetation stand out in an image. NDVIs are calculated using the following equation: (Red-NIR)/(Red+NIR+1e-8). The 1e-8 was added to prevent division by 0. For each band, we took the maximum and minimum value for the entire state dataset across the state of California, and used it to standardize our data. We used maximum-minimum normalization on every band in every 448x448 training sample. The values for each imagery band range were stretched to range from 0-255. An NDVI ranges from -1 to 1. For the NDVI bands, we converted the scale from -1 to 1 to 0-255 by adding 1 and multiplying by 2/255. 
 
  For each image, we had an accompanied label dataset of canopy cover which were created by using USGS LiDAR to create canopy height models. After processing the LiDAR for each city to a canopy height model, used a threshold on the canopy height model and corresponding NAIP imagery to assign all areas over 2 meters high and with an NDVI threshold of 0.4 as canopy cover. LiDAR imagery was selected to be as close to the year of the NAIP imagery (2020) as possible. Creating automated training data has shown some success in being used as training data to predict canopy cover in a previous study (Weinstein et al., 2019). Once our data was cropped into 448x448x5 sections, we split our data into 80% training data, 10% test data, and 10% evaluation data.

[Local Image](images/training_data.jpg)

*Figure 1: On the left, NAIP imagery displayed in true color that was used for the training process. On the right, the label applied to the mask where white is canopy and black is not canopy. This imagery comes from Claremont, CA in 2020.*

#### Model Choice

  One reason a neural network is appropriate for this project is because we can quickly create large amounts of training data using LiDAR. Additionally, our decision boundaries are complex. From above the values and shapes of small shrubs and grass patches can be quite similar to trees, so identifying what is a tree using only visual imagery is a difficult task. Although it takes a long time to train a neural network, we have four v100 GPUs available to use, so we can handle large amounts of data. A convolutional neural network like U-Net will work better than an artificial neural network due to needing spatial coherence and the high number of parameters that using an artificial neural network would create for our model. For an artificial neural network with two hidden layers, using our 448x448x5 input data there would be a 2,014,103,010,560 model parameters, which is quite unreasonable!
    
  Because of success in other studies segmenting tree canopies, we decided to create a convolution neural network with a U-Net architecture, which was originally created for biomedical image segmentation (Ronneberger et al., 2015; Wang et al., 2021; Martins et al.,2021). Several publications have found U-Net to work better than or similarly to other deep learning architectures for tree canopy segmentation (Wang et al., 2021; Martins et al.,2021). Based on U-Nets’ success in other publications, we chose it for our initial trials. 

![Local Image](images/model_architecture.jpg)

*Figure 2: Model Structure for the standard four block encoder-decoder with skip connections U-Net architecture we used.*

#### Network Architecture

  We used a standard U-Net architecture with an encoder-decoder structure with skip connections. The input has 5 channels which correspond to the four bands and vegetation index in the imagery. The encoder, which down samples the imagery, is made of four blocks. Each block has a 3x3 convolution filter, a reLU activation function, and a max pooling layer with a stride and kernel size of 2x2. The convolution layers have a padding of 1 pixel to prevent the image from reducing in size. 
      
  The decoder, which up samples the imagery, has four blocks. Each block has a transposed convolutional layer with a reLU activation filter, with a kernel size and stride of 2x2. There are skip connections from the encoder to decoder blocks which help the model maintain spatial details. The final layer is sent through a 1x1 convolution filter and creates a binary output map where 1 represents canopy and 0 represents everything else. The structure of this model is shown in figure 2.
    
  The model uses a cross entropy loss function to adjust model weights during training, which is often used for classification problems like ours and utilizes softmax activation followed by log transformation (Lau, 2022). Validation loss during training is shown in figure 4.

#### Model Parameters

  The model was created and run using the PyTorch machine learning library.  The model uses an Adams optimizer with standard parameters to update network weights (Kingma and Ba, 2017). I ran the model on two Tesla V100 GPUs. I trained the best model for 150 epochs with a batch size of 16. The learning rate was set to 0.0001. I ran several tests altering the number of training samples and epochs to see how performance was affected. 

## Results

  I trained the model three times with increasing epochs and training data. The first model was trained across 100 epochs with 500 training samples and resulted in a precision of 0.718, a recall of 0.314, and an F-Score of 0.427 (Figure 3). The second mode was trained across 150 epochs with 8,000 training samples and resulted in a precision of 0.753, a recall of 0.65, and an F-Score of 0.701 (Figure 3). The final model was trained across 200 epochs with 8,999 training samples and results in a precision of 0.773, a recall of 0.637, and an F-Score of 0.698.

*Figure 3: Model results. I ran the model three times with varying training epochs and amount of training samples used.*

*Figure 4: The validation loss over the model steps.*

*Figure 5: Top: The input image, true mask label data, and predicted canopy from our model in Santa Monica, CA. Bottom: The input image, true mask label data, and predicted canopy from our model in Riverside, CA.*

## Discussion

  In my experiment attempting to use U-Net to classify canopy cover in urban areas, I was able to classify some canopy, and think the results look reasonable. However, I only classified a small portion of the canopy in each of my test images compared to the true total (Figure 5). Ideally, the F-score would be closer to 0.9 rather than 0.7. There is a lot of work to do before the results can be used for any sort of analysis. When I first increased our amount of training data and the number of epochs used to run the model, the model had a higher F-Score. One way I might improve the model is by adding additional training data and using it to train the model. However, increasing the amount of training samples and number of epochs from 8,000 to 8,999 and 150 t0 200 respectively, the F-Scores were very simple, with the higher number of samples and epochs having a slightly lower F-Score at 0.698 compared to 0.701. Either adding more training data might not increase our accuracy, or we would need to add high quantities of training data. I am in the process of creating new canopy cover masks as training data for wider areas across the state that are pit-free. Pits in canopy height models are places where the laser passed through the canopy and the top return for a section of the canopy was measured at the ground level. Our canopy height model sometimes has pits within the canopies, which may be influencing our results. Hopefully, with an increased amount of training data, and more accurate training data, we can improve our model results.
    
  Additionally, I can improve the evaluation methods of the model. Evaluating the accuracy of the canopy cover model with automatically generated canopy cover masks is problematic in that there may be errors in those masks. Although the Weinstein et al. paper showed that automatically generated training data can be used to train a canopy cover model, I am in the process of creating manual annotations for close to 500 tiles across California’s cities to more accurately assess our model results. With more accurate canopy annotations for testing our model, we will have a better idea of how to change our model to get more accurate results. 


# References
Braga, J. R., Peripato, V., Dalagnol, R., P. Ferreira, M., Tarabalka, Y., O. C. Aragão, L. E., F. De Campos Velho, H., Shiguemori, E. H., & Wagner, F. H. (2020). Tree Crown Delineation Algorithm Based on a Convolutional Neural Network. Remote Sensing, 12(8), 1288. https://doi.org/10.3390/rs12081288
California Urban Forestry Act, California Assembly Bill 527 (1978). https://legiscan.com/CA/text/AB527/id/2832823#:~:text=The%20California%20Urban%20Forestry%20Act,multiple%20benefits%20in%20urban%20communities.

Codemo, A., Pianegonda, A., Ciolli, M., Favargiotti, S., & Albatici, R. (2022). Mapping Pervious Surfaces and Canopy Cover Using High-Resolution Airborne Imagery and Digital Elevation Models to Support Urban Planning. Sustainability, 14(10), 6149. https://doi.org/10.3390/su14106149

Dewitz, J. (2021). National Land Cover Database (NLCD) 2019 Products [dataset]. U.S. Geological Survey. https://doi.org/10.5066/P9KZCM54

Guo, J., Xu, Q., Zeng, Y., Liu, Z., & Zhu, X. X. (2023). Nationwide urban tree canopy mapping and coverage assessment in Brazil from high-resolution remote sensing images using deep learning. ISPRS Journal of Photogrammetry and Remote Sensing, 198, 1–15. https://doi.org/10.1016/j.isprsjprs.2023.02.007

Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. https://doi.org/10.48550/ARXIV.1412.6980

Kleerekoper, L., Van Esch, M., & Salcedo, T. B. (2012). How to make a city climate-proof, addressing the urban heat island effect. Resources, Conservation and Recycling, 64, 30–38. https://doi.org/10.1016/j.resconrec.2011.06.004

Livesley, S. J., McPherson, E. G., & Calfapietra, C. (2016). The Urban Forest and Ecosystem Services: Impacts on Urban Water, Heat, and Pollution Cycles at the Tree, Street, and City Scale. Journal of Environmental Quality, 45(1), 119–124. https://doi.org/10.2134/jeq2015.11.0567

Martins, J. A. C., Nogueira, K., Osco, L. P., Gomes, F. D. G., Furuya, D. E. G., Gonçalves, W. N., Sant’Ana, D. A., Ramos, A. P. M., Liesenberg, V., Dos Santos, J. A., De Oliveira, P. T. S., & Junior, J. M. (2021). Semantic Segmentation of Tree-Canopy in Urban Environment with Pixel-Wise Deep Learning. Remote Sensing, 13(16), 3054. https://doi.org/10.3390/rs13163054

McPherson, E. G. (2010). Selecting Reference Cities for i-Tree Streets. Arboriculture & Urban Forestry, 36(5), 230–240. https://doi.org/10.48044/jauf.2010.031

Miraki, M., Sohrabi, H., Fatehi, P., & Kneubuehler, M. (2021). Individual tree crown delineation from high-resolution UAV images in broadleaf forest. Ecological Informatics, 61, 101207. https://doi.org/10.1016/j.ecoinf.2020.101207

Nowak, D. J., Crane, D. E., & Stevens, J. C. (2006). Air pollution removal by urban trees and shrubs in the United States. Urban Forestry & Urban Greening, 4(3–4), 115–123. https://doi.org/10.1016/j.ufug.2006.01.007

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. https://doi.org/10.48550/ARXIV.1505.04597

Tau, R. (2022, March 8). Cross-Entropy, Negative Log-Likelihood, and All That Jazz. Towards Data Science. https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81

Wang, Z., Fan, C., & Xian, M. (2021). Application and Evaluation of a Deep Learning Architecture to Urban Tree Canopy Mapping. Remote Sensing, 13(9), 1749. https://doi.org/10.3390/rs13091749

Weinstein, B. G., Marconi, S., Bohlman, S., Zare, A., & White, E. (2019). Individual Tree-Crown Detection in RGB Imagery Using Semi-Supervised Deep Learning Neural Networks. Remote Sensing, 11(11), 1309. https://doi.org/10.3390/rs11111309

Zhao, H., Morgenroth, J., Pearse, G., & Schindler, J. (2023). A Systematic Review of Individual Tree Crown Detection and Delineation with Convolutional Neural Networks (CNN). Current Forestry Reports. https://doi.org/10.1007/s40725-023-00184-3


