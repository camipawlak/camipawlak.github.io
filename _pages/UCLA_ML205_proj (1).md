---
layout: archive
title: "Canopy Cover Segmentation From Aerial Imagery Using Deep Learning"
permalink: /ML205/
---



## Introduction

  Urban trees benefit people's health, happiness and wellbeing by providing tangible
benefits such as reducing the negative impacts from the urban heat island effect and saving on
energy bills by alleviating extreme temperature (Livesly et al., 2016; California Urban Forestry
Act of 1978). In addition to these tangible benefits urban trees provide services such as sequestering carbon, reducing particulate air pollution, increasing property values, supporting job creation and business growth, and improving urban residents mental health (Livesly et al., 2016; Nowak et al., 2006; Kleerekoper et al., 2012; California Urban Forestry Act of 1978). Measuring and managing urban forests to understand and maximize their benefits is an expensive and time intensive process. One way to measure an urban forest is by mapping urban canopy cover.  The California Urban Forestry Act created a goal to achieve a statewide "10-percent increase of tree canopy cover in urban areas by 2035, with priority for increasing tree canopy cover in disadvantaged and low-income communities and low-canopy areas" (California Urban Forestry Act of 1978). To achieve this goal, the state needs an accurate baseline map of canopy cover for California's urban areas.

In urban environments, change in canopy cover happens slowly. Larger trees often do not
increase in size because they are regularly trimmed. Because trees are small when planted,
increases in canopy cover can be hard to detect. Many studies have created tree cover canopy
maps for small areas at high resolutions (using imagery and/or Lidar data from unpiloted aerial
vehicles or planes (Braga et al., 2020; Codemo et al., 2022; Miraki et al., 2021; Zhao et al.,
2023)), large areas with a combination of high-resolution data sources, making it difficult to
create repeat datasets to examine changes over time (Guo et al., 2023), or large repeatable areas
at low resolutions (using imagery from satellites such as Landsat or Sentinel (Dewitz & U.S.
Geological Survey, 2019)). Our study plans to address ways to quantify urban forest change over
time at large scales, and the accuracy of measuring that change. To achieve that goal, we first need to develop methods for quantifying canopy cover that we can apply confidently over multiple years of imagery. In our larger study, we plan to create canopy cover maps for multiple years using NAIP aerial imagery. For this study, we start with developing a neural network that segments tree canopy out from 2020 NAIP imagery in the southern California climate zone.

# Data
Our training data consists of 448x448 tifs of NAIP imagery. We clipped these images out of the BLAH BLAH DATASET ON ARCGIS ONLINE. The imagery has four bands: red, blue, green, and near-infrared. We added a fifth band which is well known and commonly used vegetation index called a NDVI which was calculated with this formula: (Red-NIR)/(Red+NIR+1e-8). The 1e-8 was added to prevent division by 0. The ending NDVI ranges from -1 to 1. For each band, we took the maximum and minimum value for the entire state dataset, and used it to standardize our data. We used maximum-minimum normalization on every original band in every 448x448 training sample. The values for each band range from 0-225. For the NDVI, we converted the scale from -1 to 1 to 0-255 by adding 1 and multiplying by 2/255. For each image, we had an accompanied mask dataset which was created by using USGS LiDAR to create chms, and selecting only areas on the chm above 2 meters and with an NDVI threshold of over 0.4. This automated way to create training data has shown some success in previous studies. We split out data into 80% training data, 20% test data, and 20% evaluation data. Due to processing times, we later reduced the number of images in those categories to:

Figure 1: Sample NAIP Imagery

# Modeling
blah
Figure 2: Model Structure

# Results
blah
Figure 3: Table of results
Figure 4: Example Outputs

# Discussion
blah

# Instructions for running the code
Blah

# References
Blah

