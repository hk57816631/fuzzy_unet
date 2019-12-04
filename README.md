<h1>fuzzy_unet</h1>

<p>Note: my codes based on python 3.5, keras 2.1.2, tensorflow 1.9.0, Cuda 9.0, cudnn 7.0.5. Windows and Linux and Max OS are all supported and tested.</p>
<ol>
  <li>Download the dataset. Save the dataset in ./BUS/data2/original/ (original image) and ./BUS/data2/GT/ (ground truth)</li>
  <li>For training, just run the train.py. In console, move in to the root file of my code (where the train.py is located). Type python -M unet to train the original unet. Type python -M fuzzyunet to train the fuzzy unet.</li>
  <li>For testing, there are two ways. First one is testing a single image. Using test.py and giving img_path, and label_path it can output segmentation result of one image. For testing a set of samples, test_path.py is used. img_path, label_path and the txt file containing name list are provided and then the set of samples are segmented and save in result_path.</li>
  <li>Wavelet.py is used to do wavelet transform. Norm.py is used to do histogram normalization. Showmed.py is used to show the intermediate results of fuzzy layer.</li>
  <li>The weights of the network are saved in “model name” + _model_weight.h5</li>
</ol>

