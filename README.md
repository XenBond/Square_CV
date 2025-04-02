# Code Structure

requirements.txt: the python library needed to run the code
```
src/
    --data.py # torch dataset to read the data
    --EDA.py # exploratory data analysis, including compute the pixel histogram, the mean and standard deviation for normalization, and a trial of random forest classifier.
    --ensemble_model.py # an inference code to run 5-fold ensemble model to do inference on the test dataset.
    --evaluation.py # a code to evaluate the model, record the evaluation metrices, including a confusion matrix, precision and recall for each class, and macro-average precision, recall and F-1 score.
    --model.py # model definition file
    --train.py # the file that defines the training, validation and checkpoint saving.
    --draw_validation_curve.py # code to draw the validation loss during training. 

```

# Usage.
1. run EDA.py to get the mean and standard deviation. 

2. run train.py to train the model. After training, the user can run draw_validation_curve.py to draw the validation loss, and potentially tune the hyperparameters accordingly.

3. run evaluation.py to evaluate the ensemble model

# Error Analysis
Just got 1 minute to check the error images, the false prediction might mostly because they have different background color.
