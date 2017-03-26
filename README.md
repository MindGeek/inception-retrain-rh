# inception-retrain
Standalone version of Tensorflow Inception Retrain. Google Codelabs can be found [here](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets).

This example downloads a pre-trained version of the inception model and re-trains the last layers to do a steering prediction model which can be used in roadhackers competition (roadhackers.baidu.com).

### To train the model, run:
bash retrain.sh

### To predict by the model, run:
python test_rh_model.py model/your_saved_model.meta data/testing_dir your_result_h5_file 


