## Unsupervised Classification (unsupredictor)

unsupredictor is a library that allows to quickly cluster a given set of datapoints and generate a random forest classifier on the results to easily classify new incoming datapoints.
![](https://cdn-images-1.medium.com/max/640/1*tc8UF-h0nQqUfLC8-0uInQ.gif)
## Installation

prerequisites:
python 3.x , pip3
```bash
pip install -r requirements.txt
```

## Usage
Training on a unsupervised dataset
```python
from unsupredictor import *

predictor = UnsupervisedLearnPredictor(<input_raw_file_path>)
predictor.predict(data)
```
Running inference
```python
from unsupredictor import *

predictor = UnsupervisedLearnPredictor(None)
predictor.load_precomputed_classifier(<model_path>)
predictor.predict(data)
```
## Contributing
Pull requests are very much welcome