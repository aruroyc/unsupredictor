## Unsupervised Classification (unsupredictor)

unsupredictor is a library that allows to quickly cluster a given set of datapoints and generate a random forest classifier on the results to easily classify new incoming datapoints.
![](cluster.gif)
## Installation

prerequisites:
python 3.x , pip3
```bash
pip install -r requirements.txt
```

## Usage

```python
from unsupredictor import *

predictor = UnsupervisedLearnPredictor(<input_raw_file_path>)
predictor.learn_classes_and_save_model()
predictor.predict(data)
```
## Contributing
Pull requests are very much welcome
