# Exploring Methods of Image Classification: CIS 4930 @ FSU (Data Mining) Final Project
### Group Members
- Ben Bao
- Cameron Ball
- John Fleming
- Steven Arteaga


## Dependencies
1. Code tested on Python 3.9. While the code may work on other versions, we cannot guarantee that functionality.
2. For `Predictions.py`:
    - Tabulate
      - `python3.9 -m pip install tabulate`
    - joblib
      - `python3.9 -m pip install joblib`
    - NumPy
      - `python3.9 -m pip install numpy`
    - matplotlib
      - `python3.9 -m pip install matplotlib`
    - seaborn
      - `python3.9 -m pip install seaborn`
    - pandas
      - `python3.9 -m pip install pandas`
    - tensorflow
      - `python3.9 -m pip install tensorflow`
3. For `DecisionTree.py`:
    - joblib
      - `python3.9 -m pip install joblib`
    - pandas
      - `python3.9 -m pip install pandas`
    - scikit-learn
      - `python3.9 -m pip install scikit-learn`
4. For `MLModelTrain.py`:
    - tensorflow
      - `python3.9 -m pip install tensorflow`
    - matplotlib
      - `python3.9 -m pip install matplotlib`
5. For `MobileNet.py`:
    - tensorflow
      - `python3.9 -m pip install tensorflow`
    - matplotlib
      - `python3.9 -m pip install matplotlib`
    - keras
      - `python3.9 -m pip install keras`
6. For `SVM.py`:
    - scikit-image
      - `python3.9 -m pip install scikit-image`
    - NumPy
      - `python3.9 -m pip install numpy`
    - pandas
      - `python3.9 -m pip install pandas`

## Usage
1. Download the dependencies if you do not already have them for Python 3.9.
2. Run `Predictions.py` to run the tests, get the model accuracies, and print the final results. To run it, you need a file that is too large to store on GitHub.
    1. `cd` into `src/`, then run `python3.9 MLModelTrain.py`. This will generate the `MLModelVGG16.h5` file.
    2. If you receive errors about "local issuer certificates", you need to run `Install Certificates.command` that was given to you with your Python 3.9 installation.
3. After `MLModelVGG16.h5` is generated, you can run `Predictions.py`. If you still want want to run the other model trainings from scratch, `cd` into `src/`, then run `python3.9 <file>`.
