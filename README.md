# **Heart Disease Prediction with Keras and Hyperparameter Tuning**

### **Objective**
The goal of this project is to build a binary classification model using Keras to predict heart disease based on health indicators. The model is designed to handle class imbalance, optimize performance through hyperparameter tuning using Keras Tuner, and evaluate its effectiveness using metrics such as accuracy, ROC-AUC curves, and TensorBoard visualizations.

### **Dataset Description**
The dataset used for this task is `heart_disease_health_indicators.csv`, which includes various health indicators and a target variable indicating whether the patient has had heart disease or an attack.

- **Number of Samples**: `253,661`
- **Number of Features**: `22` (21 features + 1 target variable)
- **Target Variable**: `HeartDiseaseorAttack` (binary classification: `0` - No heart disease, `1` - Heart disease)

### **Features Overview**
The dataset consists of the following columns:

| #  | Column                     | Non-Null Count | Data Type |
|----|----------------------------|----------------|-----------|
| 0  | HeartDiseaseorAttack       | 253,661        | int64     |
| 1  | HighBP                     | 253,661        | int64     |
| 2  | HighChol                   | 253,661        | int64     |
| 3  | CholCheck                  | 253,661        | int64     |
| 4  | BMI                        | 253,661        | int64     |
| 5  | Smoker                     | 253,661        | int64     |
| 6  | Stroke                     | 253,661        | int64     |
| 7  | Diabetes                   | 253,661        | int64     |
| 8  | PhysActivity               | 253,661        | int64     |
| 9  | Fruits                     | 253,661        | int64     |
| 10 | Veggies                    | 253,661        | int64     |
| 11 | HvyAlcoholConsump          | 253,661        | int64     |
| 12 | AnyHealthcare              | 253,661        | int64     |
| 13 | NoDocbcCost                | 253,661        | int64     |
| 14 | GenHlth                   | 253,661        | int64     |
| 15 | MentHlth                   | 253,661        | int64     |
| 16 | PhysHlth                   | 253,661        | int64     |
| 17 | DiffWalk                   | 253,661        | int64     |
| 18 | Sex                        | 253,661        | int64     |
| 19 | Age                        | 253,661        | int64     |
| 20 | Education                  | 253,661        | int64     |
| 21 | Income                     | 253,661        | int64     |

This dataset provides a comprehensive overview of various health indicators that can be used to build a predictive model for heart disease.


### Steps to Run the Code in Google Colab
1. Upload the orbit.csv file to your Google Colab environment.
2. Copy and paste the provided code into a new Colab notebook.
3. Adjust the file path in the pd.read_csv() function to match the location of your orbit.csv file.
4. Run the code cell by cell to execute the neural network model training and evaluation.

### Dependencies
- NumPy: !pip install numpy
- Pandas: !pip install pandas
- TensorFlow: !pip install tensorflow
- Matplotlib: !pip install matplotlib
- Scikit-learn: !pip install scikit-learn
- Keras: !pip install keras

### Dependencies
## The following Python libraries are required to run the code:
  - pandas: Data manipulation and analysis (pip install pandas)
  - numpy: Numerical computations (pip install numpy)
  - scikit-learn: Machine learning and data preprocessing tools (pip install scikit-learn)
  - imbalanced-learn: Handling imbalanced datasets using techniques like SMOTE (pip install imbalanced-learn)
  - keras: Neural networks and deep learning framework (pip install keras)
  - keras-tuner: Hyperparameter tuning for Keras models (pip install keras-tuner)
  - tensorboard: Visualizing model performance (pip install tensorboard)
  - matplotlib: Data visualization (pip install matplotlib)
  - seaborn: Statistical data visualization (pip install seaborn)
pip install --upgrade scikit-learn imbalanced-learn keras keras-tuner tensorboard matplotlib seaborn

## Directory Structure
heart-disease-prediction/
│
├── heart_disease_prediction.ipynb  # Main Jupyter Notebook file
├── heart_disease_health_indicators.csv  # Dataset file
├── logs/  # TensorBoard log files
│   └── fit/  
├── README.md  # Documentation

## Project Workflow
# 1. Data Preprocessing:
  Load the dataset and split it into features (X) and target variable (y).
  Handle class imbalance using techniques such as SMOTE and RandomUnderSampler.

# 2. Model Building:
  Build a Sequential model using Keras with fully connected layers and a Dropout layer to prevent overfitting.
  Compile the model with the Adam optimizer and binary cross-entropy loss function.

# 3. Model Training:
  Train the model using the training set and validate on the validation set.
  Use EarlyStopping to prevent overfitting and TensorBoard to visualize the training progress.

# 4. Evaluation:
  Evaluate the model on the test set using metrics like accuracy, confusion matrix, and ROC-AUC curves.
  Plot the ROC curve to visualize model performance.

# 5. Hyperparameter Tuning:
  Use Keras Tuner to find the optimal architecture and hyperparameters (number of layers, units, and learning rate).

