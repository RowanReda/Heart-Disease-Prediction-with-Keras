# **Heart Disease Prediction with Keras and Hyperparameter Tuning**

## **Objective**
The goal of this project is to build a binary classification model using Keras to predict heart disease based on health indicators. The model is designed to handle class imbalance, optimize performance through hyperparameter tuning using Keras Tuner, and evaluate its effectiveness using metrics such as accuracy, ROC-AUC curves, and TensorBoard visualizations.

## **Dataset Description**
The dataset used for this task is `heart_disease_health_indicators.csv`, which includes various health indicators and a target variable indicating whether the patient has had heart disease or an attack.

- **Number of Samples**: `253,680`
- **Number of Features**: `22` (21 features + 1 target variable)
- **Target Variable**: `HeartDiseaseorAttack` (binary classification: `0` - No heart disease, `1` - Heart disease)

### **Features Overview**
Some of the features included are:
- `BMI`: Body Mass Index
- `HighBP`: High Blood Pressure
- `Smoker`: Whether the person is a smoker
- `Diabetes`: Whether the person has diabetes
- ... and many other health indicators.

## Steps to Run the Code in Google Colab
1. Upload the orbit.csv file to your Google Colab environment.
2. Copy and paste the provided code into a new Colab notebook.
3. Adjust the file path in the pd.read_csv() function to match the location of your orbit.csv file.
4. Run the code cell by cell to execute the neural network model training and evaluation.

## Dependencies
- NumPy: !pip install numpy
- Pandas: !pip install pandas
- TensorFlow: !pip install tensorflow
- Matplotlib: !pip install matplotlib
- Scikit-learn: !pip install scikit-learn
- Keras: !pip install keras

## Dependencies
# The following Python libraries are required to run the code:
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

# Directory Structure
heart-disease-prediction/
│
├── heart_disease_prediction.ipynb  # Main Jupyter Notebook file
├── heart_disease_health_indicators.csv  # Dataset file
├── logs/  # TensorBoard log files
│   └── fit/  
├── README.md  # Documentation

# Project Workflow
1. Data Preprocessing:
  Load the dataset and split it into features (X) and target variable (y).
  Handle class imbalance using techniques such as SMOTE and RandomUnderSampler.

3. Model Building:
  Build a Sequential model using Keras with fully connected layers and a Dropout layer to prevent overfitting.
  Compile the model with the Adam optimizer and binary cross-entropy loss function.

3. Model Training:
  Train the model using the training set and validate on the validation set.
  Use EarlyStopping to prevent overfitting and TensorBoard to visualize the training progress.

4. Evaluation:
    Evaluate the model on the test set using metrics like accuracy, confusion matrix, and ROC-AUC curves.
  Plot the ROC curve to visualize model performance.

5.Hyperparameter Tuning:
  Use Keras Tuner to find the optimal architecture and hyperparameters (number of layers, units, and learning rate).

