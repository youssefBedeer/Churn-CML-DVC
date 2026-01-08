# ============================================================
# Import Required Libraries
# ============================================================
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
from imblearn.over_sampling import SMOTE 
from PIL import Image
from joblib import dump

from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from sklearn.metrics import f1_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression


# ============================================================
# 2️⃣ Read Dataset
# ============================================================
data_path = os.path.join(os.getcwd(), 'data', 'dataset.csv')
df = pd.read_csv(data_path)


# ============================================================
# 3️⃣ Basic Data Cleaning
# ============================================================
df= df.drop(columns=["RowNumber", "CustomerId", "Surname"], axis=1)
age_mask = df["Age"] > 80 # return true for age > 80
df = df[~age_mask] # ~ to remove true


# ============================================================
# 4️⃣ Define Features and Target
# ============================================================
X = df.drop(columns=["Exited"])
y = df["Exited"]


# ============================================================
# 5️⃣ Train / Test Split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# ============================================================
# 6️⃣ Feature Categorization
# ============================================================
num_cols = ["CreditScore","Age","Balance","EstimatedSalary"]
categ_cols = ['Gender', 'Geography']
ready_cols = list( set(X_train.columns.tolist()) - set(num_cols) - set(categ_cols) )


# ============================================================
# 7️⃣ Build Preprocessing Pipelines
# ============================================================
num_pipeline = Pipeline(steps=[
    ('selector', DataFrameSelector(num_cols)),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categ_pipeline = Pipeline(steps=[
    ('selector', DataFrameSelector(categ_cols)),
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(drop='first', sparse_output=False))
])

ready_pipeline = Pipeline(steps=[
    ('selector', DataFrameSelector(ready_cols)),
    ('imputer', SimpleImputer(strategy='most_frequent'))
])


# ============================================================
# 8️⃣ Combine Pipelines
# ============================================================
all_pipeline = FeatureUnion(transformer_list=[
        ('numerical', num_pipeline),
        ('categorical', categ_pipeline),
        ('ready', ready_pipeline)
    ])

## apply
X_train_final = all_pipeline.fit_transform(X_train)
X_test_final = all_pipeline.transform(X_test)


# ============================================================
# 9️⃣ Handle Class Imbalance
# ============================================================
vals_count = 1 - (np.bincount(y_train) / len(y_train))
vals_count = vals_count / np.sum(vals_count)
dict_weight = {}
for i in range(2):
    dict_weight[i] = vals_count[i]
    
## SMOTE
over = SMOTE(sampling_strategy=0.7)
X_train_resampled, y_train_resampled = over.fit_resample(X_train_final, y_train)


# ============================================================
# 🔟 Initialize Metrics File
# ============================================================
with open("metrics.txt", "w") as f:
    pass


# ============================================================
# 1️⃣1️⃣ Define Training Function
# ============================================================
def train_model(X_train, y_train, plot_name="", class_weight=None):
    global model_name
    
    # model = RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42, class_weight=class_weight) 
    
    model = LogisticRegression(class_weight=class_weight, penalty="l2")
    model_name = model.__class__.__name__
    
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test_final)
    
    ## evaluate using f1_score 
    score_train = f1_score(y_train, y_pred_train)
    score_test = f1_score(y_test, y_pred_test)

    ## Plot the confusion matrix 
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, cbar=False, fmt='.2f', cmap='Blues')
    plt.title(f"{plot_name}")
    plt.xticks(ticks=np.arange(2) + 0.5 , labels=[False, True])
    plt.yticks(ticks=np.arange(2) + 0.5 , labels=[False, True])
    
    ## save plot locally 
    plt.savefig(f"{plot_name}.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    ## write scores into metrics.txt 
    with open("metrics.txt", "a") as f:
        f.write(f"{model_name} {plot_name}\n")
        f.write(f"F1-Score of Training is: {score_train*100:.2f} %\n")
        f.write(f"F1-Score of Testing is : {score_test*100:.2f} %\n")
        f.write("----"*10 + "\n")
        
    ## save model
    os.makedirs( os.path.join( os.getcwd(), "models" ), exist_ok=True )
    dump(model, os.path.join(os.getcwd(), "models", f'{model_name}-{plot_name}.pkl'))
    return True


# ============================================================
# 1️⃣2️⃣ Train Models Under Different Scenarios
# ============================================================
train_model(X_train_final, y_train, "without-imbalance", None)
train_model(X_train_final, y_train, "with-class-weights", class_weight=dict_weight)
train_model(X_train_resampled, y_train_resampled, "with-SMOTE", None)


# ============================================================
# 1️⃣3️⃣ Combine Confusion Matrices
# ============================================================
confusion_matrix_paths = [f'./without-imbalance.png', f'./with-class-weights.png', f'./with-SMOTE.png']

plt.figure(figsize=(15,5))
for i, path in enumerate(confusion_matrix_paths, 1):
    img = Image.open(path)
    plt.subplot(1,len(confusion_matrix_paths), i)
    plt.imshow(img)
    plt.axis('off')
    
plt.suptitle(model_name, fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'conf_matrix.png', bbox_inches='tight', dpi=300)


# ============================================================
# 1️⃣4️⃣ Cleanup
# ============================================================
for path in confusion_matrix_paths:
    os.remove(path)