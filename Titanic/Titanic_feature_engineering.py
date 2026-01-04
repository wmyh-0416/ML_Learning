import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    train = pd.read_csv("/Users/wmyh0416/Desktop/ML_Learning/Titanic/Titanic_datasets/train.csv")
    test = pd.read_csv("/Users/wmyh0416/Desktop/ML_Learning/Titanic/Titanic_datasets/test.csv")

    #Feature engineering
    #Passengers with small to medium family sizes tend to have higher survival rates
    train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
    test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

    train["Title"] = train["Name"].str.extract(r" ([A-Za-z]+)\.",expand=False)
    test["Title"] = test["Name"].str.extract(r" ([A-Za-z]+)\.",expand=False)

    rare_titles = ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"]
    train["Title"] = train["Title"].replace(rare_titles,"Rare")
    test["Title"] = test["Title"].replace(rare_titles,"Rare")

    features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked", #Feature selection: These features are more responsible for the prediction
                "FamilySize","Title"]   # new features after feature engineering!!!
    X = train[features]
    y = train["Survived"]
    X_test = test[features]

    num_cols = ["Pclass","Age","SibSp","Parch","Fare","FamilySize"]
    cat_cols = ["Sex","Embarked","Title"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]),num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols)
        ]
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    print("Validation accuracy:", accuracy_score(y_val, val_pred))

    model.fit(X,y)
    test_pred = model.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": test_pred
    })
    submission.to_csv("submission_feature_engineering.csv", index=False)
    print("Saved submission_feature_engineering.csv")


if __name__ == "__main__":
    main()