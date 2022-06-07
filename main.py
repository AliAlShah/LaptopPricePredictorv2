import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('laptops.csv', encoding="latin-1")
df.dropna()

df = df[["Company", "Inches", "Ram", "OpSys", "Weight", "Price_euros"]]
print(df.info())

def cat_to_num(*args):
    def convert(parameter):
        x = list(df[parameter].unique())
        cat_num_dict = {}
        for u in x:
            cat_num_dict[u] = x.index(u)
        
        print(cat_num_dict)

        new_list = []
        for i in df[parameter]:
            new_list.append(cat_num_dict[i])

        df[parameter] = new_list

    for i in args:
        convert(i)

def remove_kg():
    new_list = []
    for i in df["Weight"]:
        new_list.append(float(i[:-2]))
    df["Weight"] = new_list
remove_kg()

def remove_gb():
    new_list= []
    for i in df["Ram"]:
        new_list.append(float(i[:-2]))
    df["Ram"] = new_list
remove_gb()
cat_to_num("Company", "OpSys")
print(df.info())

x = df.drop(["Price_euros"], axis=1)
y = df["Price_euros"]


def train():
    best = 0
    for i in range(10000):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        model = LinearRegression()
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        print(best)
        if score > best:
            best = score
            with open("model.pickle", "wb") as f:
                pickle.dump(model, f)

#"Company", "Inches", "Ram", "OpSys", "Weight", "Price_euros"