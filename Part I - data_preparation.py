import numpy as np
import pandas as pd
import re
from sklearn import linear_model, tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import missingno as mno

classes = {
    "German Styles" : ["Oktoberfest/Märzen","Altbier", "Dark Mild", "Doppelbock","Dusseldorf Altbier", "Eisbock", "Festbier", "Kölsch", "Märzen", "Mild", "Roggenbier","Weizen/Weissbier"],
    "Belgian and French Styles" : ["Straight (Unblended) Lambic","Saison","Belgian Blond Ale", "Belgian Dubbel", "Belgian Specialty Ale", "Bière de Garde", "Flanders Brown Ale/Oud Bruin", "Flanders Red Ale", "Fruit Lambic", "Gueuze", "Lambic", "Oud Bruin", "Saison Straight (Unblended) Lambic", "Trappist Single"],
    "Ciders and Meads": ["Cyser (Apple Melomel)","Apple Wine", "Common Cider", "Cyster (Apple Melomel)", "Dry Mead", "English Cider", "French Cider", "Metheglin", "New England Cider", "Open Category Mead", "Other Fruit Melomel", "Pyment (Grape Melomel)", "Semi-Sweet Mead", "Sweet Mead", "Traditional Perry"],
    "Dark Ales": ["Tropical Stout","Brown Porter","Dry Stout","American Brown Ale", "American Porter", "American Stout", "Baltic Porter", "British Brown Ale", "Brow Porter", "English Porter","Foreign Extra Stout", "Irish Extra Stout", "Irish Stout", "London Brown Ale", "Northern English Brown", "Oatmeal Stout", "Robust Porter", "Russian Imperial Stout", "Southern English Brown", "Sweet Stout", "Tropical Stou"],
    "Pale Ales": ["Ordinary Bitter","Standard/Ordinary Bitter","Irish Red Ale","Zombie Dust Clone - EXTRACT","American Amber Ale", "American IPA", "American Pale Ale", "Belgian Pale Ale", "Blonde Ale", "British Golden Ale", "Double IPA", "English IPA", "Imperial IPA", "Specialty IPA: Belgian IPA", "Specialty IPA: Black IPA", "Specialty IPA: Brown IPA", "Specialty IPA: Red IPA", "Specialty IPA: Rye IPA", "Specialty IPA: White IPA"],
    "Hybrid and Specialty": ["Vanilla Cream Ale","Alternative Grain Beer", "Alternative Sugar Beer", "Braggot", "Brett Beer", "California Common", "Gose", "Mixed-Fermentation Sour Beer", "Sahti", "Wild Specialty Beer"],
    "Lagers and Bocks": ["International Dark Lager","International Dark Lager ","International Amber Lager","Czech Premium Pale Lager","International Pale Lager","American Light Lager","Traditional Bock","Bohemian Pilsener","American Lager", "American Light Lager Bohemian Pilsner", "California Common Beer", "Classic American Pilsner", "Classic Rauchbier", "Classic Style Smoked Beer", "Cream Ale", "Czech Amber Lager", "Czech Dark Lager", "Czech Pale Lager", "Czech premium Pale Lager", "Dark American Lager", "Dortmunder Export", "Dunkles Bock", "German Helles Exportbier", "German Leichtbier", "German Pils", "German Pilsner (Pils)", "Helles Bock", "International Pales Lager", "Kellerbier: Amber Kellerbier", "Kellerbier: Pale Kellerbier", "Kentucky Common", "Light American Lager", "Maibock/Helles Bock", "Munich Dunkel", "Munich Helles", "North German Altbier", "Piwo Grodziskie", "Pre-Prohibition Lager", "Pre-Prohibition Porter", "Premium American Lager", "Rauchbier", "Schwarzbier", "Scottish Export", "Scottish Export 80/-", "Scottish Heavy", "Scottish Heavy 70/-", "Scottish Light", "Scottish Light 60/-", "Standard American Lager", "Vienna Lager"],
    "Seasonal and Specialty": ["Spice  Herb  or Vegetable Beer","Southern Tier Pumking clone","Australian Sparkling Ale", "Autumn Seasonal Beer", "Clone Beer", "Experimental Beer", "Fruit and Spice Beer", "Fruit Beer", "Fruit Cider", "Holiday/Winter Special Spiced Beer", "Lichtenhainer", "Mixed-Style Beer", "Other Smoked Beer", "Other Specialty Cider or Perry", "Specialty Beer", "Specialty Fruit Beer", "Specialty Smoked Beer", "Specialty Wood-Aged Beer", "Winter Seasonal Beer", "Wood-Aged Beer"],
    "Strong Ales": ["Special/Best/Premium Bitter","American Barleywine", "American Strong Ale", "Belgian Dark Strong Ale", "Belgian Golden Strong Ale", "Belgian Tripel", "Best Bitter", "British Strong Ale", "English Barleywine", "Extra Special/Strong Bitter (ESB)", "Imperial Stout", "Old Ale", "SPecial/Best/Premium Bitter", "Strong Bitter", "Strong Scotch Ale", "Wee Heavy"],
    "Wheat and Rye Beers": ["Dunkles Weissbier","Wheatwine","Witbier","American Wheat Beer", "American Wheat or Rye Beer", "Berliner Weisse", "Dunkelweizen", "Roggenbier (German Rye Beer)", "Weissbier", "Weizenbock", "WheatWine", "Wittbie"]
}

def findclass(subgenre):
    for key, values in classes.items():
        if subgenre in values:
            return key
    return 'N/A'

def standardizing(df, column):
    mean = df[column].mean()
    std = df[column].std()
    standardized_column = (df[column] - mean) / std
    return standardized_column

def nominal_to_numerical(df, column, method):
    if method == 1:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
    elif method == 2:
        df = pd.get_dummies(df, columns=[column], prefix=[column])
        for col in df:
            if column in col:
                df[col] = df[col].astype(int)
    return df


# Loading dataset
df = pd.read_csv('recipeData.csv',encoding='ISO-8859-1')

# DATA PREPARATION
# dropping the columns, we do not want to take into account
df = df.drop(columns=['PrimingMethod','PrimingAmount','UserId','URL','PitchRate'])

# # Handling missing data in BoilGravity
# boilgravity_mean = df['BoilGravity'].mean(numeric_only=True)
# df['BoilGravity'] = df['BoilGravity'] .fillna(boilgravity_mean)

# # Adding Class Attribute - GENRE
# df['Genre'] = ""
# df = df.dropna(subset=['Style'])


# # Assigning appropriate value to Genre attribute based on its subgenre
# for index, row in df.iterrows():
#     df.loc[index, 'Genre'] = findclass(row['Style'])


# # missing_columns -> columns with missing data that will be handled
# missing_columns = ["MashThickness","PrimaryTemp"]
df.drop(columns=["BeerID", "Name", "StyleID"], inplace=True)
# print(df.dtypes)

numeric_df = df.select_dtypes(include=[np.number])
print(numeric_df.dtypes)

ranges = numeric_df.max() - numeric_df.min()

# Create the bar plot
ranges.plot(kind='bar')
plt.title('Range of Values for Each Numeric Column')
plt.xlabel('Columns')
plt.ylabel('Range')
plt.show()


# # Standardizing numerical attributes (z-score)
# for column in df.columns:
#     if column not in missing_columns and column not in ["SugarScale", "BrewMethod","Genre","Style"]:
#             df[column] = standardizing(df, column)

# # converting nominal attributes into numerical
# dfa = nominal_to_numerical(df,"SugarScale",2)
# dfa = nominal_to_numerical(dfa,"BrewMethod",1)
# dfa = nominal_to_numerical(dfa, "Genre",1)


# # using Deterministic Regression Imputation for handling missing data
# for feature in missing_columns:
#     dfa[feature + '_imp'] = dfa[feature]     
#     parameters = list(set(dfa.columns) - set(missing_columns) - set(column+'_imp' for column in missing_columns) - set(["Style"]))
#     dr = dfa.dropna(subset=[feature + '_imp'])
#     # Create a Linear Regression model to estimate the missing data
#     model = linear_model.LinearRegression()
#     model.fit(X = dr[parameters], y = pd.DataFrame(dr[feature + '_imp']))
#     #observe that I preserve the index of the missing data from the original dataframe
#     missing_values = dfa[feature+ '_imp'].isnull()
#     if missing_values.any():
#         predictions = model.predict(dfa.loc[missing_values, parameters])
#         dfa.loc[missing_values, feature] = predictions

# dfa = dfa.drop(columns={'MashThickness_imp','PrimaryTemp_imp'})


# # standardizing columns that had missing values before
# for column in missing_columns:
#     df[column] = dfa[column]
#     df[column] = standardizing(df, column)

# cols = list(df.columns.values)


# # Dropping columns that we believe is irrelevant for classification
# if df[['MashThickness', 'PrimaryTemp']].isnull().any().any():
#     df.drop(columns=['MashThickness', 'PrimaryTemp'], inplace=True)

# df.to_csv('modified.csv', index=False)

