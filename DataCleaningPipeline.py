# -*- coding: utf-8 -*-
"""
Trevor Smith
11/5/20

This script contains the code for the full data cleaning pipeline developed in
"Data Cleaning Pipeline.ipynb". We will import the fn RunPipeline to do our
data cleaning in one line of code.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.imput import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


"""
create a transformer class we can use to add our new attributes
"""
# note: do not add *args or **kwargs for BaseEstimator
# this allows us to use get_params and set_params
# TransformerMixin will automatically allow us to call fit_transform() 
# as long as we define fit() and transform()
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X):
        return self # nothing else to do
    def transform(self, X):
        """
        Since this is the first transformation in our pipeline, we should expect
        a pandas DataFrame for X
        
        returns a numpy array
        """
        # the column indices we will use to make new attributes
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

        rooms_per_household = X[:,rooms_ix]/X[:,households_ix]
        bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
        population_per_household = X[:,population_ix]/X[:,households_ix]
        
        return np.c_[X, rooms_per_household, bedrooms_per_room, population_per_household]

"""
Define a function we can import anywhere in our project and allows us
to run our data through the full pipeline
"""
def RunPipeline(X):
    """
    Function to prepare our data set for an ML algorithm
    See "Data Cleaning Pipeline.ipynb" for details on this process
    
    Parameters
    ----------
        X : pandas DataFrame containing the attributes (but no labels)
    Returns
    -------
        numpy ndarray containing cleaned attributes
        columns: ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                  'total_bedrooms', 'population', 'households', 
                  'median_income', 'rooms_per_household', 'bedrooms_per_room', 
                  'population_per_household', '<1H OCEAN', 'INLAND', 'ISLAND', 
                  'NEAR BAY', 'NEAR OCEAN']

    """
    
    # start with the transformations on the numerical attribs only
    num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
    ])
    
    # just need one transformation for the categorical attribs
    cat_pipeline = OneHotEncoder()
    
    # column indices for numerical vs categorical data
    # we expect all numerical + 1 categorical column at the end
    num_cols = list(range(len(X)-1))
    cat_cols = [len(X)-1]
    
    # define full pipeline of column specific transformations
    full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
    ])
    
    # run the full set of transformations and return the cleaned data
    return full_pipeline.fit_transform(X)
    