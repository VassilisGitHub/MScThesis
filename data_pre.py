from statsbombpy import sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Suppress all warnings
warnings.filterwarnings("ignore")

def get_df(events_fifa):

    #Get the pass volumns from events df
    pass_vars = [col for col in events_fifa.columns if col.startswith('pass_')]
    
    #Create the pass
    risk_df = events_fifa[pass_vars]

    #Get the df were at least one of the dependent vars is not null
    
    #risk_df = pass_df.dropna(subset=['pass_outcome','pass_shot_assist','pass_goal_assist'], how='all')
    
    #Get the df only with relevant pass_outcome values and the NaN values with | 
    #risk_df = risk_df[risk_df['pass_outcome'].isin(['Incomplete', 'Out', 'Pass Offside','Unknown', 'Injury Clearance']) | risk_df['pass_outcome'].isna()]


    #Get the explanatory vars of interest
    #For now pass_angle is omitted , 'pass_end_location'
    risk_df = risk_df[['pass_outcome', 'pass_shot_assist','pass_goal_assist', 'pass_length' ,'pass_angle', 'pass_height', 'pass_body_part']]

    #print('Number of records before removing NUll explanatory: {:,}'.format(len(risk_df)))


    #Get the df with only the completed explanatory vars
    risk_df= risk_df.dropna(subset=['pass_length', 'pass_angle' , 'pass_height', 'pass_body_part'])
    #print('Number of records after removing NUll explanatory: {:,}'.format(len(risk_df)))

    return risk_df



def data_prep(risk_df):

    #Assign 1 to pass_outcome
    risk_df['pass_outcome'] = risk_df['pass_outcome'].replace(to_replace=['Incomplete', 'Out', 'Pass Offside','Unknown'], value= int(1))

    #Assign 1 to pass_shot_assist
    risk_df['pass_shot_assist'] = risk_df['pass_shot_assist'].replace(to_replace=[True],value= int(1))

    #Assign 1 to pass_goal_assist
    risk_df['pass_goal_assist'] = risk_df['pass_goal_assist'].replace(to_replace=[True],value= int(1))

    #Remove Injury clearance rows
    risk_df = risk_df[risk_df['pass_outcome'] != 'Injury Clearance' ]
    
    #Count risky and not risky passes
    print('Number of risky passes: {:,}'.format(risk_df['pass_outcome'].sum()))
    print('Number of NOT risky passes: {:,}'.format(risk_df['pass_shot_assist'].sum()))
    print('Number of NOT risky passes: {:,}'.format(risk_df['pass_goal_assist'].sum()))

    #Create dependent variable
    #Drop pass_shot_assist	pass_goal_assist and create risk_df where risky pass is outcome = 1
    #and not risky pass is outcome = 0
    risk_df = risk_df.drop(columns=['pass_shot_assist', 'pass_goal_assist'])
    

    risk_df['pass_outcome'] = risk_df['pass_outcome'].fillna(int(0))

    return risk_df

def one_hot(risk_df):
    
    #One Hot Encoding
    categorical_cols = ['pass_height', 'pass_body_part']

    risk_df = pd.get_dummies(risk_df , columns= categorical_cols)

    return risk_df


def get_df_analysis(events_fifa):

    #Assign 1 to pass_outcome
    events_fifa['pass_outcome'] = events_fifa['pass_outcome'].replace(to_replace=['Incomplete', 'Out', 'Pass Offside','Unknown', 'Injury Clearance'], value= int(1))
    events_fifa['pass_outcome'] = events_fifa['pass_outcome'].fillna(int(0))
    
    df_vars = ['pass_outcome','match_id','id', 'play_pattern', 'player', 'position', 'tactics', 'possession_team']

    
    #Create the pass
    df = events_fifa[df_vars]

    #Transform dict tactics to get formations
    for i in range(len(df['tactics'])):
        if isinstance(df['tactics'][i], dict):
            a = df['tactics'][i].get('formation' , 0)
        else: 
            a = 0 
    
        df.at[i , 'tactics'] = a
    
    return df


def evaluation(y_test, logit_predictions_binary):
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, logit_predictions_binary)

    # Calculate evaluation measures
    accuracy = accuracy_score(y_test, logit_predictions_binary)*100
    misclassification_rate = (1 - accuracy/100)*100
    sensitivity = (conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]))*100
    specificity = (conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]))*100
    precision = (conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1]))*100

    accuracy = format(accuracy, '.2f')
    misclassification_rate = format(misclassification_rate, '.2f')
    sensitivity = format(sensitivity, '.2f')
    specificity = format(specificity, '.2f')
    precision = format(precision, '.2f')

    #  Create a dictionary with the results
    results = {
        'Accuracy': accuracy,
        'Misclassification Rate': misclassification_rate,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision
    }

    df_eval = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])

    return df_eval