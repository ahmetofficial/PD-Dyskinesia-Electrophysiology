"""
Statistic utilisation functions
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests

def set_up_mixedlm_with_interaction(dataset, response_variable, independed_variable, block_variable, random_effect, random_intercept, random_slope, REML_state):
    if((random_intercept==True) & (random_slope==False)):
        model = smf.mixedlm(f"" +response_variable + " ~ " + independed_variable + " * " +  block_variable, dataset, 
                            groups=dataset[random_effect], re_formula="~1").fit(reml=REML_state)
    elif((random_intercept==True) & (random_slope==True)):
        model = smf.mixedlm(f"" +response_variable + "~ " + independed_variable + " * " +  block_variable, dataset, 
                            groups=dataset[random_effect], re_formula="~1+"+independed_variable).fit(reml=REML_state)
    return model
    
def run_LMM_model_with_interaction(dataset, response_variables, independed_variable, block_variable, 
                                   random_effect, random_intercept, random_slope):

    groups   = sorted(dataset[independed_variable].unique()) 
    segments = sorted(dataset[block_variable].unique()) 
    
    print("Linear Mixed Effect Model with Interaction Started")
    print("------------------------------------------------------------------")
    print("--> independent variable : " + str(independed_variable) + " [" +  ", ".join(groups) + "] ")
    print("--> block variable       : " + str(block_variable) + " [" +  ", ".join(segments) + "] ")
    print("--> interaction          : " + str(independed_variable) + " * " + str(str(block_variable)))
    print("--> random effect        : " + str(random_effect))
    print("--> random intercept     : " + str(random_intercept))
    print("--> random slope         : " + str(random_intercept))
    print("------------------------------------------------------------------")
    
    reference_severity = groups[0]
    reference_segment  = segments[0]
    results            = pd.DataFrame(columns=["feature", "reference_severity", "reference_segment", "comparison_severity", "comparison_segment", 
                                               "model", "coefficient","p_value"])

    for feature in response_variables:
        print("--> response variable    : " + feature)
        model = set_up_mixedlm_with_interaction(dataset=dataset, response_variable=feature, 
                                                independed_variable=independed_variable, block_variable=block_variable, random_effect=random_effect, 
                                                random_intercept=random_intercept, random_slope=random_slope, REML_state=True)
        
        df            = pd.DataFrame(model.pvalues).reset_index()
        df["term"]    = df["index"]
        df["p_value"] = df[0]
        df            = df[["term","p_value"]]
        
        for index, row in df.iterrows():
            if(":" in row.term):
                new_row                        = {} 
                new_row["feature"]             = feature
                new_row["reference_severity"]  = reference_severity
                new_row["reference_segment"]   = reference_segment
                new_row["comparison_severity"] = row.term.split(":")[0].split(".")[1][0:-1]
                new_row["comparison_segment"]  = row.term.split(":")[1].split(".")[1][0:-1]
                new_row["model"]               = model
                new_row["coefficient"]         = model.params[index]
                new_row["p_value"]             = row.p_value
                results.loc[len(results)]      = new_row 

    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    return results
    

def set_up_mixedlm(dataset, response_variable, independed_variable, random_effect, random_intercept, random_slope, REML_state):
    if((random_intercept==True) & (random_slope==False)):
        model = smf.mixedlm(f"" +response_variable + "~ " + independed_variable, dataset, 
                            groups=dataset[random_effect], re_formula="~1").fit(reml=REML_state)
    elif((random_intercept==True) & (random_slope==True)):
        model = smf.mixedlm(f"" +response_variable + "~ " + independed_variable, dataset, 
                            groups=dataset[random_effect], re_formula="~1+"+independed_variable).fit(reml=REML_state)
    return model

def run_LMM_model(dataset, response_variables, independed_variable, random_effect, random_intercept, random_slope, correction_method):

    df_LMM_results         = pd.DataFrame(columns=["feature", "group_1", "group_2", "coefficient", "model", "converged", "p_value"]) 
    groups                 = sorted(dataset[independed_variable].unique())                      # the distinct group categories
    reference_group        = sorted(dataset[independed_variable].unique())[0]                   # the reference group based on alphabetical order
    no_of_remaining_groups = len(dataset[independed_variable].unique()) - 1                     # the number of groups that can be compared with the reference group
    
    print("Linear Mixed Effect Model Started")
    print("------------------------------------------------------------------")
    print("--> independent variable : " + str(independed_variable))
    print("--> groups               : " + ", ".join(groups))
    print("--> random effect        : " + str(random_effect))
    print("--> random intercept     : " + str(random_intercept))
    print("--> random slope         : " + str(random_intercept))
    print("------------------------------------------------------------------")
    
    for feature in response_variables:
    
        print("--> response variable    : " + feature)
        
        model   = set_up_mixedlm(dataset=dataset, response_variable=feature, independed_variable=independed_variable, 
                                                  random_effect=random_effect, random_intercept=random_intercept, random_slope=random_slope, REML_state=True)
        pvalues = model.pvalues # get uncorrected p-values from the LMM model
        coeffs  = model.params  # get coefficients from the LMM model
        
        for group_i in range(no_of_remaining_groups):
            row                = {}
            row["feature"]     = feature
            row["group_1"]     = reference_group
            row["group_2"]     = groups[group_i+1]
            row["coefficient"] = coeffs[group_i+1]
            pvalue             = pvalues[group_i+1]
            row["p_value"]     = pvalue
            row["model"]       = model
            row["converged"]   = model.converged
    
            if(np.isnan(pvalue) == True): # then there is a big possibility of having collinearity between groups
    
                print("    -> issue: " + reference_group + " vs " + row["group_2"])
                
                # singularity refers to a case where one or more variables in the dataset are perfectly correlated with others. 
                # This results in a design matrix that is not invertible, which can lead to errors when fitting linear mixed effect models.
                singularity_issue = check_singularity_issue(dataset=dataset, independed_variable=independed_variable, 
                                                            response_variable=feature, drop_first=False)
    
                # Variance Inflation Factor (VIF) is used to detect multicollinearity in our LMM model by quantifying how much the variance of 
                # a regression coefficient is inflated due to multicollinearity among the predictors/independent variables. 
                # When all groups are included, their sum sometimes can be perfectly correlated with the intercept (or constant), leading to an "infinitely large" VIF.
                vif_scores = measure_variance_inflation_factor(dataset=dataset, independed_variable="grouping_2", 
                                                               response_variable="post_event_gamma_mean", drop_first=False)
                if(singularity_issue==True):
                    print("        -> warning: some eigenvalues close to zero, indicating a potential singularity issue in covariance matrix.")
                if(np.isinf(vif_scores.vif).any()==True): # check if vif scores go to inf which implies a high degree of multicollinearity
                    print("        -> warning: infinite VIF values, indicating a high degree of multicollinearity leading failed estimation of model parameters")

                print("        -> switching from REML to ML Estimation")

                # try again with fitting LMM with Maximum likelihood estimation instead of Restricted Maximum Likelihood Estimation
                model   = set_up_mixedlm(dataset=dataset, response_variable=feature, independed_variable=independed_variable, 
                                         random_effect=random_effect, random_intercept=random_intercept, random_slope=random_slope, REML_state=False)
                pvalues = model.pvalues # get uncorrected p-values from the LMM model
                coeffs  = model.params  # get coefficients from the LMM model
                row["coefficient"] = coeffs[group_i+1]
                pvalue             = pvalues[group_i+1]
                row["p_value"]     = pvalue
                row["model"]       = model
                row["converged"]   = model.converged

                if(np.isnan(pvalue) == False):
                    print("        -> issue resolved: the result for given two groups was added to the output!")
                    df_LMM_results.loc[len(df_LMM_results)] = row
                else:
                    print("        -> issue unsolved: the result for given two groups was not add to the output!")
            else:
                df_LMM_results.loc[len(df_LMM_results)] = row

    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    
    # apply multiple comparison correction
    df_LMM_results["p_value_corrected"] = multipletests(df_LMM_results.p_value, alpha=0.05, method=correction_method)[1]   
    return df_LMM_results

def apply_multiple_correction(p_values, correction_method):
    return multipletests(p_values, alpha=0.05, method=correction_method)[1]

def interpret_vif(vif):
    if vif < 5:
        return 'low'
    elif 5 <= vif < 10:
        return 'moderate'
    else:
        return 'high'
        
def measure_variance_inflation_factor(dataset, independed_variable, response_variable, drop_first):

    # convert the random effect column to dummy variables (one-hot encoding)
    df_encoded = pd.get_dummies(dataset[[independed_variable,response_variable]], columns=[independed_variable], drop_first=drop_first)
    
    # define the independent variables (the one-hot encoded columns) and the dependent variable
    X = df_encoded.drop(columns=[response_variable]).astype(int)
    y = df_encoded[response_variable]
    
    # adding a constant term helps in understanding the multicollinearity of the model while accounting for the baseline effect
    X = sm.add_constant(X)
    
    # calculate VIF for each variable
    vif_data                   = pd.DataFrame()
    vif_data['feature']        = X.columns
    vif_data['vif']            = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data['interpretation'] = vif_data['vif'].apply(interpret_vif)

    return vif_data


def check_singularity_issue(dataset, independed_variable, response_variable, drop_first):

    # convert the random effect column to dummy variables (one-hot encoding)
    df_encoded = pd.get_dummies(dataset[[independed_variable,response_variable]], columns=[independed_variable], drop_first=drop_first)
    
    # define the independent variables (the one-hot encoded columns) and the dependent variable
    X = df_encoded.drop(columns=[response_variable]).astype(int)
    y = df_encoded[response_variable]
    
    # adding a constant term helps in understanding the multicollinearity of the model while accounting for the baseline effect
    X = sm.add_constant(X)
    
    # check for singularity by examining the rank of the design matrix
    rank        = np.linalg.matrix_rank(X)   
    # check eigenvalues of the correlation matrix
    eigenvalues = np.linalg.eigvals(X.T @ X)
    
    # Check if any eigenvalues are close to zero (indicative of singularity)
    if np.any(np.isclose(eigenvalues, 0)):
        return True
    else:
        return False