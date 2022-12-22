import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import median_abs_deviation
from sklearn.neighbors import LocalOutlierFactor

class OutlierMethod:
    def iqr_method(self, df):
        Q1,Q3 = np.percentile(df, [25,75])

        IQR = Q3 - Q1

        lcl = Q1 - (1.5 * IQR)

        ucl = Q3 + (1.5 * IQR)

        # Get outliers
        outlier = df[(df < lcl) | (df > ucl)]
        #print(outlier)

        # Get inlier
        # i.e cells which are not outliers
        inlier = df[df >= lcl]
        inlier = inlier[inlier <= ucl].dropna()

        # Get both outlier and inlier
        outlier_and_inlier = df

        return (outlier, inlier, outlier_and_inlier)
    
    def zscore_method(self, df):
        median = np.median(df)
        rz_score = np.abs(df - median)/median_abs_deviation(df, scale='normal')
        
        # Get outliers
        outlier = rz_score[np.abs(rz_score) >= 3.5]
        
        # Get inlier
        inlier = df[np.abs(rz_score) < 3.5].dropna()
        
        # Get both outlier and inlier
        outlier_and_inlier = df
        
        return(outlier, inlier, outlier_and_inlier)
    
    def lof_method(self, df):
        X = df.values
        
        # Create a copy of dataframe and get column name(s)
        temp1 = df.copy()
        index_name = temp1.index.names
        
        # Reset index of dataframe to numeric range (i.e, 0 to length of dataframe minus 1)
        df.reset_index(level=index_name, drop=True, inplace=True)
        
        temp2 = df.copy()
        
        # fit the model for outlier detection (default)
        lof = LocalOutlierFactor(n_neighbors=2)
        
        # (when LOF is used for outlier detection, the estimator has no predict,
        # decision_function and score_samples methods).
        y_pred = lof.fit_predict(X)
        
        # Mask y_pred to dataframe where -1
        mask = df[y_pred == -1]
        
        # Create a dataframe, drop the index of the original dataframe and set all the records to null values
        out_table = df.copy().drop(columns=index_name, errors='ignore')
        out_table.iloc[:, :] = np.NaN
        
        # Replace out_table with mask where rows (of the two dataframes) match
        for idx in mask.index.unique():
            out_table.loc[idx] = mask.loc[idx]
        
        # Set index of the out_table to index of the original dataframe
        out_table.index = temp1.index
        
        # Get outlier, which is only one column. Other columns are dropped, then convert from series to dataframe
        outlier = out_table.iloc[:, 0].to_frame()
        
        # Get inlier
        inlier = temp1[y_pred == 1]
        
        # Get both outlier and inlier
        outlier_and_inlier = temp1
        
        return (outlier, inlier, outlier_and_inlier)
    
class Analytics:
    def outlier_rate(self, outliers_alone):
        
        outlier_rates = []
        
        for outlier_df in outliers_alone:
            outlier_count = len(outlier_df) - outlier_df.isnull().sum()
            perc_out = (100 * outlier_count) / len(outlier_df)
            
            outlier_rates.append(float(perc_out))
        
        df_outlier_rates = pd.DataFrame({'outlier %': outlier_rates})
        
        return df_outlier_rates
    
    def get_pvalue(self, inliers, outliers_and_inliers):
        pvalues = []
        
        for idx, _ in enumerate(outliers_and_inliers):
            group_ttest1 = (list(outliers_and_inliers[idx].iloc[:, -1]))
            group_ttest2 = (list(inliers[idx].iloc[:, -1]))
            
            if max(np.var(group_ttest1), np.var(group_ttest2))/min(np.var(group_ttest1), np.var(group_ttest2)) < 4:
                equal_var = True
            else:
                equal_var = False
        
            stat, pvalue = stats.ttest_ind(a=group_ttest1, b=group_ttest2, equal_var=equal_var)
            
            pvalues.append(pvalue)
            
        df_pvalues = pd.DataFrame({'pvalue': pvalues})
            
        return df_pvalues