import pandas as pd
import numpy as np
from outlier_method_analytic import OutlierMethod, Analytics

class OutlierDetection(OutlierMethod, Analytics):
    def __init__(self, data):
        super().__init__()
        self.data = data
        
    def processData(self): 
        df_list = list()
        for driver in self.params: #['ptv', 'kratio', 'tico', 'seg', 'date']
            app1=self.data[[driver,'seg','date']]

            # Check the outlier method
            params_copy = self.params.copy()
            if self.n_variable == 'multivariate':
                params_copy.extend(['seg', 'date'])
                app1 = self.data[params_copy]

            for j in range(len(app1.seg.unique())): #['abc', 'abd', 'cde']
                if self.all_segments:
                    # Set variable choose to True if all segments is True
                    # This will ensure that 'df' below uses only 'date' column for selection
                    choose = True
                else:
                    # Set variable choose to 'app1.seg == app1.seg.unique()[j]' if all segments is False
                    # This will ensure that 'df' below uses both 'date' column and 'seg' column for selection
                    choose = app1.seg == app1.seg.unique()[j]

                for i in range(len(app1.date.unique())): #['5/1/2001', '6/1/2001', '7/1/2001']
                    df = app1[(app1.date == app1.date.unique()[i]) & (choose)]

                    df.set_index(['date','seg'], inplace=True)
                    
                    df_list.append(df)
                    
                if self.all_segments:
                    break
                    
            if self.n_variable == 'multivariate':
                break
                
        return df_list
        
    def get_outliers(self, n_variable=None, params=None, all_segments=True, method=None):
        self.variables_map = {'univariate': ['iqr', 'zscore'], 'multivariate': ['lof']}
        self.valid_methods = [method for variable in self.variables_map for method in self.variables_map[variable]]
        
        self.params,  self.all_segments = params, all_segments
        self.method, self.all_segments = method, all_segments
        self.n_variable = n_variable
        
        outliers, inliers, outliers_and_inliers = [], [], []
        
        if self.n_variable not in self.variables_map:
            print('n_variable parameter can only be "univariate" or "multivariate"')
            return False
        
        if not self.params or type(self.params) != list:
            print('Enter an array of param(s)')
            return False
        
        if self.method not in self.variables_map[self.n_variable]:
            print('n_variable and method missmatch')
            return
        
        if self.method not in self.valid_methods:
            print('Enter valid method. Valid methods are "iqr", "lof", "iForest"')
            return False
        else:
            for df in self.processData():
                if self.method == 'iqr':
                    outlier, inlier, outlier_and_inlier = self.iqr_method(df)
                    
                if self.method == 'zscore':
                    outlier, inlier, outlier_and_inlier = self.zscore_method(df)

                if self.method == 'lof':
                    outlier, inlier, outlier_and_inlier = self.lof_method(df)
                    
                    
                outliers.append(outlier)
                inliers.append(inlier)
                outliers_and_inliers.append(outlier_and_inlier)
                
            return (outliers, inliers, outliers_and_inliers)
    
    def combined_analytics(self, n_variable, params=None, method=None):
        self.params, self.method, self.n_variable = params, method, n_variable
        
        dataframe_all_segment = self.get_outliers(self.n_variable, self.params, True, self.method)
        dataframe_segmented = self.get_outliers(self.n_variable, self.params, False, self.method)
        
        if not dataframe_all_segment:
            print('Cannot perform operation')
            return False
        
        outliers_alone_all_segment, inliers_all_segment, outliers_and_inliers_all_segment = dataframe_all_segment
        outliers_alone_segmented, inliers_segmented, outliers_and_inliers_segmented = dataframe_segmented
        
        
        df_outrate1 = self.create_outlier_rate_table(False, outliers_alone_segmented)
        if not isinstance(df_outrate1, pd.core.frame.DataFrame):
            return (False, False)
        df_outrate2 = self.create_outlier_rate_table(True, outliers_alone_all_segment)
        
        df_pvalue1 = self.create_pvalue_table(False, inliers_segmented, outliers_and_inliers_segmented)
        df_pvalue2 = self.create_pvalue_table(True, inliers_all_segment, outliers_and_inliers_all_segment)
     
        outlier_rate = pd.concat([df_outrate1, df_outrate2]).reset_index(drop=True)
        pvalue = pd.concat([df_pvalue1, df_pvalue2]).reset_index(drop=True)
        
        return (outlier_rate, pvalue)
    
    def create_outlier_rate_table(self, all_segments, outliers_alone):
        outlier_rate = self.outlier_rate(outliers_alone)
        
        dates, segs, params = [], [], []
        
        for outlier_df in outliers_alone:
            idx = list(outlier_df.index.unique())
            
            dates.append(idx[0][0])
            segs.append(idx[0][1])
            params.append(list(outlier_df.columns)[0])
           
        if all_segments:
            segment = 'all segments'
        else:
            segment = segs
            
        if self.n_variable == 'multivariate':
            param = 'all params'
        else:
            param = params
            
        
        df_outlier_rates = pd.DataFrame({'date': dates, 'seg': segment, 'param': param})
        df_outlier_rates['outlier %'] = outlier_rate.iloc[:, -1]
        
        return df_outlier_rates
    
    def create_pvalue_table(self, all_segments, inliers, outliers_and_inliers):
        pvalue = self.get_pvalue(inliers, outliers_and_inliers)
        
        dates, segs, params = [], [], []
        
        for idx, _ in enumerate(outliers_and_inliers):
            idn = list(outliers_and_inliers[idx].index.unique())
            
            dates.append(idn[0][0])
            segs.append(idn[0][1])
            params.append(list(outliers_and_inliers[idx].columns)[0])
            
        if all_segments:
            segment = 'all segments'
        else:
            segment = segs
            
        if self.n_variable == 'multivariate':
            param = 'all params'
        else:
            param = params
            
        df_pvalues = pd.DataFrame({'date': dates, 'seg': segment, 'param': param})
        df_pvalues['pvalue'] = pvalue.iloc[:, -1]
        
        return df_pvalues
    
    
    
        
        
     
        
    
    
