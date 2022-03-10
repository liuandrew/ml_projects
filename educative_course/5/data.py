import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_dataframes():
    train_df = pd.read_csv('weekly_sales.csv')
    features_df = pd.read_csv('features.csv')
    stores_df = pd.read_csv('stores.csv')
    return train_df, features_df, stores_df

def get_traindata():
    train_df, features_df, stores_df = read_dataframes()
    merged_features = features_df.merge(stores_df, on='Store')

    na_values = pd.isna(merged_features)
    na_features = na_values.any()
    # print(na_features)

    # col_names = merged_features.columns
    # for i in range(len(col_names)):
    #     feature = col_names[i]
    #     if na_features[i]:
    #         print('{} column is missing {} values'.format(feature, sum(na_values[feature])))
            
    markdowns = [
        'MarkDown1',
        'MarkDown2',
        'MarkDown3',
        'MarkDown4',
        'MarkDown5',
    ]
    merged_features = merged_features.drop(columns=markdowns)
    # print(merged_features.columns.tolist())

    # na_cpi_int = na_values['CPI'].astype(int)
    # na_indexes_cpi = na_cpi_int.to_numpy().nonzero()[0]
    na_indexes_cpi = np.where(na_values['CPI'])[0]
    na_indexes_une = np.where(na_values['Unemployment'])[0]
    na_rows = merged_features.iloc[na_indexes_cpi]
    # print(na_rows['Date'].unique())
    # print(na_rows.groupby('Store').count()['Date'].unique())

    # Clean up missing values
    for idx in na_indexes_cpi:
        merged_features.at[idx, 'CPI'] = merged_features.at[idx - 1, 'CPI']
    for idx in na_indexes_une:
        merged_features.at[idx, 'Unemployment'] = merged_features.at[idx - 1, 'Unemployment']
        
    features = ['Store', 'Date', 'IsHoliday']    
    final_dataset = train_df.merge(merged_features, on=features)
    final_dataset = final_dataset.drop(columns=['Date'])

    final_dataset['IsHoliday'] = final_dataset['IsHoliday'].astype(np.int64)

    return final_dataset

if __name__ == '__main__':
    final_dataset = get_traindata()
    
    plot_df = final_dataset[['Weekly_Sales', 'Temperature']]
    rounded_temp = plot_df['Temperature'].round(0)
    
    plot_df = plot_df.groupby(rounded_temp).mean()
    plt.scatter(plot_df['Temperature'], plot_df['Weekly_Sales'])
    plt.show()
    
    plot_df = final_dataset[['Wee']]