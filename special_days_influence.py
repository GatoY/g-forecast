from fbprophet import Prophet
from tqdm import tqdm
import pandas as pd
def fp_influence(
                store_df,
                specialEvents,
                analyze_column):
    '''
    Args:
        

    Output:
    '''


    ### prepare history data
    history = store_df[['dateTime',analyze_column]]
    history.rename(columns={"dateTime": "ds", analyze_column: "y"},inplace=True)

    ### prepare holiday data
    store_df_state = store_df['state'].unique()[0]
    specialEvents = specialEvents[specialEvents['state'] == store_df_state]

    namelist = list(specialEvents.name.unique()) 
    holiday_concat_list = []
    
    for name in tqdm(namelist):
        # store_factor1
        temp_store_df = specialEvents[specialEvents.name == name]
        ds = list(temp_store_df.date)
        temp_df = pd.DataFrame({'holiday':name,
                                'ds':ds,
                                'lower_window':0,
                                'upper_window':0,})
        holiday_concat_list.append(temp_df)
    
    holidays = pd.concat(tuple(holiday_concat_list))
    holidays.reset_index(drop=True,inplace=True)
    ### fit
    m = Prophet()
    m.fit(history)

    future = m.make_future_dataframe(periods=365)
    m = Prophet(holidays=holidays)
    forecast = m.fit(history).predict(future)

    ### extract influence
    influence_df = pd.DataFrame(columns=['name','influence'])
    for name in tqdm(namelist):
        influence = forecast[forecast[name].abs() > 0][name].mean()
        temp_df = pd.DataFrame({
            'name': [name],
            'influence': [influence],
        })

        influence_df = influence_df.append(temp_df)
    return influence_df