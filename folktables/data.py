import os
import numpy as np
from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSPublicCoverage, ACSMobility, ACSTravelTime

states_list = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]
Dataset = {"ACSIncome":ACSIncome, "ACSEmployment":ACSEmployment,"ACSPublicCoverage":ACSPublicCoverage,"ACSMobility":ACSMobility,"ACSTravelTime":ACSTravelTime}
def get_data():
    for task in Dataset.keys():
        data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
        if not os.path.exists(f"data/{task}"):
            os.makedirs(f"data/{task}")
        for state in states_list:
            data = data_source.get_data(states=[state], download=False)
            features, labels, groups = Dataset[task].df_to_numpy(data)
            np.save(f"data/{task}/{state}_X", features)
            np.save(f"data/{task}/{state}_y", labels)
            np.save(f"data/{task}/{state}_g", groups)
        print(f"{task} finished!")

get_data()