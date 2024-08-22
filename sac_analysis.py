import pandas as pd
from io import StringIO
from pymoo.indicators.hv import HV
import numpy as np 

data0 = """
field,value,step
val_charts/episodic_return,18.299999237060547,0
val_charts/diversity,0.21166666666666673,0
val_charts/episodic_return,17.049999237060547,1000
val_charts/diversity,0.22749999999999998,1000
val_charts/episodic_return,19.200000762939453,2000
val_charts/diversity,0.2141666666666667,2000
val_charts/episodic_return,20.75,3000
val_charts/diversity,0.2241666666666667,3000
val_charts/episodic_return,19.700000762939453,4000
val_charts/diversity,0.22166666666666668,4000
val_charts/episodic_return,20.25,5000
val_charts/diversity,0.22166666666666668,5000
val_charts/episodic_return,20.850000381469727,6000
val_charts/diversity,0.22333333333333333,6000
val_charts/episodic_return,21.149999618530273,7000
val_charts/diversity,0.22166666666666668,7000
val_charts/episodic_return,21.0,8000
val_charts/diversity,0.21166666666666667,8000
val_charts/episodic_return,20.649999618530273,9000
val_charts/diversity,0.21166666666666667,9000
val_charts/episodic_return,19.399999618530273,10000
val_charts/diversity,0.22083333333333335,10000
"""
# Provided data
data1 = """
field,value,step
val_charts/episodic_return,40.45000076293945,0
val_charts/diversity,0.2165,0
val_charts/episodic_return,43.0,1000
val_charts/diversity,0.21900000000000003,1000
val_charts/episodic_return,47.45000076293945,2000
val_charts/diversity,0.21788888888888885,2000
val_charts/episodic_return,45.79999923706055,3000
val_charts/diversity,0.2123888888888889,3000
val_charts/episodic_return,36.20000076293945,4000
val_charts/diversity,0.20961111111111114,4000
val_charts/episodic_return,46.599998474121094,5000
val_charts/diversity,0.21855555555555556,5000
val_charts/episodic_return,48.599998474121094,6000
val_charts/diversity,0.2212222222222222,6000
val_charts/episodic_return,48.349998474121094,7000
val_charts/diversity,0.21683333333333335,7000
val_charts/episodic_return,47.150001525878906,8000
val_charts/diversity,0.21916666666666668,8000
val_charts/episodic_return,47.599998474121094,9000
val_charts/diversity,0.2172222222222222,9000
val_charts/episodic_return,45.79999923706055,10000
val_charts/diversity,0.22027777777777774,10000
"""

data2 = """
field,value,step
val_charts/episodic_return,41.650001525878906,0
val_charts/diversity,0.20994444444444443,0
val_charts/episodic_return,42.400001525878906,1000
val_charts/diversity,0.21700000000000003,1000
val_charts/episodic_return,41.70000076293945,2000
val_charts/diversity,0.22016666666666662,2000
val_charts/episodic_return,43.400001525878906,3000
val_charts/diversity,0.2206666666666667,3000
val_charts/episodic_return,42.70000076293945,4000
val_charts/diversity,0.21494444444444447,4000
val_charts/episodic_return,43.900001525878906,5000
val_charts/diversity,0.22161111111111115,5000
val_charts/episodic_return,43.5,6000
val_charts/diversity,0.20894444444444446,6000
val_charts/episodic_return,42.0,7000
val_charts/diversity,0.20566666666666672,7000
val_charts/episodic_return,35.5,8000
val_charts/diversity,0.21916666666666668,8000
val_charts/episodic_return,44.5,9000
val_charts/diversity,0.22349999999999998,9000
val_charts/episodic_return,43.0,10000
val_charts/diversity,0.21405555555555558,10000
"""

data3 = """
field,value,step
val_charts/diversity,0.20816666666666667,0
val_charts/episodic_return,40.099998474121094,1000
val_charts/diversity,0.20672222222222225,1000
val_charts/episodic_return,41.95000076293945,2000
val_charts/diversity,0.21511111111111111,2000
val_charts/episodic_return,42.150001525878906,3000
val_charts/diversity,0.21877777777777782,3000
val_charts/episodic_return,41.0,4000
val_charts/diversity,0.21105555555555555,4000
val_charts/episodic_return,41.349998474121094,5000
val_charts/diversity,0.21550000000000002,5000
val_charts/episodic_return,40.75,6000
val_charts/diversity,0.2131111111111111,6000
val_charts/episodic_return,40.849998474121094,7000
val_charts/diversity,0.21283333333333335,7000
val_charts/episodic_return,39.54999923706055,8000
val_charts/diversity,0.21455555555555555,8000
val_charts/episodic_return,40.150001525878906,9000
val_charts/diversity,0.21961111111111112,9000
val_charts/episodic_return,42.04999923706055,10000
val_charts/diversity,0.20777777777777778,10000
"""

data4 = """
field,value,step
val_charts/episodic_return,36.75,0
val_charts/diversity,0.21577777777777776,0
val_charts/episodic_return,37.599998474121094,1000
val_charts/diversity,0.21772222222222223,1000
val_charts/episodic_return,35.79999923706055,2000
val_charts/diversity,0.21211111111111114,2000
val_charts/episodic_return,40.150001525878906,3000
val_charts/diversity,0.21533333333333332,3000
val_charts/episodic_return,38.79999923706055,4000
val_charts/diversity,0.21772222222222223,4000
val_charts/episodic_return,27.75,5000
val_charts/diversity,0.21250000000000005,5000
val_charts/episodic_return,36.599998474121094,6000
val_charts/diversity,0.21122222222222226,6000
val_charts/episodic_return,37.20000076293945,7000
val_charts/diversity,0.20427777777777778,7000
val_charts/episodic_return,36.79999923706055,8000
val_charts/diversity,0.21161111111111114,8000
val_charts/episodic_return,39.099998474121094,9000
val_charts/diversity,0.2147222222222222,9000
val_charts/episodic_return,35.75,10000
val_charts/diversity,0.21583333333333338,10000
"""

data5 = """
field,value,step
val_charts/episodic_return,42.400001525878906,0
val_charts/diversity,0.2112222222222222,0
val_charts/episodic_return,44.0,1000
val_charts/diversity,0.2106666666666667,1000
val_charts/episodic_return,44.150001525878906,2000
val_charts/diversity,0.20933333333333337,2000
val_charts/episodic_return,44.54999923706055,3000
val_charts/diversity,0.21394444444444444,3000
val_charts/episodic_return,43.20000076293945,4000
val_charts/diversity,0.2070555555555556,4000
val_charts/episodic_return,43.45000076293945,5000
val_charts/diversity,0.2066666666666667,5000
val_charts/episodic_return,41.150001525878906,6000
val_charts/diversity,0.20550000000000002,6000
val_charts/episodic_return,42.099998474121094,7000
val_charts/diversity,0.20294444444444446,7000
val_charts/episodic_return,49.0,8000
val_charts/diversity,0.20761111111111114,8000
val_charts/episodic_return,44.5,9000
val_charts/diversity,0.21188888888888888,9000
val_charts/episodic_return,43.45000076293945,10000
val_charts/diversity,0.2083888888888889,10000
"""

return_list = []
diversity_list = []
hv_list = []
for i, data in enumerate([data0]):
    print(f"Data {i}")
    # Load the data into a DataFrame
    df = pd.read_csv(StringIO(data))

    # Calculate the average episodic return and diversity

    avg_episodic_return = df[df['field'] == 'val_charts/episodic_return']['value'].mean()
    avg_diversity = df[df['field'] == 'val_charts/diversity']['value'].mean()

    # concatenate column return and diversity
    df = df.pivot(index='step', columns='field', values='value').reset_index()


    # calculate hypervolume
    ref_point = np.array([0.0, 0.0])
    points = df[['val_charts/episodic_return', 'val_charts/diversity']].values

    hv = HV(ref_point=ref_point * -1)(np.array(points) * -1)

    return_list.append(avg_episodic_return)
    diversity_list.append(avg_diversity)
    hv_list.append(hv)
    print(avg_episodic_return, avg_diversity, hv)

print("Average return", np.mean(return_list), "+- ", np.std(return_list))
print("Average diversity", np.mean(diversity_list), "+- ", np.std(diversity_list))
print("Average HV", np.mean(hv_list), "+- ", np.std(hv_list))