import pandas as pd
from io import StringIO
from pymoo.indicators.hv import HV
import numpy as np 
# Provided data
data1 = """
field,value,step
val_charts/episodic_return,20.75,0
val_charts/diversity,0.2325,0
val_charts/episodic_return,19.799999237060547,1000
val_charts/diversity,0.2333333333333333,1000
val_charts/episodic_return,23.149999618530273,2000
val_charts/diversity,0.21750000000000003,2000
val_charts/episodic_return,22.5,3000
val_charts/diversity,0.21250000000000005,3000
val_charts/episodic_return,23.549999237060547,4000
val_charts/diversity,0.22916666666666669,4000
val_charts/episodic_return,23.299999237060547,5000
val_charts/diversity,0.22583333333333333,5000
val_charts/episodic_return,24.25,6000
val_charts/diversity,0.21750000000000003,6000
val_charts/episodic_return,23.75,7000
val_charts/diversity,0.21333333333333337,7000
val_charts/episodic_return,22.25,8000
val_charts/diversity,0.2141666666666667,8000
val_charts/episodic_return,21.5,9000
val_charts/diversity,0.21916666666666668,9000
val_charts/episodic_return,22.899999618530273,10000
val_charts/diversity,0.22916666666666666,10000
"""

data2 = """
field,value,step
val_charts/episodic_return,21.200000762939453,0
val_charts/diversity,0.2166666666666667,0
val_charts/episodic_return,21.700000762939453,1000
val_charts/diversity,0.20666666666666664,1000
val_charts/episodic_return,20.649999618530273,2000
val_charts/diversity,0.22750000000000004,2000
val_charts/episodic_return,20.75,3000
val_charts/diversity,0.2225,3000
val_charts/episodic_return,21.350000381469727,4000
val_charts/diversity,0.2166666666666667,4000
val_charts/episodic_return,20.799999237060547,5000
val_charts/diversity,0.23166666666666663,5000
val_charts/episodic_return,17.600000381469727,6000
val_charts/diversity,0.21166666666666667,6000
val_charts/episodic_return,20.299999237060547,7000
val_charts/diversity,0.21750000000000003,7000
val_charts/episodic_return,20.100000381469727,8000
val_charts/diversity,0.21000000000000002,8000
val_charts/episodic_return,20.100000381469727,9000
val_charts/diversity,0.22999999999999998,9000
val_charts/episodic_return,23.299999237060547,10000
val_charts/diversity,0.215,10000
"""

data3 = """
field,value,step
val_charts/episodic_return,23.200000762939453,0
val_charts/diversity,0.22333333333333333,0
val_charts/episodic_return,24.649999618530273,1000
val_charts/diversity,0.23000000000000004,1000
val_charts/episodic_return,23.450000762939453,2000
val_charts/diversity,0.2225,2000
val_charts/episodic_return,20.950000762939453,3000
val_charts/diversity,0.22833333333333336,3000
val_charts/episodic_return,22.399999618530273,4000
val_charts/diversity,0.23000000000000004,4000
val_charts/episodic_return,16.700000762939453,5000
val_charts/diversity,0.24000000000000005,5000
val_charts/episodic_return,22.200000762939453,6000
val_charts/diversity,0.21916666666666665,6000
val_charts/episodic_return,22.549999237060547,7000
val_charts/diversity,0.22166666666666668,7000
val_charts/episodic_return,15.0,8000
val_charts/diversity,0.22583333333333333,8000
val_charts/episodic_return,18.600000381469727,9000
val_charts/diversity,0.22083333333333335,9000
val_charts/episodic_return,23.0,10000
val_charts/diversity,0.22333333333333333,10000
"""

# Load the data into a DataFrame
df = pd.read_csv(StringIO(data3))

# Calculate the average episodic return and diversity

avg_episodic_return = df[df['field'] == 'val_charts/episodic_return']['value'].mean()
avg_diversity = df[df['field'] == 'val_charts/diversity']['value'].mean()

# concatenate column return and diversity
df = df.pivot(index='step', columns='field', values='value').reset_index()


# calculate hypervolume
ref_point = np.array([0.0, 0.0])
points = df[['val_charts/episodic_return', 'val_charts/diversity']].values

hv = HV(ref_point=ref_point * -1)(np.array(points) * -1)

print(avg_episodic_return, avg_diversity, hv)
