import pickle
import numpy as np
import pandas as pd

import plotly
import plotly.graph_objs as go
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

with open("./state/Labels_4-ErrorResults-31x31-v2.pkl", "rb") as inpickle:
    results = pickle.load(inpickle)

#     print("\t   (nodes_number, epoch_number, batch_size_number, error_count, accuracy, train_delta_secs, test_delta_secs)")

data = []
for k,a in results.items():
    for v in a:
        data.append(v)

dt = np.dtype('int,int,int,int,float,float,float')

df = pd.DataFrame(data, 
                  columns=['nodes', 'epochs', 'batch_size', 'error_count', 'accuracy', 'train_delta_secs', 'test_delta_secs']).astype(dtype={'nodes':int, 'epochs':int, 'batch_size':int, 'error_count':int, 'accuracy':float, 'train_delta_secs':float, 'test_delta_secs':float})

print(f"{df.shape=}")
print(f"{df.info()=}")

df9000 = df[df['accuracy'] >= 0.9]
print(f"{df9000.shape=}")
df9900 = df[df['accuracy'] >= 0.99]
print(f"{df9900.shape=}")
df9990 = df[df['accuracy'] >= 0.999]
print(f"{df9990.shape=}")
dfmax = df[df.accuracy == df.accuracy.max()]
print(f"{dfmax.shape=}")

#Set marker properties
df9000['asserts'] = df9000['error_count'].max() - df9000['error_count'] + 1

'''
fig = go.Scatter3d(x=df9000['nodes'],
                   y=df9000['epochs'],
                   z=df9000['batch_size'],
                    marker=dict(color=markercolor,
                                opacity=1,
                                reversescale=True,
                                colorscale='Blues',
                                size=5),
                    line=dict (width=0.02),
                   mode='markers')
layout = go.Layout(scene=dict(xaxis=dict(title='nodes'), 
                              yaxis=dict(title='epochs'), 
                              zaxis=dict(title='batch_size')),)

plotly.offline.plot({"data":[fig], "layout": layout}, auto_open=True)
'''

fig2 =px.scatter_3d(df9000, 
                    x='nodes', 
                    y='epochs',
                    z='batch_size',
                    color='accuracy',
                    size='asserts'
                    )

# tight layout
fig2.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig2.show()