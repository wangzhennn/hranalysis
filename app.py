import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import altair as alt
import umap.umap_ as umap
import seaborn as sns
sns.set()

hru=pd.read_csv("hru.csv")

print(hru.head(5))
