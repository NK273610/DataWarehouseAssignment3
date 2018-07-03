import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap




dataframe=pd.read_csv("Building_Permits.csv")

plt.hist(dataframe['PERMIT_TYPE'], normed=True, bins=30)
plt.ylabel('Count')
df=dataframe.groupby(['X','Y']).size().reset_index().rename(columns={0:'count'})
lngs = df['X'].astype('float')
lats = df['Y'].astype('float')
mags = df['count'].astype('float').apply(lambda x: 0.1 * x)
plt.figure(figsize=(8, 8))
earth = Basemap()
earth.drawcoastlines(color='0.50', linewidth=0.25)
earth.shadedrelief()
plt.scatter(lngs, lats, mags,
c='blue',alpha=0.5, zorder=10)
plt.xlabel("M4.5 earthquakes in the past 30 days from June 28, 2018 (USGS)")
# Workaround for blanc image saving
fig1 = plt.gcf()
fig1.savefig('4.5quakes.png', dpi=350)
plt.figure(figsize=(4, 4))
plt.rcParams.update({'font.size': 5})
plt.show()
plt.draw()

