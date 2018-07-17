## Use this for Azure AD authentication
from msrestazure.azure_active_directory import AADTokenCredentials
import pandas as pd
## Required for Azure Data Lake Store account management
from azure.mgmt.datalake.store import DataLakeStoreAccountManagementClient
from azure.mgmt.datalake.store.models import DataLakeStoreAccount

import plotly
import plotly.graph_objs
## Required for Azure Data Lake Store filesystem management
from azure.datalake.store import core, lib, multithread
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot,figure
# Common Azure imports
import adal
from azure.mgmt.resource.resources import ResourceManagementClient
from azure.mgmt.resource.resources.models import ResourceGroup

## Use these as needed for your application
import logging, getpass, pprint, uuid, time

tenant = '60b81999-0b7f-412d-92a3-e17d8ae9e3e0'
RESOURCE = 'https://datalake.azure.net/'
client_id = 'e4b4dbc2-41d4-4c4f-b6cc-94082b1801fd'
client_secret = 'jX44VjXPf2hdSVxrTS/Bdu2K6b05F61CXRaJrk9dIxE='

adlCreds = lib.auth(tenant_id = tenant,
                client_secret = client_secret,
                client_id = client_id,
                resource = RESOURCE)

adlsFileSystemClient = core.AzureDLFileSystem(adlCreds, store_name='datawarehouse')


# x='/output/dataquery1.csv'
#
# with adlsFileSystemClient.open(x, 'rb') as f:
#     df = pd.read_csv(f)

# plotly.offline.plot({
# "data": [
#     plotly.graph_objs.Bar(x=df['Main'],y=df['Average_Vehicles'])
# ]
# })

# x='/output/dataquery2.csv'
#
# with adlsFileSystemClient.open(x, 'rb') as f:
#     df = pd.read_csv(f)
#
#     print df
#
# plotly.offline.plot({
# "data": [
#     plotly.graph_objs.Pie(labels=df['Main'],values=df['Total_Traffic'])
# ]
# })

# x='/output/dataquery3.csv'
#
# with adlsFileSystemClient.open(x, 'rb') as f:
#     df = pd.read_csv(f)
#
#     print df
#
# plotly.offline.plot({
# "data": [
#     plotly.graph_objs.Scatter(x=df['Year'],y=df['Total_Traffic'])
# ]
# })

# x='/output/dataquery4.csv'
#
# with adlsFileSystemClient.open(x, 'rb') as f:
#     df = pd.read_csv(f)
#
#     print df
#
# plotly.offline.plot({
# "data": [
#     plotly.graph_objs.Scatter(x=df['Day'],y=df['Total'],
#     mode = 'markers',
#     name = 'markers')
# ]
# })

x='/output/dataquery5.csv'

with adlsFileSystemClient.open(x, 'rb') as f:
    df = pd.read_csv(f)

    print df



plotly.offline.plot({
"data": [
    plotly.graph_objs.Scatter3d(x=df['Day'],y=df['Total'],z=df['Main'],mode='markers')
]
})






