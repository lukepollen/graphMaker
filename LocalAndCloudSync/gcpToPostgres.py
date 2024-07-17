### Imports ###

# Python standard library imports


# Third party library imports
import pandas as pd
import yaml

from metabeaver.GoogleCloudPlatform.BigQuery.getTableData import getData

### End of Function Definition ###


### Frozen Variables ###

rowAmount = float('inf')

### End of Frozen Variables ###


### Dynamic Logic ###

# Access cloud and project settings defined in the yaml file.
with open('config-yaml.yaml', 'r') as file:
    configuration = yaml.safe_load(file)

tableList = ['wandworth_council',
             'lancashire_gov',
             'ukpowernetworks',
             'nationalrail']
metaTable = []
for table in tableList:
    configuration['GCP']['table_name'] = table
    currentTable = getData(rowAmount, configuration=configuration)
    metaTable.append(currentTable)
metaTable = pd.concat(metaTable)
pdfFrame = metaTable[metaTable['Page'].str.endswith('.pdf', na=False)]

### End of Imports ###



























