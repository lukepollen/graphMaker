### Imports ###

from metabeaver.GoogleCloudPlatform.BigQuery.getTableData import getData
import ollama

### End of Imports ###

### Function Definition ###

def generateTriples(text, prompt='', model='sciphi/triplex'):

    if prompt == '':

        prompt = "Please consider the following text: "
        prompt = prompt + text
        prompt = prompt + '\n'
        prompt = prompt + "You are a Natural Language Processing robot."
        prompt = prompt + "You must return a tuple in the form of (ENTITYONE, RELATIONSHIP, ENTITYTWO)."
        prompt = prompt + "You must ONLY respond with rows of discovered (ENTITYONE, RELATIONSHIP, ENTITYTWO)."

    response = ollama.chat(model=model, messages=[
      {
        'role': 'user',
        'content': prompt,
      },
    ])
    print('Got a response like: ')
    print(response['message']['content'])
    return response

# Get the transformer vectors we created from analysing textual features of the language, used for document comparisons.
netcallData = getData(float('inf'), # Get all rows from GCP BigQuery
                     table_name='netcall_com' # Use config-yaml project and dataset, but this table
                     )
netcallPages = netcallData['Page']
print('Got all pages!')

allResponses = []
for eachPage in netcallPages:
    generateTriples(eachPage, prompt='', model='llama3')