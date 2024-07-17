### Imports ###

from transformers import AutoTokenizer, AutoModelForCausalLM

### End of Imports ###


### Function Definition ###

# Given a valid path, will load a pretained model. Defaults to local search, otherwise tries HuggingFace, local, URL.
def loadPretrainedModel(configPathToModel: str, isLocalOnly=True) -> list:
    tokenizer = AutoTokenizer.from_pretrained(configPathToModel, local_files_only=isLocalOnly)
    model = AutoModelForCausalLM.from_pretrained(configPathToModel, local_files_only=isLocalOnly)

    return [tokenizer, model]


# Retrieves a tokenizer and a model, given a valid path to model directory, and responds to the LLM input string.
def runQuery(configPathToModel: str,
             inputString: str,
             isLocalOnly=True,
             max_length=512,
             repeat_penalty=1.2
             ) -> str:
    # Try to load a valid tokenizer and model, given a location string.
    try:
        tokenizerAndModel = loadPretrainedModel(configPathToModel, isLocalOnly)
        tokenizer = tokenizerAndModel[0]
        model = tokenizerAndModel[1]
        try:
            # Take our string, turn into series of tokens, and respond to it with the Large Language Model.
            input_ids = tokenizer(inputString, return_tensors='pt')
            outputs = model.generate(**input_ids,
                                     max_length=max_length,
                                     repetition_penalty=repeat_penalty
                                     )
            print(tokenizer.decode(outputs[0]))
        except Exception as e:
            print('Could not generate a response with the loaded model!')
            print(str(e))
    # Print a warning message associated with the error that occurred.
    except Exception as e:
        print('Could not load model, an exception occurred!')
        print(str(e))


### End of Function Definition ###

### Frozen Variables ###

pathToGemma = 'C:/Users/lukep/OneDrive/workandplay/Logic and Software/Computer Science/LLM/gemma/2b'

### End of Frozen Variables ###


### Dynamic Logic ###

test = runQuery(pathToGemma,
                """Please provide me a summary of the following text. 
                If possible, please try to follow the format of
                Name: Their Name
                Phone Number: Their Number
                Email: Their Email
                And some summary of their work, in complete sentences.
                Here is the text:

                Bird & Bird - Cat Hughes Success Stories People Capabilities Insights About Reach Careers Trending 
                Topics Our blogs TwoBirds TV News & Deals Events Alumni Contact Search Subscribe Twitter LinkedIn WeChat
                People Capabilities Insights About Reach Careers Clear ENG Dansk Deutsch English Español Français 
                Italiano Polski Suomi 한국어 中文 日本語 Cat Hughes Associate UK Email catherine.hughes@twobirds.com 
                Phone +44 (0)20 7415 6000 Vcard Home People Cat Hughes Share Twitter LinkedIn About Me 
                I am an associate in our Privacy and Data Protection Group, based in London. 
                I advise UK and international clients across a variety of sectors on a wide range of data protection 
                and e-privacy issues. My work covers all aspects of data protection law, including advising on data 
                protection compliance projects, data transfers, DPIAs, data security ...
                """
                )

### End of Dynamic Logic ###


























