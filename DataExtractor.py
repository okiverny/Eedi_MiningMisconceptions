import pandas as pd

def process_question(data: pd.DataFrame, question_id: int, is_labeled=True) -> dict:
    row = data.loc[[question_id]]

    # QuestionId
    QuestionId = question_id

    # General subject
    SubjectId = row['SubjectId'].values[0]
    SubjectName = row['SubjectName'].values[0]

    # Detailed topic
    ConstructId = row['ConstructId'].values[0]
    ConstructName = row['ConstructName'].values[0]

    # Question statement
    QuestionText = row['QuestionText'].values[0]

    # Correct answer
    CorrectAnswer = row['CorrectAnswer'].values[0]
    CorrectAnswerText = row[f'Answer{CorrectAnswer}Text'].values[0]
    
    # Incorrect answers
    IncorrectAnswers, IncorrectOptions = [], []
    MisconceptionIds = []
    for option in ['A', 'B', 'C', 'D']:
        if option == CorrectAnswer: continue

        # Incorrect answers
        IncorrectAnswerText = row[f'Answer{option}Text'].values[0]
        IncorrectAnswers.append(IncorrectAnswerText)
        IncorrectOptions.append(option)

        # If provided, appending misconseptions
        if is_labeled:
            try:
                MisconceptionId = row[f'Misconception{option}Id'].values[0]
                MisconceptionIds.append(MisconceptionId)
            except KeyError:
                print('No labels are present in the DataFrame. Please, check your data!')

    # Creating a dictionary for output
    output = {
        'QuestionId': QuestionId,
        'SubjectId': SubjectId,
        'SubjectName': SubjectName,
        'ConstructId': ConstructId,
        'ConstructName': ConstructName,
        'QuestionText': QuestionText,
        'CorrectAnswer': CorrectAnswer,
        'CorrectAnswerText': CorrectAnswerText,
        'IncorrectOptions': IncorrectOptions,
        'IncorrectAnswers': IncorrectAnswers,
        'MisconceptionIds': MisconceptionIds if is_labeled else None,
    }

    return output


def standardize_question_data(question_details: dict, data: dict) -> dict:
    # Standartize the data vs incorrect answers
    for i in range(3):
        for key in question_details:
            if key in ['IncorrectAnswers', 'MisconceptionIds', 'IncorrectOptions', 'QuestionId_Answer']: continue
            data[key].append(question_details[key])

        # Remaining data in lists
        data['IncorrectAnswer'].append(question_details['IncorrectAnswers'][i])
        data['IncorrectOption'].append(question_details['IncorrectOptions'][i])
        data['QuestionId_Answer'].append( f"{question_details['QuestionId']}_{question_details['IncorrectOptions'][i]}" )
        if question_details['MisconceptionIds']:
            data['MisconceptionId'].append(question_details['MisconceptionIds'][i])
        else:
            data['MisconceptionId'].append(0)

    return data

def process_data(input_df: pd.DataFrame, is_labeled: bool) -> pd.DataFrame:
    # Create new data collection
    data = {
        'QuestionId': [],
        'SubjectId': [],
        'SubjectName': [],
        'ConstructId': [],
        'ConstructName': [],
        'QuestionText': [],
        'CorrectAnswer': [],
        'CorrectAnswerText': [],
        'IncorrectAnswer': [],
        'IncorrectOption': [],
        'QuestionId_Answer': [],
        'MisconceptionId': [],
    }

    # Reading question block and standardize the input data
    for question_id in input_df.index.values:
        question_details = process_question(input_df, question_id, is_labeled=is_labeled)
        data = standardize_question_data(question_details, data)
        #print(question_details)
        #print(data)

    return pd.DataFrame(data=data)