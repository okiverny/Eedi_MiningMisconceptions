import pandas as pd

def process_question(data: pd.DataFrame, question_id: int) -> dict:
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
    IncorrectAnswers = []
    for option in ['A', 'B', 'C', 'D']:
        if option == CorrectAnswer: continue

        IncorrectAnswerText = row[f'Answer{option}Text'].values[0]
        IncorrectAnswers.append(IncorrectAnswerText)

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
        'IncorrectAnswers': IncorrectAnswers,
    }

    return output 