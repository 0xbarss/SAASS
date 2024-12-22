import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


def load_bert():
    return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')

def load_data(filename):
    data = pd.read_excel(filename)
    return data

def export_data(data, path):
    data.to_excel(path, index=False)

def parse_answer_sheet(answerSheet):
    answerSheetDict = {}
    for _, row in answerSheet.iterrows():
        questionID = row["Question- ID"]
        answerSheetDict[questionID] = {
            "score_1": row["SCORE-1"].split(", "),
            "score_2": row["SCORE-2"].split(", "),
            "score_3": row["SCORE-3"].split(", ")
        }
    return dict(sorted(answerSheetDict.items(), key=lambda x: x[0]))

def compute_cosine_similarity(answers, parsedAnswerSheet, model):
    results = []
    for _, studentAnswers in answers.iterrows():
        studentID = studentAnswers["Unnamed: 0"]
        result = {"Student ID": studentID, "Student Total Score": 0}
        
        for questionID in parsedAnswerSheet.keys():
            studentAnswer = studentAnswers[f"{questionID}-answer"]
            if pd.isna(studentAnswer):
                continue
            questionData = parsedAnswerSheet[questionID]

            bestScore = 0
            bestMatch = None
            bestSimilarity = 0

            studentEmbedding = model.encode([studentAnswer], convert_to_tensor=True)

            for score, expectedAnswers in questionData.items():
                expectedEmbedding = model.encode(expectedAnswers, convert_to_tensor=True)
                similarities = cos_sim(studentEmbedding, expectedEmbedding)
                maxSimilarity = similarities.max().item()

                if maxSimilarity > bestSimilarity:
                    bestSimilarity = maxSimilarity
                    bestMatch = expectedAnswers[similarities.argmax()]
                    bestScore = int(score.split("_")[1])
            
            result[f"{questionID}-Predicted Score"] = bestScore
            result[f"{questionID}-Matched Answer"] = bestMatch
            result[f"{questionID}-Cosine Similarity"] = bestSimilarity
            result["Student Total Score"] += bestScore
        
        results.append(result)

    return pd.DataFrame(results)

def main():
    ANSWERS_PATH = "answers.xlsx"
    ANSWER_SHEET_PATH = "answer_sheet.xlsx"
    OUTPUT_PATH = "output.xlsx"

    answers = load_data(ANSWERS_PATH)
    parsedAnswerSheet = parse_answer_sheet(load_data(ANSWER_SHEET_PATH))

    model = load_bert()

    resultsDF = compute_cosine_similarity(answers, parsedAnswerSheet, model)

    export_data(resultsDF, OUTPUT_PATH)
    print("SAASS completed.")
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()