import pandas as pd


class DataLoader:

    @staticmethod
    def load_validation_data():
        # load validation data
        val_data = pd.read_csv("../data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv")
        print("Labels: ", val_data.columns.tolist())
        val_right_ending_nr = val_data[['AnswerRightEnding']].values
        val_context_sentences = val_data.iloc[:, 1:5].values
        val_ending_sentence1 = val_data[['RandomFifthSentenceQuiz1']].values
        val_ending_sentence2 = val_data[['RandomFifthSentenceQuiz2']].values
        return val_right_ending_nr, val_context_sentences, val_ending_sentence1, val_ending_sentence2

    @staticmethod
    def load_training_data():
        # load training data
        train_data = pd.read_csv("../data/train_stories.csv")
        print("Labels: ", train_data.columns.tolist())
        train_context_sentences = train_data.iloc[:, 2:6].values
        train_ending_sentence = train_data[['sentence5']].values
        train_story_title = train_data[['storytitle']].values
        return train_context_sentences, train_ending_sentence, train_story_title


