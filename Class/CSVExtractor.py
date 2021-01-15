import pandas as pd

class CSVExtractor(object):
    
    def __init__(self, settings):
        self.path = settings.data_folder
        self.settings = settings

    def extractFeatures(self):
        features_df = pd.read_csv(self.path, sep=";")
        features_df["Image"] = ""

        return features_df

    def normalizeFeatures(self, features_df, fuzzifier):
        for x in fuzzifier.features:
            features_df[x.label] = (
                features_df[x.label] - features_df[x.label].min()) / (
                    features_df[x.label].max() - features_df[x.label].min())

        return features_df

    def worker(self, fuzzifier):
        features_df = self.extractFeatures()
        normalized_features_df = self.normalizeFeatures(features_df, fuzzifier)
        return normalized_features_df
