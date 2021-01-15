import numpy as np
import pandas as pd
import Settings.GeneralSettings as generalSettings
from Settings.SettingsPimaIndiansDiabetes import Settings
from Class.FuzzifierProgressive import FuzzifierProgressive
from Class.Fuzzifier import Fuzzifier
from Class.CSVExtractor import CSVExtractor
from Scripts.Fuzzify import Fuzzify
from Scripts.ValueTest import ValueTest
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.simplefilter("ignore")


def main():
    best_params = pd.DataFrame(
        [['Gaussian Progressive', 7, 'Mean'], ['Gaussian Progressive', 7, 'Center'], ['Gaussian Equal', 9, 'Center'],
         ['Gaussian Progressive', 11, 'Center']],
        columns=['Fuzzifier Type', 'Gausses', 'Adjustment'])

    results = pd.DataFrame({
        'Fuzzifier Type': pd.Series([], dtype='str'),
        'Gausses': pd.Series([], dtype='int'),
        'Adjustment': pd.Series([], dtype='str'),
        'Sigma Offset': pd.Series([], dtype='float'),
        'Center Offset': pd.Series([], dtype='float'),
        'Accuracy': pd.Series([], dtype='float'),
        'F1 Score': pd.Series([], dtype='float')
    })

    i = 0
    for _, data in best_params.iterrows():
        gauss = data['Gausses']
        style = data['Fuzzifier Type']
        adjustment = data['Adjustment']
        if adjustment == "Mean":
            adjustment = -1
        else:
            adjustment = -2

        for sigma_offset in np.arange(0.00, 0.032, 0.005):
            for center_offset in np.arange(0.00, 0.22, 0.02):

                resultsKFold = pd.DataFrame({
                    'Accuracy': pd.Series([], dtype='float'),
                    'F1 Score': pd.Series([], dtype='float')
                })

                print("Gausses: " + str(gauss) + " Adjustment: " + str(adjustment) + "\n" + style)
                generalSettings.gausses = gauss
                generalSettings.style = style
                generalSettings.adjustment_value = adjustment
                settings = Settings(generalSettings)

                d_results = [settings.class_2, settings.class_1]
                if style == 'Gaussian Progressive':
                    fuzzifier = FuzzifierProgressive(settings, d_results)
                else:
                    fuzzifier = Fuzzifier(settings, d_results)

                csvExtractor = CSVExtractor(settings)
                features_df = csvExtractor.worker(fuzzifier)
                train_features_df, test_features_df = train_test_split(features_df, test_size=0.3,
                                                                       stratify=features_df.Decision, random_state=23)

                skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=23)
                X = train_features_df.drop('Decision', axis=1)
                y = train_features_df.Decision

                for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
                    train_data = train_features_df.iloc[train_index]
                    train_data_for_worker = train_data.copy()
                    test_data = train_features_df.iloc[test_index]
                    fuzzify = Fuzzify()
                    changed_decisions, features_number_after_reduct, implicants_number, features, decision_table_with_reduct, reductor = fuzzify.workerKFold(
                        settings, train_data_for_worker, settings.adjustment_value, sigma_offset, fuzzifier)
                    valueTest = ValueTest(settings, settings.s_function_width, settings.is_training, load_data=False)
                    valueTest.createAntecedents(reductor, features, decision_table_with_reduct, train_data, "Train")
                    valueTest.sOptymalizationWorker(settings, settings.default_s_value, settings.show_results,
                                                    center_offset)
                    valueTest = ValueTest(settings, settings.s_function_width, not settings.is_training,
                                          load_data=False)
                    valueTest.createAntecedents(reductor, features, decision_table_with_reduct, test_data, "Test")
                    valueTest.sOptymalizationWorker(settings, settings.default_s_value, settings.show_results,
                                                    center_offset)
                    resultsKFold = resultsKFold.append({
                        'Accuracy': valueTest.results.iloc[0]['Accuracy'],
                        'F1 Score': valueTest.results.iloc[0]['F1 Score']
                    }, ignore_index=True)

                results = results.append({
                    'Fuzzifier Type': style,
                    'Gausses': gauss,
                    'Adjustment': adjustment,
                    'Sigma Offset': sigma_offset,
                    'Center Offset': center_offset,
                    'Accuracy': resultsKFold['Accuracy'].mean(),
                    'F1 Score': resultsKFold['F1 Score'].mean()
                }, ignore_index=True)

        results.to_csv(r'Test - Datasets\ResultsIT2\PimaIndiansDiabetesSigmaCenterKFoldGrid' + str(i) + '.csv', index=None)
        i += 1

    results.to_csv(r'Test - Datasets\ResultsIT2\PimaIndiansDiabetesSigmaCenterKFoldGrid.csv', index=None)


if __name__ == '__main__':
    main()
