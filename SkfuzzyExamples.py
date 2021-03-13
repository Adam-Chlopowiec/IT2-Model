from scikitfuzzy.skfuzzy.control.antecedent_consequent import Antecedent, Consequent
from scikitfuzzy.skfuzzy.membership import sigmf
from scikitfuzzy.skfuzzy.control.term import TermAggregate
from scikitfuzzy.skfuzzy.control.rule import Rule
from scikitfuzzy.skfuzzy.control.controlsystem import ControlSystem, ControlSystemSimulation

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


def create_antecedents(universe, labels):
    antecedents = []
    for label in labels:
        antecedents.append(Antecedent(universe, label))
    return antecedents[0] if len(antecedents) == 1 else antecedents


def create_consequents(universe, labels):
    consequents = []
    for label in labels:
        consequents.append(Consequent(universe, label))
    return consequents[0] if len(consequents) == 1 else consequents


def create_terms_for_fuzzy_variable(fuzzy_variable, labels, mfs):
    for label, mf in zip(labels, mfs):
        fuzzy_variable[label] = mf


def generate_sigmf(universe, center, width):
    return sigmf(universe, center, width)


def makePrediction(row_from_df, control_system_sim, features_names):
    system_input = {}
    for name in features_names:
        system_input[name] = row_from_df[name]

    control_system_sim.inputs(system_input)

    try:
        control_system_sim.compute()

    except:
        row_from_df['Predicted Value'] = 0.5
        return row_from_df

    row_from_df['Predicted Value'] = control_system_sim.output['Decision']
    return row_from_df


def normalize_dataframe(df):
    values = df.values
    min_max_scaler = MinMaxScaler()
    normalized_values = min_max_scaler.fit_transform(values)
    return pd.DataFrame(normalized_values, columns=df.columns)


def read_dataset(filepath):
    return pd.read_csv(filepath, sep=';')


def read_and_prepare_data(filepath, decision_mapping=None):
    dataset = read_dataset(filepath)
    if decision_mapping is not None:
        dataset['Decision'] = dataset['Decision'].map(decision_mapping)

    normalized_dataset = normalize_dataframe(dataset.drop('Decision', axis=1))
    normalized_dataset['Decision'] = dataset['Decision']
    normalized_dataset['Predicted Value'] = None
    normalized_dataset['Decision Fuzzy'] = None
    return normalized_dataset


def set_decisions(row, threshold):
    if row['Predicted Value'] > threshold:
        row['Decision Fuzzy'] = 1
    else:
        row['Decision Fuzzy'] = 0

    return row


def process_results(df):
    df = df.apply(set_decisions, threshold=0.5, axis=1)
    return accuracy_score(df['Decision'], df['Decision Fuzzy'])


def main():
    universe = np.arange(0, 1, 0.001)
    features_names = ['F0', 'F1', 'F2', 'F3']
    antecedents = create_antecedents(universe, features_names)
    for antecedent in antecedents:
        center_points = [0.5, 0.5]
        widths = [-5, 5]
        create_terms_for_fuzzy_variable(antecedent, ['Low', 'High'],
                                        [sigmf(universe, center_point, width) for center_point, width in zip(center_points, widths)])

    consequent = create_consequents(universe, ['Decision'])
    center_points = [0.5, 0.5]
    widths = [-5, 5]
    create_terms_for_fuzzy_variable(consequent, ['Low', 'High'],
                                    [sigmf(universe, center_point, width) for center_point, width in zip(center_points, widths)])

    # first_rule_premise = TermAggregate(
    #                        TermAggregate(antecedents[0]['Low'], antecedents[1]['High'], 'and'),
    #                    antecedents[2]['Low'], 'or')

    first_rule_premise = antecedents[0]['Low'] & antecedents[1]['High'] | antecedents[2]['Low']

    # second_rule_premise = TermAggregate(
    #                        TermAggregate(antecedents[0]['High'], antecedents[2]['High'], 'and'),
    #                     antecedents[3]['High'], 'and')

    second_rule_premise = antecedents[0]['High'] & antecedents[2]['High'] & antecedents[3]['High']

    first_rule = Rule(first_rule_premise, consequent['Low'])
    second_rule = Rule(second_rule_premise, consequent['High'])

    control_system = ControlSystem([first_rule, second_rule])
    control_system_sim = ControlSystemSimulation(control_system)

    dataset = read_and_prepare_data('Data/Blood.csv', decision_mapping={'Donated': 0, 'Not': 1})
    dataset = dataset.apply(makePrediction, control_system_sim=control_system_sim, features_names=features_names, axis=1)
    accuracy = process_results(dataset)

    print(accuracy)


if __name__ == '__main__':
    main()
