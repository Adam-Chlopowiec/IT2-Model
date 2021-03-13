from Scripts.LoadCSV import LoadCSV
from Scripts.Fuzzify import Fuzzify as Fuzzify
from Scripts.ValueTest import ValueTest as ValueTest
import Settings.GeneralSettings as generalSettings
from Settings.SettingsHaberman import Settings
import warnings
warnings.simplefilter("ignore")


def main():
    generalSettings.gausses = 5
    generalSettings.style = "Gaussian Equal"
    generalSettings.adjustment_value = - 1
    settings = Settings(generalSettings)

    loadCSV = LoadCSV()
    samples_stats, train_stats, test_stats, train_samples = loadCSV.worker(settings)

    fuzzify = Fuzzify()
    changed_decisions, features_number_after_reduct, implicants_number, fuzzify_parameters, times = \
        fuzzify.worker(settings, settings.adjustment_value, sigma_offset=0.1)

    valueTest = ValueTest(settings, settings.s_function_width, settings.is_training)
    valueTest.sOptymalizationWorker(settings, settings.default_s_value, settings.show_results, center_offset=0.15)

    valueTest = ValueTest(settings, settings.s_function_width, not settings.is_training)
    valueTest.sOptymalizationWorker(settings, settings.default_s_value, settings.show_results, center_offset=0.15)


if __name__ == '__main__':
    main()