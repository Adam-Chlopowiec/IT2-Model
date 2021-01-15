import numpy as np

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import scikitfuzzy.skfuzzy.controlType2 as ctrl

import pandas as pd
from scipy.stats import norm
import sympy as sy
from sympy import symbols, Eq, solve

class FuzzifierProgressive(object):

    def __init__(self, settings, d_results):
        self.features = []
        self.feature_labels = []
        self.x_range = np.arange(settings.set_min, settings.set_max, settings.fuzzy_sets_precision)
        self.d_results = d_results
        self.settings = settings
        self.fuzzify_parameters = []

        for i in range(0, self.settings.feature_numbers):
            feature_label = "F" + str(i)
            self.features.append(
                ctrl.Antecedent(self.x_range, feature_label))
            self.feature_labels.append(feature_label)

        self.decision = ctrl.Consequent(self.x_range, 'Decision')

    def gaussLeft(self, x, mean, sigma):
        y = np.zeros(len(x))
        idx = x <= mean
        y[idx] = 1 - np.exp(-((x[idx] - mean) ** 2.) / (2 * sigma ** 2.))
        return y

    def gaussianFunction(self, x, mean, sigma, center_l, center_r, bottom_l, bottom_r, sigma_l, sigma_r, sigma_offset = 0):
        y = np.zeros(len(x))
        if (center_l == -1) and (center_r == -1):
            idx = (x < bottom_r) & (x >= bottom_l)
            y[idx] =  np.exp(-((x - mean) ** 2.) / (2 * (sigma + sigma_offset) ** 2.))
        elif center_l == -1:  
            idx = (x >= bottom_l) & (x < center_r)
            y[idx] = np.exp(-((x[idx] - mean) ** 2.) / (2 * (sigma + sigma_offset) ** 2.))
            idx = (x >= center_r) & (x < bottom_r)
            y[idx] = 1 - np.exp(-((x[idx] - bottom_r) ** 2.) / (2 * (sigma_r - sigma_offset) ** 2.))
        elif center_r == -1:
            idx = (x >= bottom_l) & (x < center_l)
            y[idx] = 1 - np.exp(-((x[idx] - bottom_l) ** 2.) / (2 * (sigma_l - sigma_offset) ** 2.))
            idx = (x >= center_l) & (x < bottom_r)
            y[idx] = np.exp(-((x[idx] - mean) ** 2.) / (2 * (sigma + sigma_offset) ** 2.))
        else:
            idx = (x >= bottom_l) & (x < center_l)
            y[idx] = 1 - np.exp(-((x[idx] - bottom_l) ** 2.) / (2 * (sigma_l - sigma_offset) ** 2.))

            idx = (x >= center_l) & (x < center_r)
            y[idx] = np.exp(-((x[idx] - mean) ** 2.) / (2 * (sigma + sigma_offset) ** 2.))
          
            idx = (x >= center_r) & (x < bottom_r)
            y[idx] = 1 - np.exp(-((x[idx] - bottom_r) ** 2.) / (2 * (sigma_r - sigma_offset) ** 2.))
        return y

    def gaussRight(self, x, mean, sigma):
        y = np.zeros(len(x))
        idx = x >= mean
        y[idx] = 1 - np.exp(-((x[idx] - mean) ** 2.) / (2 * sigma ** 2.))
        return y

    def leftGaussianValue(self, x, mean, sigma):
        if x <= mean:
            verylow_value = 1 - np.exp(-((x - mean) ** 2.) / (2 * sigma ** 2.))
        else:
            verylow_value = 0

        return verylow_value

    def gaussianValue(self, x, mean, sigma, center_l, center_r, bottom_l, bottom_r, sigma_l, sigma_r):
        
        if (center_l == -1) and (center_r == -1):       
            if (x < bottom_r) and (x >= bottom_l):
                y =  np.exp(-((x - mean) ** 2.) / (2 * sigma ** 2.))
            else:
                y = 0
        elif center_l == -1:
            if (x < center_r) and (x >= bottom_l):
                y = np.exp(-((x - mean) ** 2.) / (2 * sigma ** 2.))
            elif (x < bottom_r) and (x >= center_r):
                y = 1 - np.exp(-((x - bottom_r) ** 2.) / (2 * sigma_r ** 2.))
            else:
                y = 0
        elif center_r == -1:
            if (x < bottom_r) and (x >= center_l):
                y = np.exp(-((x - mean) ** 2.) / (2 * sigma ** 2.))
            elif (x < center_l) and (x >= bottom_l):
                y = 1 - np.exp(-((x - bottom_l) ** 2.) / (2 * sigma_l ** 2.))
            else:
                y = 0
        else:
            if (x >= bottom_l) and (x < center_l):
                y = 1 - np.exp(-((x - bottom_l) ** 2.) / (2 * sigma_l ** 2.))
            elif (x >= center_l) and (x < center_r):
                y = np.exp(-((x - mean) ** 2.) / (2 * sigma ** 2.))
            elif (x >= center_r) and (x < bottom_r):
                y = 1 - np.exp(-((x - bottom_r) ** 2.) / (2 * sigma_r ** 2.))
            else:
                y = 0
        return y

    def rightGaussianValue(self, x, mean, sigma):
        if x >= mean:
            veryhigh_value = 1 - np.exp(-((x - mean) ** 2.) / (2 * sigma ** 2.))
        else:
            veryhigh_value = 0
        
        return veryhigh_value

    def calculateBothSigma(self, mean_1, mean_2):
        x, sigma = symbols('x sigma')
        eq1 = Eq(sy.exp(-((x - mean_1) ** 2.) / (2 * sigma ** 2.)) - 0.5)
        eq2 = Eq(sy.exp(-((x - mean_2) ** 2.) / (2 * sigma ** 2.)) - 0.5)

        res = solve((eq1,eq2), (x, sigma))
        x_value = 1
        sigma_value = 1

        for x in res:
            if x[1] >= 0:
                if x[1] < sigma_value:
                    x_value = x[0]
                    sigma_value = x[1]

        return np.float64(x_value), np.float64(sigma_value)

    def calculateSigma(self, mean_1, mean_2, sigma_value):
        x, sigma = symbols('x sigma')
        eq1 = Eq(sy.exp(-((x - mean_1) ** 2.) / (2 * sigma ** 2.)) - 0.5)
        eq2 = Eq(sy.exp(-((x - mean_2) ** 2.) / (2 * sigma_value ** 2.)) - 0.5)

        res = solve((eq1,eq2), (x, sigma))
        x_value = 1
        sigma_value = 1

        for x in res:
            if x[1] >= 0:
                if x[1] < sigma_value:
                    x_value = x[0]
                    sigma_value = x[1]

        return np.float64(x_value), np.float64(sigma_value)

    def generateGausses3(self, x_range: np.ndarray, mean: float, sigma: float, sigma_offset: float = 0):
        verylow = self.gaussLeft(self.x_range, mean, sigma - sigma_offset)
        middle = self.gaussianFunction(self.x_range, mean, sigma, -1, -1, 0, 1.01, -1, -1, sigma_offset)
        veryhigh = self.gaussRight(self.x_range, mean, sigma - sigma_offset)
        return verylow, middle, veryhigh
    
    def generateGausses5(self, x_range: np.ndarray, mean: float, sigma_offset: float = 0):
        minus_mean = 1 - mean
        if (minus_mean * (1/4) >= (mean * (1/4))):
            self.__cr_middle, self.__center_sigma = self.calculateBothSigma(mean, mean + (mean * (1/4)))
        else:
            self.__cr_middle, self.__center_sigma = self.calculateBothSigma(mean, mean + (minus_mean * (1/4)))

        self.__cr_low, self.__left_sigma = self.calculateSigma(mean - (mean * (1/4)), mean, self.__center_sigma)
        _, self.__right_sigma = self.calculateSigma(mean, mean + (minus_mean * (1/4)), self.__center_sigma)

        verylow = self.gaussLeft(self.x_range, mean - (mean * (1/4)), self.__left_sigma - sigma_offset)
        low = self.gaussianFunction(self.x_range, mean - (mean * (1/4)), self.__left_sigma, -1, self.__cr_low, 0, mean, -1, self.__center_sigma, sigma_offset)
        middle = self.gaussianFunction(self.x_range, mean, self.__center_sigma, self.__cr_low, self.__cr_middle, mean - (mean * (1/4)), mean + (minus_mean * (1/4)), self.__left_sigma, self.__right_sigma, sigma_offset)
        high = self.gaussianFunction(self.x_range, mean + (minus_mean * (1/4)), self.__right_sigma, self.__cr_middle, -1, mean, 1.01, self.__center_sigma, -1, sigma_offset)
        veryhigh = self.gaussRight(self.x_range, mean + (minus_mean * (1/4)), self.__right_sigma - sigma_offset)
        return verylow, low, middle, high, veryhigh
    
    def generateGausses7(self, x_range: np.ndarray, mean: float, sigma_offset: float = 0):
        minus_mean = 1 - mean
        if (minus_mean * (1/10) >= (mean * (1/10))):
            self.__cr_middle, self.__center_sigma = self.calculateBothSigma(mean, mean + (mean * (1/10)))
        else:
            self.__cr_middle, self.__center_sigma = self.calculateBothSigma(mean, mean + (minus_mean * (1/10)))
            
        self.__cr_middlelow, self.__middle_left_sigma = self.calculateSigma(mean - (mean * (1/10)), mean, self.__center_sigma)
        self.__cr_low, self.__left_sigma = self.calculateSigma(mean - (mean * (4/10)), mean - (mean * (1/10)), self.__middle_left_sigma)

        _, self.__middle_right_sigma = self.calculateSigma(mean + (minus_mean * (1/10)), mean, self.__center_sigma)
        self.__cr_high, self.__right_sigma = self.calculateSigma(mean + (minus_mean * (4/10)), mean + (minus_mean * (1/10)), self.__middle_right_sigma)

        verylow = self.gaussLeft(self.x_range, mean - (mean * (4/10)), self.__left_sigma - sigma_offset)
        low = self.gaussianFunction(self.x_range, mean - (mean * (4/10)), self.__left_sigma, -1, self.__cr_low, 0, mean - (mean * (1/10)), -1, self.__middle_left_sigma, sigma_offset)
        middlelow = self.gaussianFunction(self.x_range, mean - (mean * (1/10)), self.__middle_left_sigma, self.__cr_low, self.__cr_middlelow, mean - (mean * (4/10)), mean, self.__left_sigma, self.__center_sigma, sigma_offset)

        middle = self.gaussianFunction(self.x_range, mean, self.__center_sigma, self.__cr_middlelow, self.__cr_middle, mean - (mean * (1/10)), mean + (minus_mean * (1/10)), self.__middle_left_sigma, self.__middle_right_sigma, sigma_offset)

        middlehigh = self.gaussianFunction(self.x_range, mean + (minus_mean * (1/10)), self.__middle_right_sigma, self.__cr_middle, self.__cr_high, mean, mean + (minus_mean * (4/10)), self.__center_sigma, self.__right_sigma, sigma_offset)
        high = self.gaussianFunction(self.x_range, mean + (minus_mean * (4/10)), self.__right_sigma, self.__cr_high, -1, mean + (minus_mean * (1/10)), 1.01, self.__middle_right_sigma, -1, sigma_offset)
        veryhigh = self.gaussRight(self.x_range, mean + (minus_mean * (4/10)), self.__right_sigma - sigma_offset)
        return verylow, low, middlelow, middle, middlehigh, high, veryhigh
    
    def generateGausses9(self, x_range: np.ndarray, mean: float, sigma_offset: float = 0):
        minus_mean = 1 - mean
        if (minus_mean * (1/20) >= (mean * (1/20))):
            self.__cr_middle, self.__center_sigma = self.calculateBothSigma(mean, mean + (mean * (1/20)))
        else:
            self.__cr_middle, self.__center_sigma = self.calculateBothSigma(mean, mean + (minus_mean * (1/20)))
            
        self.__cr_middlelow, self.__middle_left_sigma = self.calculateSigma(mean - (mean * (1/20)), mean, self.__center_sigma)
        self.__cr_middlelowminus, self.__middle_left_minus_sigma = self.calculateSigma(mean - (mean * (4/20)), mean - (mean * (1/20)), self.__middle_left_sigma)
        self.__cr_low, self.__left_sigma = self.calculateSigma(mean - (mean * (10/20)), mean - (mean * (4/20)), self.__middle_left_minus_sigma)

        _, self.__middle_right_sigma = self.calculateSigma(mean + (minus_mean * (1/20)), mean, self.__center_sigma)
        self.__cr_middlehighplus, self.__middle_right_plus_sigma = self.calculateSigma(mean + (minus_mean * (4/20)), mean + (minus_mean * (1/20)), self.__middle_right_sigma)
        self.__cr_high, self.__right_sigma = self.calculateSigma(mean + (minus_mean * (10/20)), mean + (minus_mean * (4/20)), self.__middle_right_plus_sigma)

        verylow = self.gaussLeft(self.x_range, mean - (mean * (10/20)), self.__left_sigma - sigma_offset)
        low = self.gaussianFunction(self.x_range, mean - (mean * (10/20)), self.__left_sigma, -1, self.__cr_low, 0, mean - (mean * (4/20)), -1, self.__middle_left_minus_sigma, sigma_offset)
        middlelowminus = self.gaussianFunction(self.x_range, mean - (mean * (4/20)), self.__middle_left_minus_sigma, self.__cr_low, self.__cr_middlelowminus, mean - (mean * (10/20)), mean - (mean * (1/20)), self.__left_sigma, self.__middle_left_sigma, sigma_offset)
        middlelow = self.gaussianFunction(self.x_range, mean - (mean * (1/20)), self.__middle_left_sigma, self.__cr_middlelowminus, self.__cr_middlelow, mean - (mean * (4/20)), mean, self.__middle_left_minus_sigma, self.__center_sigma, sigma_offset)

        middle = self.gaussianFunction(self.x_range, mean, self.__center_sigma, self.__cr_middlelow, self.__cr_middle, mean - (mean * (1/20)), mean + (minus_mean * (1/20)), self.__middle_left_sigma, self.__middle_right_sigma, sigma_offset)

        middlehigh = self.gaussianFunction(self.x_range, mean + (minus_mean * (1/20)), self.__middle_right_sigma, self.__cr_middle, self.__cr_high, mean, mean + (minus_mean * (4/20)), self.__center_sigma, self.__middle_right_plus_sigma, sigma_offset)
        middlehighplus = self.gaussianFunction(self.x_range, mean + (minus_mean *(4/20)), self.__middle_right_plus_sigma, self.__cr_middlehighplus, self.__cr_high, mean  + (minus_mean * (1/20)), mean  + (minus_mean * (10/20)), self.__middle_right_sigma, self.__right_sigma, sigma_offset)
        high = self.gaussianFunction(self.x_range, mean + (minus_mean * (10/20)), self.__right_sigma, self.__cr_high, -1, mean + (minus_mean * (4/20)), 1.01, self.__middle_right_plus_sigma, -1, sigma_offset)
        veryhigh = self.gaussRight(self.x_range, mean + (minus_mean * (10/20)), self.__right_sigma - sigma_offset)
        return verylow, low, middlelowminus, middlelow, middle, middlehigh, middlehighplus, high, veryhigh
    
    def generateGausses11(self, x_range: np.ndarray, mean: float, sigma_offset: float = 0):
        minus_mean = 1 - mean
        if (minus_mean * (1/35) >= (mean * (1/35))):
            self.__cr_middle, self.__center_sigma = self.calculateBothSigma(mean, mean + (mean * (1/35)))
        else:
            self.__cr_middle, self.__center_sigma = self.calculateBothSigma(mean, mean + (minus_mean * (1/35)))

        self.__cr_middlelowplus, self.__middle_left_plus_sigma = self.calculateSigma(mean - (mean * (1/35)), mean, self.__center_sigma)
        self.__cr_middlelow, self.__middle_left_sigma = self.calculateSigma(mean - (mean * (4/35)), mean - (mean * (1/35)), self.__middle_left_plus_sigma)
        self.__cr_middlelowminus, self.__middle_left_minus_sigma = self.calculateSigma(mean - (mean * (10/35)), mean - (mean * (4/35)), self.__middle_left_sigma)
        self.__cr_low, self.__left_sigma = self.calculateSigma(mean - (mean * (20/35)), mean - (mean * (10/35)), self.__middle_left_minus_sigma)

        _, self.__middle_right_sigma = self.calculateSigma(mean + (minus_mean * (1/35)), mean, self.__center_sigma)
        self.__cr_middlehighplus, self.__middle_right_plus_sigma = self.calculateSigma(mean + (minus_mean * (4/35)), mean + (minus_mean * (1/35)), self.__middle_right_sigma)
        self.__cr_high, self.__right_sigma = self.calculateSigma(mean + (minus_mean * (10/35)), mean + (minus_mean * (4/35)), self.__middle_right_plus_sigma)
        self.__cr_middlehighminus, self.__middle_right_minus_sigma = self.calculateSigma(mean + (minus_mean * (20/35)), mean + (minus_mean * (10/35)), self.__right_sigma)

        verylow = self.gaussLeft(self.x_range, mean - (mean * (20/35)), self.__left_sigma - sigma_offset)
        low = self.gaussianFunction(self.x_range, mean - (mean * (20/35)), self.__left_sigma, -1, self.__cr_low, 0, mean - (mean * (10/35)), -1, self.__middle_left_minus_sigma, sigma_offset)
        middlelowminus = self.gaussianFunction(self.x_range, mean - (mean * (10/35)), self.__middle_left_minus_sigma, self.__cr_low, self.__cr_middlelowminus, mean - (mean * (20/35)), mean - (mean * (4/35)), self.__left_sigma, self.__middle_left_sigma, sigma_offset)
        middlelow = self.gaussianFunction(self.x_range, mean - (mean * (4/35)), self.__middle_left_sigma, self.__cr_middlelowminus, self.__cr_middlelow, mean - (mean * (10/35)), mean - (mean * (1/35)), self.__middle_left_minus_sigma, self.__middle_left_plus_sigma, sigma_offset)
        middlelowplus = self.gaussianFunction(self.x_range, mean - (mean * (1/35)), self.__middle_left_plus_sigma, self.__cr_middlelow, self.__cr_middlelowplus, mean - (mean * (4/35)), mean, self.__middle_left_sigma, self.__center_sigma, sigma_offset)

        middle = self.gaussianFunction(self.x_range, mean, self.__center_sigma, self.__cr_middlelow, self.__cr_middle, mean - (mean * (1/35)), mean + (minus_mean * (1/35)), self.__middle_left_sigma, self.__middle_right_sigma, sigma_offset)

        middlehigh = self.gaussianFunction(self.x_range, mean + (minus_mean * (1/35)), self.__middle_right_sigma, self.__cr_middle, self.__cr_high, mean, mean + (minus_mean * (4/35)), self.__center_sigma, self.__middle_right_plus_sigma, sigma_offset)
        middlehighplus = self.gaussianFunction(self.x_range, mean + (minus_mean * (4/35)), self.__middle_right_plus_sigma, self.__cr_middlehighplus, self.__cr_high, mean  + (minus_mean * (1/35)), mean  + (minus_mean * (10/35)), self.__middle_right_sigma, self.__right_sigma, sigma_offset)
        high = self.gaussianFunction(self.x_range, mean + (minus_mean * (10/35)), self.__right_sigma, self.__cr_high, self.__cr_middlehighminus, mean + (minus_mean * (4/35)), mean + (minus_mean * (20/35)), self.__middle_right_plus_sigma, self.__middle_right_minus_sigma, sigma_offset)
        middlehighminus = self.gaussianFunction(self.x_range, mean + (minus_mean * (20/35)), self.__middle_right_minus_sigma, self.__cr_middlehighminus, -1, mean  + (minus_mean * (10/35)), 1.01, self.__right_sigma, -1, sigma_offset)
        veryhigh = self.gaussRight(self.x_range, mean + (minus_mean * (20/35)), self.__middle_right_minus_sigma - sigma_offset)
        return verylow, low, middlelowminus, middlelow, middlelowplus, middle, middlehighminus, middlehigh, middlehighplus, high, veryhigh
    
    def numbersToRowSets(self, idx, values, mean, sigma_offset = 0):
        values = values.values
        return_array = []

        if mean == -1:
            mean, _ = norm.fit(values)
        if mean == -2:
            mean = 0.5

        _, sigma = norm.fit(values)

        if self.settings.show_results:
            print("Feature " + str(idx) + ":")
            print("\tMean: " + str(mean))
            print("\tSigma: " + str(sigma))

        minus_mean = 1 - mean
        
        if self.settings.gausses == 3:
            lower_functions = self.generateGausses3(self.x_range, mean, sigma, sigma_offset * (-1))
            upper_functions = self.generateGausses3(self.x_range, mean, sigma, sigma_offset)
            names = (self.settings.verylow, self.settings.middle, self.settings.veryhigh)
            for lower, upper, name in zip(lower_functions, upper_functions, names):
                self.features[idx][name] = (lower, upper)

        elif self.settings.gausses == 5:
            lower_functions = self.generateGausses5(self.x_range, mean, sigma_offset * (-1))
            upper_functions = self.generateGausses5(self.x_range, mean, sigma_offset)
            names = (self.settings.verylow, self.settings.low, self.settings.middle, self.settings.high, self.settings.veryhigh)
            for lower, upper, name in zip(lower_functions, upper_functions, names):
                self.features[idx][name] = (lower, upper)

        elif self.settings.gausses == 7:
            lower_functions = self.generateGausses7(self.x_range, mean, sigma_offset * (-1))
            upper_functions = self.generateGausses7(self.x_range, mean, sigma_offset)
            names = (self.settings.verylow, self.settings.low, self.settings.middlelow, self.settings.middle, self.settings.middlehigh, self.settings.high, self.settings.veryhigh)
            for lower, upper, name in zip(lower_functions, upper_functions, names):
                self.features[idx][name] = (lower, upper)
            
        elif self.settings.gausses == 9:
            lower_functions = self.generateGausses9(self.x_range, mean, sigma_offset * (-1))
            upper_functions = self.generateGausses9(self.x_range, mean, sigma_offset)
            names = (self.settings.verylow, self.settings.low, self.settings.middlelowminus, self.settings.middlelow, self.settings.middle, self.settings.middlehigh, self.settings.middlehighplus, self.settings.high, self.settings.veryhigh)
            for lower, upper, name in zip(lower_functions, upper_functions, names):
                self.features[idx][name] = (lower, upper)

        elif self.settings.gausses == 11:
            lower_functions = self.generateGausses11(self.x_range, mean, sigma_offset * (-1))
            upper_functions = self.generateGausses11(self.x_range, mean, sigma_offset)
            names = (self.settings.verylow, self.settings.low, self.settings.middlelowminus, self.settings.middlelow, self.settings.middlelowplus, self.settings.middle, self.settings.middlehighminus, self.settings.middlehigh, self.settings.middlehighplus, self.settings.high, self.settings.veryhigh)
            for lower, upper, name in zip(lower_functions, upper_functions, names):
                self.features[idx][name] = (lower, upper)
            
        self.fuzzify_parameters.append(mean)
        self.fuzzify_parameters.append(sigma)
                              
        for x in values:
            verylow_value = low_value = middlelowminus_value = middlelow_value = middlelowplus_value = middle_value = middlehighminus_value = middlehigh_value = middlehighplus_value = high_value = veryhigh_value = 0

            if self.settings.gausses == 3:
                verylow_value = self.leftGaussianValue(x, mean, sigma)
                middle_value = self.gaussianValue(x, mean, sigma, -1, -1, 0, 1.01, -1, -1)
                veryhigh_value = self.rightGaussianValue(x, mean, sigma)
                
            if self.settings.gausses == 5:
                middle_value = self.gaussianValue(x, mean, self.__center_sigma, self.__cr_low, self.__cr_middle, mean - (mean * (1/4)), mean + (minus_mean * (1/4)), self.__left_sigma, self.__right_sigma)
                verylow_value = self.leftGaussianValue(x, mean - (mean * (1/4)), self.__left_sigma)
                low_value = self.gaussianValue(x, mean - (mean * (1/4)), self.__left_sigma, -1, self.__cr_low, 0, mean, -1, self.__center_sigma)
                high_value = self.gaussianValue(x, mean + (minus_mean * (1/4)), self.__right_sigma, self.__cr_middle, -1, mean, 1.01, self.__center_sigma, -1)
                veryhigh_value = self.rightGaussianValue(x, mean + (minus_mean * (1/4)), self.__right_sigma)

            elif self.settings.gausses == 7:
                middle_value = self.gaussianValue(x, mean, self.__center_sigma, self.__cr_middlelow, self.__cr_middle, mean - (mean * (1/10)), mean + (minus_mean * (1/10)), self.__middle_left_sigma, self.__middle_right_sigma)
                verylow_value = self.leftGaussianValue(x, mean - (mean * (4/10)), self.__left_sigma)
                low_value = self.gaussianValue(x, mean - (mean * (4/10)), self.__left_sigma, -1, self.__cr_low, 0, mean - (mean * (1/10)), -1, self.__middle_left_sigma)
                middlelow_value = self.gaussianValue(x, mean - (mean * (1/10)), self.__middle_left_sigma, self.__cr_low, self.__cr_middlelow, mean - (mean * (4/10)), mean, self.__left_sigma, self.__center_sigma)
                middlehigh_value = self.gaussianValue(x, mean + (minus_mean * (1/10)), self.__middle_right_sigma, self.__cr_middle, self.__cr_high, mean, mean + (minus_mean * (4/10)), self.__center_sigma, self.__right_sigma)
                high_value = self.gaussianValue(x, mean + (minus_mean * (4/10)), self.__right_sigma, self.__cr_high, -1, mean + (minus_mean * (1/10)), 1.01, self.__middle_right_sigma, -1)
                veryhigh_value = self.rightGaussianValue(x, mean + (minus_mean * (4/10)), self.__right_sigma)
                
            elif self.settings.gausses == 9:
                verylow_value = self.leftGaussianValue(x, mean - (mean * (10/20)), self.__left_sigma)
                low_value = self.gaussianValue(x, mean - (mean * (10/20)), self.__left_sigma, -1, self.__cr_low, 0, mean - (mean * (4/20)), -1, self.__middle_left_minus_sigma)
                middlelowminus_value = self.gaussianValue(x, mean - (mean * (4/20)), self.__middle_left_minus_sigma, self.__cr_low, self.__cr_middlelowminus, mean - (mean * (10/20)), mean - (mean * (1/20)), self.__left_sigma, self.__middle_left_sigma)
                middlelow_value = self.gaussianValue(x, mean - (mean * (1/20)), self.__middle_left_sigma, self.__cr_middlelowminus, self.__cr_middlelow, mean - (mean * (4/20)), mean, self.__middle_left_minus_sigma, self.__center_sigma)

                middle_value = self.gaussianValue(x, mean, self.__center_sigma, self.__cr_middlelow, self.__cr_middle, mean - (mean * (1/20)), mean + (minus_mean * (1/20)), self.__middle_left_sigma, self.__middle_right_sigma)

                middlehigh_value = self.gaussianValue(x, mean + (minus_mean * (1/20)), self.__middle_right_sigma, self.__cr_middle, self.__cr_high, mean, mean + (minus_mean * (4/20)), self.__center_sigma, self.__middle_right_plus_sigma)
                middlehighplus_value = self.gaussianValue(x, mean + (minus_mean *(4/20)), self.__middle_right_plus_sigma, self.__cr_middlehighplus, self.__cr_high, mean  + (minus_mean * (1/20)), mean  + (minus_mean * (10/20)), self.__middle_right_sigma, self.__right_sigma)
                high_value = self.gaussianValue(x, mean + (minus_mean * (10/20)), self.__right_sigma, self.__cr_high, -1, mean + (minus_mean * (4/20)), 1.01, self.__middle_right_plus_sigma, -1)
                veryhigh_value = self.rightGaussianValue(x, mean + (minus_mean * (10/20)), self.__right_sigma)
                
            elif self.settings.gausses == 11:
                verylow_value = self.leftGaussianValue(x, mean - (mean * (20/35)), self.__left_sigma)
                low_value = self.gaussianValue(x, mean - (mean * (20/35)), self.__left_sigma, -1, self.__cr_low, 0, mean - (mean * (10/35)), -1, self.__middle_left_minus_sigma)
                middlelowminus_value = self.gaussianValue(x, mean - (mean * (10/35)), self.__middle_left_minus_sigma, self.__cr_low, self.__cr_middlelowminus, mean - (mean * (20/35)), mean - (mean * (4/35)), self.__left_sigma, self.__middle_left_sigma)
                middlelow_value = self.gaussianValue(x, mean - (mean * (4/35)), self.__middle_left_sigma, self.__cr_middlelowminus, self.__cr_middlelow, mean - (mean * (10/35)), mean - (mean * (1/35)), self.__middle_left_minus_sigma, self.__middle_left_plus_sigma)
                middlelowplus_value = self.gaussianValue(x, mean - (mean * (1/35)), self.__middle_left_plus_sigma, self.__cr_middlelow, self.__cr_middlelowplus, mean - (mean * (4/35)), mean, self.__middle_left_sigma, self.__center_sigma)

                middle_value = self.gaussianValue(x, mean, self.__center_sigma, self.__cr_middlelow, self.__cr_middle, mean - (mean * (1/35)), mean + (minus_mean * (1/35)), self.__middle_left_sigma, self.__middle_right_sigma)

                middlehigh_value = self.gaussianValue(x, mean + (minus_mean * (1/35)), self.__middle_right_sigma, self.__cr_middle, self.__cr_high, mean, mean + (minus_mean * (4/35)), self.__center_sigma, self.__middle_right_plus_sigma)
                middlehighplus_value = self.gaussianValue(x, mean + (minus_mean * (4/35)), self.__middle_right_plus_sigma, self.__cr_middlehighplus, self.__cr_high, mean  + (minus_mean * (1/35)), mean  + (minus_mean * (10/35)), self.__middle_right_sigma, self.__right_sigma)
                high_value = self.gaussianValue(x, mean + (minus_mean * (10/35)), self.__right_sigma, self.__cr_high, self.__cr_middlehighminus, mean + (minus_mean * (4/35)), mean + (minus_mean * (20/35)), self.__middle_right_plus_sigma, self.__middle_right_minus_sigma)
                middlehighminus_value = self.gaussianValue(x, mean + (minus_mean * (20/35)), self.__middle_right_minus_sigma, self.__cr_middlehighminus, -1, mean  + (minus_mean * (10/35)), 1.01, self.__right_sigma, -1)
                veryhigh_value = self.rightGaussianValue(x, mean + (minus_mean * (20/35)), self.__middle_right_minus_sigma)

            max_value = max([verylow_value, low_value, middlelowminus_value, middlelow_value, middlelowplus_value, middle_value, middlehighminus_value, middlehigh_value, middlehighplus_value, high_value, veryhigh_value])
            
            if max_value == verylow_value:
                return_value = self.features[idx][self.settings.verylow].label    
            elif max_value == low_value:
                 return_value = self.features[idx][self.settings.low].label                   
            elif max_value == middlelowminus_value:
                 return_value = self.features[idx][self.settings.middlelowminus].label  
            elif max_value == middlelow_value:
                 return_value = self.features[idx][self.settings.middlelow].label  
            elif max_value == middlelowplus_value:
                 return_value = self.features[idx][self.settings.middlelowplus].label                                                     
            elif max_value == middle_value:
                 return_value = self.features[idx][self.settings.middle].label      
            elif max_value == middlehighminus_value:
                 return_value = self.features[idx][self.settings.middlehighminus].label      
            elif max_value == middlehigh_value:
                 return_value = self.features[idx][self.settings.middlehigh].label      
            elif max_value == middlehighplus_value:
                 return_value = self.features[idx][self.settings.middlehighplus].label      
            elif max_value == high_value:
                 return_value = self.features[idx][self.settings.high].label                                                                                      
            elif max_value == veryhigh_value:
                 return_value = self.features[idx][self.settings.veryhigh].label                                                                                      

            return_array.append(return_value)

        return return_array
 
    def presentFuzzyFeature_Charts(self):
        for x in self.features:
            x.view()

    def fuzzify(self, features_table, mean_param, sigma_offset = 0):
        if isinstance(mean_param, (int, np.integer)):
            for idx, x in enumerate(self.features):
                features_table[x.label] = self.numbersToRowSets(idx, features_table[x.label], mean_param, sigma_offset)
        
        else:
            for idx, x in enumerate(self.features):
                features_table[x.label] = self.numbersToRowSets(idx, features_table[x.label], mean_param[idx], sigma_offset)
        
        if self.settings.show_results:
            self.presentFuzzyFeature_Charts()

        return features_table, self.feature_labels, self.features, self.decision, self.fuzzify_parameters
