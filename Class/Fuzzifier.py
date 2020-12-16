import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
from scipy.stats import norm
import sympy as sy
from sympy import symbols, Eq, solve

class Fuzzifier(object):

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
                (ctrl.Antecedent(self.x_range, feature_label), ctrl.Antecedent(self.x_range, feature_label))) # TODO: Możliwe, że trzeba tu będzie zrobić tupla Atecedentów
            self.feature_labels.append(feature_label)

        self.decision = ctrl.Consequent(self.x_range, 'Decision')

    def gaussLeft(self, x, mean, sigma):
        y = np.zeros(len(x))
        idx = x <= mean
        y[idx] = 1 - np.exp(-((x[idx] - mean) ** 2.) / sigma ** 2.)
        return y

    def gaussianFunction(self, x, mean, sigma, center_l, center_r, bottom_l, bottom_r):
        y = np.zeros(len(x))

        if (center_l == -1) and (center_r == -1):
            idx = (x < bottom_r) & (x >= bottom_l)
            y[idx] =  np.exp(-((x - mean) ** 2.) / sigma ** 2.)
        elif center_l == -1:  
            idx = (x >= bottom_l) & (x < center_r)
            y[idx] = np.exp(-((x[idx] - mean) ** 2.) / sigma ** 2.)
            idx = (x >= center_r) & (x < bottom_r)
            y[idx] = 1 - np.exp(-((x[idx] - bottom_r) ** 2.) / sigma ** 2.)
        elif center_r == -1:
            idx = (x >= bottom_l) & (x < center_l)
            y[idx] = 1 - np.exp(-((x[idx] - bottom_l) ** 2.) / sigma ** 2.)
            idx = (x >= center_l) & (x < bottom_r)
            y[idx] = np.exp(-((x[idx] - mean) ** 2.) / sigma ** 2.)
        else:
            idx = (x >= bottom_l) & (x < center_l)
            y[idx] = 1 - np.exp(-((x[idx] - bottom_l) ** 2.) / sigma ** 2.)

            idx = (x >= center_l) & (x < center_r)
            y[idx] = np.exp(-((x[idx] - mean) ** 2.) / sigma ** 2.)
          
            idx = (x >= center_r) & (x < bottom_r)
            y[idx] = 1 - np.exp(-((x[idx] - bottom_r) ** 2.) / sigma ** 2.)
        return y

    def gaussRight(self, x, mean, sigma):
        y = np.zeros(len(x))
        idx = x >= mean
        y[idx] = 1 - np.exp(-((x[idx] - mean) ** 2.) / sigma ** 2.)
        return y

    def leftGaussianValue(self, x, mean, sigma):
        if x <= mean:
            verylow_value = 1 - np.exp(-((x - mean) ** 2.) / sigma ** 2.)
        else:
            verylow_value = 0

        return verylow_value

    def gaussianValue(self, x, mean, sigma, center_l, center_r, bottom_l, bottom_r):

        if (center_l == -1) and (center_r == -1):       
            if (x < bottom_r) and (x >= bottom_l):
                y =  np.exp(-((x - mean) ** 2.) / sigma ** 2.)
            else:
                y = 0
        elif center_l == -1:
            if (x < center_r) and (x >= bottom_l):
                y = np.exp(-((x - mean) ** 2.) / sigma ** 2.)
            elif (x < bottom_r) and (x >= center_r):
                y = 1 - np.exp(-((x - bottom_r) ** 2.) / sigma ** 2.)
            else:
                y = 0
        elif center_r == -1:
            if (x < bottom_r) and (x >= center_l):
                y = np.exp(-((x - mean) ** 2.) / sigma ** 2.)
            elif (x < center_l) and (x >= bottom_l):
                y = 1 - np.exp(-((x - bottom_l) ** 2.) / sigma ** 2.)
            else:
                y = 0
        else:
            if (x >= bottom_l) and (x < center_l):
                y = 1 - np.exp(-((x - bottom_l) ** 2.) / sigma ** 2.)
            elif (x >= center_l) and (x < center_r):
                y = np.exp(-((x - mean) ** 2.) / sigma ** 2.)
            elif (x >= center_r) and (x < bottom_r):
                y = 1 - np.exp(-((x - bottom_r) ** 2.) / sigma ** 2.)
            else:
                y = 0
        return y

    def rightGaussianValue(self, x, mean, sigma):
        if x >= mean:
            veryhigh_value = 1 - np.exp(-((x - mean) ** 2.) / sigma ** 2.)
        else:
            veryhigh_value = 0
        
        return veryhigh_value

    def calculateSigma(self, mean_1, mean_2):
        x, sigma = symbols('x sigma')
        eq1 = Eq(sy.exp(-((x - mean_1) ** 2.) / sigma ** 2.) - 0.5)
        eq2 = Eq(sy.exp(-((x - mean_2) ** 2.) / sigma ** 2.) - 0.5)

        res = solve((eq1,eq2), (x, sigma))
        for x in res:
            if x[1] >= 0:
                x_value = x[0]
                sigma_value = x[1]
                break

        return np.float64(x_value), np.float64(sigma_value)
    
    def generateGausses3(self, x_range: np.ndarray, mean: float, sigma: float, sigma_offset: float) -> tuple:
        verylow = self.gaussLeft(self.x_range, mean, sigma + sigma_offset)
        middle = self.gaussianFunction(self.x_range, mean, sigma + sigma_offset, -1, -1, 0, 1.01)
        veryhigh = self.gaussRight(self.x_range, mean, sigma + sigma_offset)
        return verylow, middle, veryhigh
    
    def generateGausses5(self, x_range: np.ndarray, mean: float, sigma_offset: float) -> tuple:
        self.cr_low, sigma = self.calculateSigma(mean * (1/2), mean)
        verylow = self.gaussLeft(self.x_range, mean * (1/2), sigma + sigma_offset)
        low = self.gaussianFunction(self.x_range, mean * (1/2), sigma + sigma_offset, -1, self.cr_low, 0, mean)

        self.cr_middle, _ = self.calculateSigma(mean, mean * (3/2))
        middle = self.gaussianFunction(self.x_range, mean, sigma + sigma_offset, self.cr_low, self.cr_middle, mean * (1/2), mean * (3/2))

        high = self.gaussianFunction(self.x_range, mean * (3/2), sigma + sigma_offset, self.cr_middle, -1, mean, 1.01)  
        veryhigh = self.gaussRight(self.x_range, mean * (3/2), sigma + sigma_offset)
        return sigma, verylow, low, middle, high, veryhigh
    
    def generateGausses7(self, x_range: np.ndarray, mean: float, sigma_offset: float) -> tuple:
        self.cr_low, sigma = self.calculateSigma(mean * (1/3), mean * (2/3))
        verylow = self.gaussLeft(self.x_range, mean * (1/3), sigma + sigma_offset)
        low = self.gaussianFunction(self.x_range, mean * (1/3), sigma + sigma_offset, -1, self.cr_low, 0, mean * (2/3))

        self.cr_middlelow, sigma = self.calculateSigma(mean * (2/3), mean)
        middlelow = self.gaussianFunction(self.x_range, mean * (2/3), sigma + sigma_offset, self.cr_low, self.cr_middlelow, mean * (1/3), mean)

        self.cr_middle, sigma = self.calculateSigma(mean, mean * (4/3))
        middle = self.gaussianFunction(self.x_range, mean, sigma + sigma_offset, self.cr_middlelow, self.cr_middle, mean * (2/3), mean * (4/3))

        self.cr_middlehigh, sigma = self.calculateSigma(mean * (4/3), mean * (5/3))
        middlehigh = self.gaussianFunction(self.x_range, mean * (4/3), sigma + sigma_offset, self.cr_middle, self.cr_middlehigh, mean, mean * (5/3))

        high = self.gaussianFunction(self.x_range, mean * (5/3), sigma + sigma_offset, self.cr_middlehigh, -1, mean * (4/3), 1.01)  
        veryhigh = self.gaussRight(self.x_range, mean * (5/3), sigma + sigma_offset)
        return sigma, verylow, low, middlelow, middle, middlehigh, high, veryhigh
    
    def generateGausses9(self, x_range: np.ndarray, mean: float, sigma_offset: float) -> tuple:
        self.cr_low, sigma = self.calculateSigma(mean * (1/4), mean * (2/4))
        verylow = self.gaussLeft(self.x_range, mean * (1/4), sigma + sigma_offset)
        low = self.gaussianFunction(self.x_range, mean * (1/4), sigma + sigma_offset, -1, self.cr_low, 0, mean * (2/4))

        self.cr_middlelowminus, sigma = self.calculateSigma(mean * (2/4), mean * (3/4))
        middlelowminus = self.gaussianFunction(self.x_range, mean * (2/4), sigma + sigma_offset, self.cr_low, self.cr_middlelowminus, mean * (1/4), mean * (3/4))

        self.cr_middlelow, sigma = self.calculateSigma(mean * (3/4), mean)
        middlelow = self.gaussianFunction(self.x_range, mean * (3/4), sigma + sigma_offset, self.cr_middlelowminus, self.cr_middlelow, mean * (2/4), mean)

        self.cr_middle, sigma = self.calculateSigma(mean, mean * (5/4))
        middle = self.gaussianFunction(self.x_range, mean, sigma + sigma_offset, self.cr_middlelow, self.cr_middle, mean * (3/4), mean * (5/4))

        self.cr_middlehigh, sigma = self.calculateSigma(mean * (5/4), mean * (6/4))
        middlehigh = self.gaussianFunction(self.x_range, mean * (5/4), sigma + sigma_offset, self.cr_middle, self.cr_middlehigh, mean, mean * (6/4))

        self.cr_middlehighplus, sigma = self.calculateSigma(mean * (6/4), mean * (7/4))
        middlehighplus = self.gaussianFunction(self.x_range, mean * (6/4), sigma + sigma_offset, self.cr_middlehigh, self.cr_middlehighplus, mean * (5/4), mean * (7/4))

        high = self.gaussianFunction(self.x_range, mean * (7/4), sigma + sigma_offset, self.cr_middlehighplus, -1, mean * (6/4), 1.01)  
        veryhigh = self.gaussRight(self.x_range, mean * (7/4), sigma + sigma_offset)
        return sigma, verylow, low, middlelowminus, middlelow, middle, middlehigh, middlehighplus, high, veryhigh
    
    def generateGausses11(self, x_range: np.ndarray, mean: float, sigma_offset: float) -> tuple:
        self.cr_low, sigma = self.calculateSigma(mean * (1/5), mean * (2/5))
        verylow = self.gaussLeft(self.x_range, mean * (1/5), sigma + sigma_offset)
        low = self.gaussianFunction(self.x_range, mean * (1/5), sigma + sigma_offset, -1, self.cr_low, 0, mean * (2/5))

        self.cr_middlelowminus, sigma = self.calculateSigma(mean * (2/5), mean * (3/5))
        middlelowminus = self.gaussianFunction(self.x_range, mean * (2/5), sigma + sigma_offset, self.cr_low, self.cr_middlelowminus, mean * (1/5), mean * (3/5))

        self.cr_middlelow, sigma = self.calculateSigma(mean * (3/5), mean * (4/5))
        middlelow = self.gaussianFunction(self.x_range, mean * (3/5), sigma + sigma_offset, self.cr_middlelowminus, self.cr_middlelow, mean * (2/5), mean * (4/5))

        self.cr_middlelowplus, sigma = self.calculateSigma(mean * (4/5), mean * (5/5))
        middlelowplus = self.gaussianFunction(self.x_range, mean * (4/5), sigma + sigma_offset, self.cr_middlelow, self.cr_middlelowplus, mean * (3/5), mean * (5/5))

        self.cr_middle, sigma = self.calculateSigma(mean, mean * (6/5))
        middle = self.gaussianFunction(self.x_range, mean, sigma + sigma_offset, self.cr_middlelowplus, self.cr_middle, mean * (4/5), mean * (6/5))

        self.cr_middlehighminus, sigma = self.calculateSigma(mean * (6/5), mean * (7/5))
        middlehighminus = self.gaussianFunction(self.x_range, mean * (6/5), sigma + sigma_offset, self.cr_middle, self.cr_middlehighminus, mean * (5/5), mean * (7/5))

        self.cr_middlehigh, sigma = self.calculateSigma(mean * (7/5), mean * (8/5))
        middlehigh = self.gaussianFunction(self.x_range, mean * (7/5), sigma + sigma_offset, self.cr_middlehighminus, self.cr_middlehigh, mean * (6/5), mean * (8/5))

        self.cr_middlehighplus, sigma = self.calculateSigma(mean * (8/5), mean * (9/5))
        middlehighplus = self.gaussianFunction(self.x_range, mean * (8/5), sigma + sigma_offset, self.cr_middlehigh, self.cr_middlehighplus, mean * (7/5), mean * (9/5))

        high = self.gaussianFunction(self.x_range, mean * (9/5), sigma + sigma_offset, self.cr_middlehighplus, -1, mean * (8/5), 1.01)  
        veryhigh = self.gaussRight(self.x_range, mean * (9/5), sigma + sigma_offset)
        return sigma, verylow, low, middlelowminus, middlelow, middlelowplus, middle, middlehighminus, middlehigh, middlehighplus, high, veryhigh
    
    def gaussianFunValues3(self, x: float, mean: float, sigma: float, sigma_offset: float = 0) -> tuple:
        verylow_value = self.leftGaussianValue(x, mean, sigma + sigma_offset)
        middle_value = self.gaussianValue(x, mean, sigma + sigma_offset, -1, -1, 0, 1.01)
        veryhigh_value = self.rightGaussianValue(x, mean, sigma + sigma_offset)
        return verlow_value, middle_value, veryhigh_value
    
    def gaussianFunValues5(self, x: float, mean: float, sigma: float, sigma_offset: float = 0) -> tuple:
        verylow_value = self.leftGaussianValue(x, mean * (1/2), sigma + sigma_offset)
        low_value = self.gaussianValue(x, mean * (1/2), sigma + sigma_offset, -1, self.cr_low, 0, mean)
        middle_value = self.gaussianValue(x, mean, sigma + sigma_offset, self.cr_low, self.cr_middle, mean * (1/2), mean * (3/2))
        high_value = self.gaussianValue(x, mean * (3/2), sigma + sigma_offset, self.cr_middle, -1, mean, 1.01)  
        veryhigh_value = self.rightGaussianValue(x, mean * (3/2), sigma + sigma_offset)
        return verlow_value, low_value, middle_value, high_value, veryhigh_value
    
    def gaussianFunValues7(self, x: float, mean: float, sigma: float, sigma_offset: float = 0) -> tuple:
        verylow_value = self.leftGaussianValue(x, mean * (1/3), sigma + sigma_offset)
        low_value = self.gaussianValue(x, mean * (1/3), sigma + sigma_offset, -1, self.cr_low, 0, mean * (2/3))
        middlelow_value = self.gaussianValue(x, mean * (2/3), sigma + sigma_offset, self.cr_low, self.cr_middlelow, mean * (1/3), mean)
        middle_value = self.gaussianValue(x, mean, sigma + sigma_offset, self.cr_middlelow, self.cr_middle, mean * (2/3), mean * (4/3))
        middlehigh_value = self.gaussianValue(x, mean * (4/3), sigma + sigma_offset, self.cr_middle, self.cr_middlehigh, mean, mean * (5/3))
        high_value = self.gaussianValue(x, mean * (5/3), sigma + sigma_offset, self.cr_middlehigh, -1, mean * (4/3), 1.01)  
        veryhigh_value = self.rightGaussianValue(x, mean * (5/3), sigma + sigma_offset)
        return verylow_value, low_value, middlelow_value, middle_value, middlehigh_value, high_value, veryhigh_value
    
    def gaussianFunValues9(self, x: float, mean: float, sigma: float, sigma_offset: float = 0) -> tuple:
        verylow_value = self.leftGaussianValue(x, mean * (1/4), sigma + sigma_offset)
        low_value = self.gaussianValue(x, mean * (1/4), sigma + sigma_offset, -1, self.cr_low, 0, mean * (2/4))
        middlelowminus_value = self.gaussianValue(x, mean * (2/4), sigma + sigma_offset, self.cr_low, self.cr_middlelowminus, mean * (1/4), mean * (3/4))
        middlelow_value = self.gaussianValue(x, mean * (3/4), sigma + sigma_offset, self.cr_middlelowminus, self.cr_middlelow, mean * (2/4), mean)
        middle_value = self.gaussianValue(x, mean, sigma + sigma_offset, self.cr_middlelow, self.cr_middle, mean * (3/4), mean * (5/4))
        middlehigh_value = self.gaussianValue(x, mean * (5/4), sigma + sigma_offset, self.cr_middle, self.cr_middlehigh, mean, mean * (6/4))
        middlehighplus_value = self.gaussianValue(x, mean * (6/4), sigma + sigma_offset, self.cr_middlehigh, self.cr_middlehighplus, mean * (5/4), mean * (7/4))
        high_value = self.gaussianValue(x, mean * (7/4), sigma + sigma_offset, self.cr_middlehighplus, -1, mean * (6/4), 1.01)  
        veryhigh_value = self.rightGaussianValue(x, mean * (7/4), sigma + sigma_offset)
        return verylow_value, low_value, middlelowminus_value, middlelow_value, middle_value, middlehigh_value, middlehighplus_value, high_value, veryhigh_value
    
    def gaussianFunValues11(self, x: float, mean: float, sigma: float, sigma_offset: float = 0) -> tuple:
        verylow_value = self.leftGaussianValue(x, mean * (1/5), sigma + sigma_offset)
        low_value = self.gaussianValue(x, mean * (1/5), sigma + sigma_offset, -1, self.cr_low, 0, mean * (2/5))
        middlelowminus_value = self.gaussianValue(x, mean * (2/5), sigma + sigma_offset, self.cr_low, self.cr_middlelowminus, mean * (1/5), mean * (3/5))
        middlelow_value = self.gaussianValue(x, mean * (3/5), sigma + sigma_offset, self.cr_middlelowminus, self.cr_middlelow, mean * (2/5), mean * (4/5))
        middlelowplus_value = self.gaussianValue(x, mean * (4/5), sigma + sigma_offset, self.cr_middlelow, self.cr_middlelowplus, mean * (3/5), mean * (5/5))
        middle_value = self.gaussianValue(x, mean, sigma + sigma_offset, self.cr_middlelowplus, self.cr_middle, mean * (4/5), mean * (6/5))
        middlehighminus_value = self.gaussianValue(x, mean * (6/5), sigma + sigma_offset, self.cr_middle, self.cr_middlehighminus, mean * (5/5), mean * (7/5))
        middlehigh_value = self.gaussianValue(x, mean * (7/5), sigma + sigma_offset, self.cr_middlehighminus, self.cr_middlehigh, mean * (6/5), mean * (8/5))
        middlehighplus_value = self.gaussianValue(x, mean * (8/5), sigma + sigma_offset, self.cr_middlehigh, self.cr_middlehighplus, mean * (7/5), mean * (9/5))
        high_value = self.gaussianValue(x, mean * (9/5), sigma + sigma_offset, self.cr_middlehighplus, -1, mean * (8/5), 1.01)  
        veryhigh_value = self.rightGaussianValue(x, mean * (9/5), sigma + sigma_offset)
        return verylow_value, low_value, middlelowminus_value, middlelow_value, middlelowplus_value, middle_value, middlehighminus_value, middlehigh_value, middlehighplus_value, high_value, veryhigh_value
        

    def numbersToRowSets(self, idx, values, mean):
        values = values.values
        return_array = []

        if mean == -1:
            mean, _ = norm.fit(values)
        if mean == -2:
            mean = 0.5
        
        _, sigma = norm.fit(values) 
        
        SIGMA_OFFSET = 0.1

        if self.settings.show_results:
            print("Feature " + str(idx) + ":")
            print("\tMean: " + str(mean))
            print("\tSigma: " + str(sigma))

        if self.settings.gausses == 3:
            lower_functions = self.generateGausses3(self.x_range, mean, sigma, SIGMA_OFFSET * (-1))
            upper_functions = self.generateGausses3(self.x_range, mean, sigma, SIGMA_OFFSET)
            names = (self.settings.verylow, self.settings.middle, self.settings.veryhigh)
            for lower, upper, name in zip(lower_functions, upper_functions, names):
                self.features[idx][0][name] = lower
                self.features[idx][1][name] = upper

        elif self.settings.gausses == 5:
            lower_functions = self.generateGausses5(self.x_range, mean, SIGMA_OFFSET * (-1))
            sigma = lower_functions[0]
            lower_functions = lower_functions[1:]
            
            upper_functions = self.generateGausses5(self.x_range, mean, SIGMA_OFFSET)[1:]
            names = (self.settings.verylow, self.settings.low, self.settings.middle, self.settings.high, self.settings.veryhigh)
            for lower, upper, name in zip(lower_functions, upper_functions, names):
                self.features[idx][0][name] = lower
                self.features[idx][1][name] = upper

        elif self.settings.gausses == 7:
            lower_functions = self.generateGausses7(self.x_range, mean, SIGMA_OFFSET * (-1))
            sigma = lower_functions[0]
            lower_functions = lower_functions[1:]
            
            upper_functions = self.generateGausses7(self.x_range, mean, SIGMA_OFFSET)[1:]
            names = (self.settings.verylow, self.settings.low, self.settings.middlelow, self.settings.middle, self.settings.middlehigh, self.settings.high, self.settings.veryhigh)
            for lower, upper, name in zip(lower_functions, upper_functions, names):
                self.features[idx][0][name] = lower
                self.features[idx][1][name] = upper
            
        elif self.settings.gausses == 9:
            lower_functions = self.generateGausses9(self.x_range, mean, SIGMA_OFFSET * (-1))
            sigma = lower_functions[0]
            lower_functions = lower_functions[1:]
            upper_functions = self.generateGausses9(self.x_range, mean, SIGMA_OFFSET)[1:]
            names = (self.settings.verylow, self.settings.low, self.settings.middlelowminus, self.settings.middlelow, self.settings.middle
                     , self.settings.middlehigh, self.settings.middlehighplus, self.settings.high, self.settings.veryhigh)
            for lower, upper, name in zip(lower_functions, upper_functions, names):
                self.features[idx][0][name] = lower
                self.features[idx][1][name] = upper

        elif self.settings.gausses == 11:
            lower_functions = self.generateGausses11(self.x_range, mean, SIGMA_OFFSET * (-1))
            sigma = lower_functions[0]
            lower_functions = lower_functions[1:]
            upper_functions = self.generateGausses11(self.x_range, mean, SIGMA_OFFSET)[1:]
            names = (self.settings.verylow, self.settings.low, self.settings.middlelowminus, self.settings.middlelow, self.settings.middlelowplus, self.settings.middle,
                     self.settings.middlehighminus, self.settings.middlehigh, self.settings.middlehighplus, self.settings.high, self.settings.veryhigh)
            for lower, upper, name in zip(lower_functions, upper_functions, names):
                self.features[idx][0][name] = lower
                self.features[idx][1][name] = upper
            

        self.fuzzify_parameters.append(mean)
        self.fuzzify_parameters.append(sigma)
                              
        for x in values:
            verylow_value = low_value = middlelowminus_value = middlelow_value = middlelowplus_value = middle_value = middlehighminus_value = middlehigh_value = middlehighplus_value = high_value = veryhigh_value = 0

            # Ten zakomentowany kod przypisuje do powyższych zmiennych tuple dwuelementowe wartości dolnej i górnej funkcji przynależności zbioru rozmytego typu 2
            if self.settings.gausses == 3:
                #values = (verylow_value, middle_value, veryhigh_value)
                #lower_values = self.gaussianFunValues3(x, mean, sigma, SIGMA_OFFSET * (-1))
                #upper_values = self.gaussianFunValues3(x, mean, sigma, SIGMA_OFFSET)
                #for lower, upper, value in zip(lower_values, upper_values, values):
                #    value = (lower, upper)
                verylow_value, middle_value, veryhigh_value = self.gaussianFunValues3(x, mean, sigma)
                
            if self.settings.gausses == 5:
                #values = (verylow_value, low_value, middle_value, high_value, veryhigh_value)
                #lower_values = self.gaussianFunValues5(x, mean, sigma, SIGMA_OFFSET * (-1))
                #upper_values = self.gaussianFunValues5(x, mean, sigma, SIGMA_OFFSET)
                #for lower, upper, value in zip(lower_values, upper_values, values):
                #    value = (lower, upper)
                verylow_value, low_value, middle_value, high_value, veryhigh_value = self.gaussianFunValues5(x, mean, sigma)
    
            elif self.settings.gausses == 7:
                #values = (verylow_value, low_value, middlelow_value, middle_value, middlehigh_value, high_value, veryhigh_value)
                #lower_values = self.gaussianFunValues7(x, mean, sigma, SIGMA_OFFSET * (-1))
                #upper_values = self.gaussianFunValues7(x, mean, sigma, SIGMA_OFFSET)
                #for lower, upper, value in zip(lower_values, upper_values, values):
                #    value = (lower, upper)
                verylow_value, low_value, middlelow_value, middle_value, middlehigh_value, high_value, veryhigh_value = self.gaussianFunValues7(x, mean, sigma)

            elif self.settings.gausses == 9:
                #values = (verylow_value, low_value, middlelowminus_value, middlelow_value, middle_value, middlehigh_value, middlehighplus_value, high_value, veryhigh_value)
                #lower_values = self.gaussianFunValues9(x, mean, sigma, SIGMA_OFFSET * (-1))
                #upper_values = self.gaussianFunValues9(x, mean, sigma, SIGMA_OFFSET)
                #for lower, upper, value in zip(lower_values, upper_values, values):
                #    value = (lower, upper)
                verylow_value, low_value, middlelowminus_value, middlelow_value, middle_value, middlehigh_value, middlehighplus_value,
                high_value, veryhigh_value = self.gaussianFunValues9(x, mean, sigma)

            elif self.settings.gausses == 11:
                #values = (verylow_value, low_value, middlelowminus_value, middlelow_value, middlelowplus_value, middle_value, middlehighminus_value,
                #          middlehigh_value, middlehighplus_value, high_value, veryhigh_value)
                #lower_values = self.gaussianFunValues11(x, mean, sigma, SIGMA_OFFSET * (-1))
                #upper_values = self.gaussianFunValues11(x, mean, sigma, SIGMA_OFFSET)
                #for lower, upper, value in zip(lower_values, upper_values, values):
                #    value = (lower, upper)
                verylow_value, low_value, middlelowminus_value, middlelow_value, middlelowplus_value, middle_value,\
                middlehighminus_value, middlehigh_value, middlehighplus_value, high_value, veryhigh_value = self.gaussianFunValues11(x, mean, sigma)


            max_value = max([verylow_value, low_value, middlelowminus_value, middlelow_value, middlelowplus_value, middle_value, middlehighminus_value, middlehigh_value, middlehighplus_value, high_value, veryhigh_value])
            if max_value == verylow_value:
                return_value = self.features[idx][0][self.settings.verylow].label    
            elif max_value == low_value:
                 return_value = self.features[idx][0][self.settings.low].label                   
            elif max_value == middlelowminus_value:
                 return_value = self.features[idx][0][self.settings.middlelowminus].label  
            elif max_value == middlelow_value:
                 return_value = self.features[idx][0][self.settings.middlelow].label  
            elif max_value == middlelowplus_value:
                 return_value = self.features[idx][0][self.settings.middlelowplus].label                                                     
            elif max_value == middle_value:
                 return_value = self.features[idx][0][self.settings.middle].label      
            elif max_value == middlehighminus_value:
                 return_value = self.features[idx][0][self.settings.middlehighminus].label      
            elif max_value == middlehigh_value:
                 return_value = self.features[idx][0][self.settings.middlehigh].label      
            elif max_value == middlehighplus_value:
                 return_value = self.features[idx][0][self.settings.middlehighplus].label      
            elif max_value == high_value:
                 return_value = self.features[idx][0][self.settings.high].label                                                                                      
            else:
                 return_value = self.features[idx][0][self.settings.veryhigh].label                                                                                      

            return_array.append(return_value)
        return return_array
 
    def presentFuzzyFeature_Charts(self):
        for x in self.features:
            x.view()

    def fuzzify(self, features_table, mean_param):
        if isinstance(mean_param, (int, np.integer)):
            for idx, x in enumerate(self.features):
                features_table[x[0].label] = self.numbersToRowSets(idx, features_table[x[0].label], mean_param)
        
        else:
            for idx, x in enumerate(self.features):
                features_table[x[0].label] = self.numbersToRowSets(idx, features_table[x[0].label], mean_param[idx])
        
        if self.settings.show_results:
            self.presentFuzzyFeature_Charts()
            display(features_table)

        return features_table, self.feature_labels, self.features, self.decision, self.fuzzify_parameters
