"""
File: part1.py
Group Members: Katrina Bourgeois, Sean Jose
Class: COMP 3625-001
Instructor: Eric Chalmers
Due Date: October 1st, 2026
"""

from translator import UniversalTranslator
import matplotlib.pyplot as plt
import numpy as np

# create the UniversalTranslatsor object, with 2 knobs
translator = UniversalTranslator(n_dim=2)

def evaluate_score(string: str) -> int:
    """
    Evaluates how well a string is translated by comparing the number of 
    successfully translated words to the total number of words in the string

    :param string: The string that has been translated
    :return score / total: The ratio of successfully translated words to all words in the message
    """

    score = 0 # the amount of successfully translated words
    total = 964 # overall total of the message
    tokens = string.split()

    # count all translated words
    for word in tokens:
        if not word.isnumeric():
            score += 1

    return score / total

def target_area(start_list):
    """
    Takes a list of tuples containing settings and their corresponding decode rates, 
    sorts the list, then finds the most promising areas and creates new settings based
    on those areas

    :param start_list: The list of tuples to be used. Tuples are in format of ([knob0, knob1], rate)
    :return new_settings: The list of knob settings targeted towards the promising areas
    """
    start_list = sorted(start_list, key=lambda i: i[1], reverse=True) #sort list by largest decode rate
    start_list = start_list[:3] #take only top 3 decode rates and their settings

    knob0_vals = []
    knob1_vals = []

    #find the low and high values for both knobs using the low and high from the top 3 settings
    for i in start_list:
        knob0_vals.append(i[0][0])
        knob1_vals.append(i[0][1])

    knob0_low = min(knob0_vals)
    knob0_high = max(knob0_vals)

    knob1_low = min(knob1_vals)
    knob1_high = max(knob1_vals)

    #create new list of targeted settings using the new knob values found
    new_settings = [[np.random.uniform(knob0_low, knob0_high), np.random.uniform(knob1_low, knob1_high)] for num in range(30)]
    
    return new_settings

def derivative(translator, setting, i):
    """
    Uses f'(x) = lim h->0 (f(x+h) - f(x))/h formula to determine derivative of the given
    knob setting. 

    :param translator: The given UniversalTranslatsor object
    :param setting: The given list of settings 
    :param i: The index to access the specific setting for knob 0 or knob 1
    :return (f_settingnudged - f_setting) / nudge: The computed derivative
    """

    nudge = 0.02 #h value
    setting_nudged = setting.copy()

    setting_nudged[i] += nudge
    setting_nudged[i] = min(max(setting_nudged[i], 0), 1) #keep nudged knob value within 0 and 1 bound

    f_setting = evaluate_score(translator.translate(setting)) #determine decode rate for original setting
    f_settingnudged = evaluate_score(translator.translate(setting_nudged)) #determine decode rate for nudged setting

    return (f_settingnudged - f_setting) / nudge

def gradient_descent(start_settings, learning_rate, steps):
    """
    Uses x = x - a*f'(x) gradient descent formula to determine best rate and settings 
    from the given start settings. a*f'(x) is added to x instead of subtracted to find
    maximum instead of minimum.

    :param start_settings: The given settings
    :param learning_rate: The given learning rate to be used 
    :param steps: The maximum number of steps that can be taken to change the settings
    :return best_settings: The knob 0 and knob 1 setting that produced the best decode rate
    :return best_rate: The corresponding decode rate for the best settings
    """

    current_settings = start_settings.copy()
    best_rate = evaluate_score(translator.translate(current_settings))
    best_settings = current_settings.copy()
    tolerance = 0.0005
    
    for step in range(steps): 
        old_settings = current_settings.copy() 
        deriv_settings = []

        #nudge each knob using derivative
        for i in range(len(current_settings)):
            deriv_settings.append(derivative(translator, current_settings, i))

            current_settings[i] += learning_rate*deriv_settings[i] #update settings using x -(-(a*f'(x)))
            current_settings[i] = min(max(current_settings[i], 0), 1) #keeps knob value within 0 and 1 bound

        #tolerance check, if change is minimal then assume close enough to maxima
        diff = 0.0 
        for j in range(len(current_settings)):
            diff += abs(old_settings[j] - current_settings[j]) #stop if close to maxima
        if (diff < tolerance):
            break

        #calculate rate for the new setting and determine the best rate so far
        rate = evaluate_score(translator.translate(current_settings)) #translate updated x
        if rate > best_rate:
            best_rate = rate
            best_settings = current_settings.copy()

    return best_settings, best_rate

#randomly generate initial list of settings
original_settings = [[np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)] for num in range(15)]
learning_rate = 0.08

decode_rate = []
settings = []
best_rate_overall = 0
best_settings_overall = []

#random exploration with gradient descent
target_tuple = ()
target_list = []

for i in original_settings:
    final_settings, final_rate = gradient_descent(i, learning_rate, 50)

    target_tuple = (final_settings, final_rate)
    target_list.append(target_tuple)

    decode_rate.append(final_rate)
    settings.append(final_settings)
   
    if final_rate > best_rate_overall:
        best_rate_overall = final_rate
        best_settings_overall = final_settings

#use values found to determine which areas to target for gradient descent
targeted_settings = target_area(target_list)

#targeted exploration with gradient descent 
for i in targeted_settings:
    final_settings, final_rate = gradient_descent(i, learning_rate, 100)
    decode_rate.append(final_rate)
    settings.append(final_settings)

    if final_rate > best_rate_overall:
        best_rate_overall = final_rate
        best_settings_overall = final_settings

print(f'Best settings: {best_settings_overall} Decode rate achieved: {best_rate_overall}')
print(f'# settings tried: {translator.n_settings_tried()}\n')
print(f'Translated message using best settings: ')
translated_string = translator.translate(best_settings_overall)
print(translated_string)

#graph plotting code from sample code provided in Assignment 1 google doc available on the COMP 3625-001 D2L
#written by Eric Chalmer
#retrieved September 30, 2026
settings = np.array(settings)

# generate a scatter plot
plt.scatter(x=settings[:, 0],
            y=settings[:, 1],
            c=decode_rate,
            vmin=0, vmax=1
            )

# add colorbar and gridlines
cbar = plt.colorbar(label="decode rate")
plt.grid()

# add labels
plt.xlabel('knob 0 setting')
plt.ylabel('knob 1 setting')
plt.title('decode rates for settings tried')

# display
plt.show()
