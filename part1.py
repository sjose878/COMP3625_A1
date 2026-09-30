from translator import UniversalTranslator
import matplotlib.pyplot as plt
import numpy as np
# create the UniversalTranslator object, with 2 knobs
translator = UniversalTranslator(n_dim=2)

# demo of how to use the UniversalTranslator object. You can delete these lines
#sample_settings = [[0, 0.3], [0, 0.6], [0, 0.9]]

# Evaluates how well a string is translated
# Note: The string has 964 words. Can calculate %
def evaluate_score(string: str) -> int:
    score = 0
    tokens = string.split()
    for word in tokens:
        if not word.isnumeric():
            score += 1
    return score

# 
# @arg start_values is a np.array that represents the knob settings first tried
# @arg nudge changes how much x2 differs from x1 (np.array)
def derive(settings, index, nudge):
    string1 = translator.translate(settings)
    x1 = evaluate_score(string1)
    diff_settings = settings
    diff_settings[index] += nudge
    string2 = translator.translate(diff_settings)
    x2 = evaluate_score(string2)
    derivative = (x2 - x1)/nudge # definiton of a derivative as a limit
    return derivative

def gradient_descent(settings, nudge, learn_rate: float, max_steps: int):
    for i in range(max_steps):
        index = 0 # track which knob is currently being derived
        for knob in settings:
            new_setting =  knob + (learn_rate * derive(settings, index, nudge))
            knob = new_setting
            index += 1
            if (new_setting - knob) < 0.0005: # stop if close to maxima
                print("Max steps exceeded")
                break
        print(f'# Score: {evaluate_score(translator.translate(settings))}')
    return settings


#random settings
#for index in sample_settings:
    translated_string = translator.translate(index)
    print(translated_string)
    print("LINE BREAK AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

# MAIN
# initial values 
start_values = np.array([0.0, 0.0])
learn_rate = 0.1
nudge = 0.001
max_steps = 1000
# data for plot
settings_tried = np.array([[]])
decode_rates = np.array([])

result = gradient_descent(start_values, nudge, learn_rate, max_steps)
final_string = translator.translate(result)
final_score = evaluate_score(final_string)
print(f'# solution: {result} Score: {final_score}')

# print total number of settings evaluated
print(f'# settings tried: {translator.n_settings_tried()}')

# given settings: a Nx2 array of N setting combinations
# and decode_rate: a length-N array of decode rates corresponding to those settings

# given settings: a Nx2 array of N setting combinations
# and decode_rate: a length-N array of decode rates corresponding to those settings

#settings = np.array(sample_settings)
# generate a scatter plot
plt.scatter(x=settings_tried[:, 0],
            y=settings_tried[:, 1],
            c=decode_rates,
            vmin=0, vmax=1
           )
#
# add colorbar and gridlines
cbar = plt.colorbar(label="decode rate")
plt.grid()

# add labels
plt.xlabel('knob 0 setting')
plt.ylabel('knob 1 setting')
plt.title('decode rates for settings tried')

# display
plt.show()
