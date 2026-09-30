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
def derive(settings, index: int, nudge: float):
    string1 = translator.translate(settings)
    x1 = evaluate_score(string1)
    diff_settings = settings.copy()
    diff_settings[index] += nudge
    string2 = translator.translate(diff_settings)
    x2 = evaluate_score(string2)
    derivative = (x2 - x1)/nudge # definiton of a derivative as a limit
    print("x1:", x1, "x2:", x2)

    return derivative

def gradient_descent(settings, nudge: float, learn_rate: float, max_steps: int, tolerance: float):
    for i in range(max_steps):
        old_settings = settings.copy()

        for j in range(len(settings)):
            settings[j] =  settings[j] + (learn_rate * derive(old_settings, j, nudge))
            # stay within bounds
            settings[j] = max(0, min(settings[j], 1))

        diff = 0
        for j in range(len(settings)):
            diff += abs(old_settings[j] - settings[j]) # stop if close to maxima
        print(f'grad: {derive(old_settings, j, nudge)}')
        if (diff < tolerance):
            print(f'Reached maxima,diff={diff}', end="")
            break
        print(f'Score: {evaluate_score(translator.translate(settings))}')
    return settings

#random settings
#for index in sample_settings:
    translated_string = translator.translate(index)
    print(translated_string)
    print("LINE BREAK AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

# MAIN
# initial values 
start_values = np.array([0.0, 0.9])
learn_rate = 0.05
nudge = 0.005
max_steps = 1000
tolerance = 1e-4
# data for plot
settings_tried = np.array([[]])
decode_rates = np.array([])

result = gradient_descent(start_values, nudge, learn_rate, max_steps, tolerance)
final_string = translator.translate(result)
final_score = evaluate_score(final_string)
print(f'# solution: {result} Score: {final_score}')
print(derive(start_values, 1, nudge))

# print total number of settings evaluated
print(f'# settings tried: {translator.n_settings_tried()}')

# given settings: a Nx2 array of N setting combinations
# and decode_rate: a length-N array of decode rates corresponding to those settings

# given settings: a Nx2 array of N setting combinations
# and decode_rate: a length-N array of decode rates corresponding to those settings

#settings = np.array(sample_settings)
# generate a scatter plot
#plt.scatter(x=settings_tried[:, 0],
#            y=settings_tried[:, 1],
#            c=decode_rates,
#            vmin=0, vmax=1
#           )
#
# add colorbar and gridlines
#cbar = plt.colorbar(label="decode rate")
plt.grid()

# add labels
plt.xlabel('knob 0 setting')
plt.ylabel('knob 1 setting')
plt.title('decode rates for settings tried')

# display
#plt.show()
