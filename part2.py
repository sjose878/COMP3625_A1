from translator import UniversalTranslator
import numpy as np


# create the UniversalTranslator object, with 10 knobs
translator = UniversalTranslator(n_dim=10)

rng = np.random.default_rng()
# data for plot
settings_tried = []
decode_rates = []

# Evaluates how well a string is translated
# @arg string is the translated string from translator
# Note: The string has 964 words.
def evaluate_score(string: str) -> int:
    score = 0
    tokens = string.split()
    for word in tokens:
        if not word.isnumeric():
            score += 1
    result = score/len(tokens)
    return result


# @arg start_values is a np.array that represents the knob settings first tried
# @arg index tells which knob is being derived with respect to
# @arg nudge changes how much x2 differs from x1 (np.array)
def derive(settings, index: int, nudge: float):
    string1 = translator.translate(settings)
    x1 = evaluate_score(string1)
    diff_settings = settings.copy()
    diff_settings[index] += nudge
    string2 = translator.translate(diff_settings)
    x2 = evaluate_score(string2)
    derivative = (x2 - x1)/nudge # definiton of a derivative as a limit
    return derivative

# @arg settings is the knob settings
# @arg nudge is how much x2 differs from x1 when calculating the slope in derive()
# @learn_rate is how fast the search function "explores"
# @max_steps is the max amount of steps the search function does before automatically stopping
# @tolerance tells how little change between steps the function will tolerate before it decides it has reached a maxima or plateau
def gradient_descent(settings, nudge: float, learn_rate: float, max_steps: int, tolerance: float):
    for i in range(max_steps):
        old_settings = settings.copy()  
        settings_tried.append(old_settings) # record for plot

        for j in range(len(settings)): # for each knob in settings
            
            settings[j] =  settings[j] + (learn_rate * derive(old_settings, j, nudge)) #calculate step
            # stay within bounds
            settings[j] = max(0, min(settings[j], 1))

        decode_rates.append(evaluate_score(translator.translate(settings))) # record for plot

        diff = 0 # reset for each step
        for j in range(len(settings)):
            diff += abs(old_settings[j] - settings[j]) # stop if close to maxima
        if (diff < tolerance):
            print(f'Reached maxima,diff={diff}')
            break
        
        print(f'Score: {evaluate_score(translator.translate(settings)):.3f}')
    return settings

# MAIN
# initial values 
num_rand_settings = 200
rand_settings = []
for i in np.arange(num_rand_settings):
    rand_settings.append([rng.random(),rng.random(),rng.random(),rng.random(),rng.random(),rng.random(),rng.random(),rng.random(),rng.random(),rng.random()])

start_values = np.array(rand_settings)
learn_rate = 0.003
nudge = 0.1
max_steps = 10000
tolerance = 1e-5 / len(rand_settings)

# Keeps track of the best settings found
best_score = -1
for setting in start_values:
    result = gradient_descent(setting, nudge, learn_rate, max_steps, tolerance)
    final_string = translator.translate(result)
    final_score = evaluate_score(final_string)
    if final_score > best_score:
        best_result = result
        best_score = final_score
        print(f'Best updated: solution: {best_result} Score: {best_score:.3f}')

print(f'# Final solution: {best_result} Score: {best_score:.3f}')


# print total number of settings evaluated
print(f'# settings tried: {translator.n_settings_tried()}')