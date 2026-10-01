from translator import UniversalTranslator
import numpy as np

# create the UniversalTranslator object, with 10 knobs
translator = UniversalTranslator(n_dim=10)

rng = np.random.default_rng()

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

        for j in range(len(settings)): # for each knob in settings
            gradient = derive(old_settings, j, nudge)
#            noise = np.random.normal(0, 0.1)
            settings[j] =  settings[j] + learn_rate * (gradient) #calculate step
            # stay within bounds
            settings[j] = max(0, min(settings[j], 1))

        diff = 0 # reset for each step
        for j in range(len(settings)):
            diff += abs(old_settings[j] - settings[j]) # stop if close to maxima
        if (diff < tolerance):
            print(f'Reached maxima,diff={diff}')
            break
        
        print(f'Score: {evaluate_score(translator.translate(settings))} Settings tried {translator.n_settings_tried()}')
    return settings

def hill_climb(settings, step_size, max_steps):
    best_score = evaluate_score(translator.translate(settings))

    for i in range(max_steps):
        for j in range(len(settings)):
            candidate = settings.copy()
            candidate[j] += np.random.uniform(-step_size, step_size)
            candidate[j] = max(0, min(candidate[j], 1))

            score = evaluate_score(translator.translate(candidate))

            if score >= best_score:
                settings = candidate
                best_score = score

        print(f"Score: {best_score:.3f}")

    return settings


# MAIN
# initial values 
num_rand_settings = 5
rand_settings = []
for i in np.arange(num_rand_settings):
    rand_settings.append([rng.random(),rng.random(),rng.random(),rng.random(),rng.random(),rng.random(),rng.random(),rng.random(),rng.random(),rng.random()])

start_values = np.array(rand_settings)
learn_rate = 0.003
nudge = 0.1
max_steps = 1000
tolerance = 1e-5 / len(rand_settings)
step_size = 0.2

# Keeps track of the best settings found
best_score = -1
for setting in start_values:
    result = hill_climb(setting, step_size, max_steps)
    final_string = translator.translate(result)
    final_score = evaluate_score(final_string)
    if final_score > best_score:
        best_result = result
        best_score = final_score
        print(f'Best updated: solution: {best_result} Score: {best_score:.3f}')

print(f'# Final solution: {best_result} Score: {best_score:.3f}')


# print total number of settings evaluated
print(f'# settings tried: {translator.n_settings_tried()}')