from translator import UniversalTranslator
import numpy as np
import math

LOWER_LIMIT = 0
UPPER_LIMIT = 1.0

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
            settings[j] = max(LOWER_LIMIT, min(settings[j], UPPER_LIMIT))

        diff = 0 # reset for each step
        for j in range(len(settings)):
            diff += abs(old_settings[j] - settings[j]) # stop if close to maxima
        if (diff < tolerance):
            print(f'Reached maxima,diff={diff}')
            break
        
        print(f'Score: {evaluate_score(translator.translate(settings))} Settings tried {translator.n_settings_tried()}')
    return settings

def hill_climb(settings, step_size: float, max_steps: int, knobs_changed: int):
    current_score = evaluate_score(translator.translate(settings))
    for i in range(max_steps):
        if current_score > 0:
            step_size = 0.1
        if current_score > 0.5:
            step_size = 0.005
        if current_score > 0.6:
            step_size = 0.0001

        # Randomly choose only some of the knobs to change
        for j in np.random.choice(len(settings), knobs_changed, replace=False):
            new_settings = settings.copy()
            new_settings[j] += np.random.uniform(-step_size, step_size)
            new_settings[j] = max(LOWER_LIMIT, min(new_settings[j], UPPER_LIMIT)) # stay in bounds

            new_score = evaluate_score(translator.translate(new_settings))

            if new_score >= current_score:
                settings = new_settings
                current_score = new_score
            
#        print(f"HC Score: {current_score}")
    return settings

def simulated_anealing(settings, temp: float, cooling_rate: float, step_size: float, max_steps: int):
    current_score = evaluate_score(translator.translate(settings))
    #accepted = True
    for i in range(max_steps):
#        for j in range(len(settings)):

        # Randomly choose only some of the knobs to change
        for j in np.random.choice(len(settings), size=2, replace=False):
            # get neighbor
            next = settings.copy()                
            next[j] += np.random.uniform(-step_size, step_size)
            next[j] = max(LOWER_LIMIT, min(next[j], UPPER_LIMIT)) # stay in bounds

            new_score = evaluate_score(translator.translate(next))
            delta = new_score - current_score

            # accept if neighbor moves uphill
                # might be rejected randomly based off temperature
            if (delta >= 0) or rng.random() < math.exp(delta/temp):
                settings = next
                current_score = new_score
             #   accepted = True
            #else:
            #    accepted = False
            print(f"Sim_Aneal Score: {current_score}")

            temp *= cooling_rate
                # stop once temperature has fully cooled down
            if temp < 1e-8:
                break
    return settings

# MAIN
# initial values 
num_rand_settings = 30
rand_settings = []
for i in np.arange(num_rand_settings):
    rand_settings.append([rng.random(),rng.random(),rng.random(),rng.random(),rng.random(),rng.random(),rng.random(),rng.random(),rng.random(),rng.random()])
start_values = np.array(rand_settings)

# GRADIENT DESCENT variables
nudge = 0.1
learn_rate = 0.003
gd_max_steps = 1000
tolerance = 1e-5 / len(rand_settings)

# HILL CLIMBING variables
hc_max_steps = 1000
hc_step_size = 0.2
hc_num_knobs_changed = 4

# SIMULATED ANEALING variables
temp = 1.5 # higher temp -> more exploration
cooling_rate = 0.99 # how fast exploration slows down
sim_step_size = 0.2
sim_max_steps = 500

# Keeps track of the best settings found
best_score = -1
for setting in start_values:

# Gradient Descent
    #result = gradient_descent(setting, nudge, learn_rate, gd_max_steps, tolerance)
# Hill Climbing
    result = hill_climb(setting, hc_step_size, hc_max_steps, hc_num_knobs_changed)
# Simulated Anealing
    #result = simulated_anealing(setting, temp, cooling_rate, sim_step_size, sim_max_steps)
 
    final_string = translator.translate(result)
    final_score = evaluate_score(final_string)
    if final_score > best_score:
        best_result = result
        best_score = final_score
        print(f'Current best solution: {best_result}\nCurrent score: {best_score:.3f}\n')

print(f'# Final solution: {best_result}\n# Decode rate: {best_score:.3f}')
# print total number of settings evaluated
print(f'# settings tried: {translator.n_settings_tried()}')