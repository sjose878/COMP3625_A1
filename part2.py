from translator import UniversalTranslator
import numpy as np
import math

# bounds of knob settings
LOWER_LIMIT = 0
UPPER_LIMIT = 1.0

# create the UniversalTranslator object, with 10 knobs
translator = UniversalTranslator(n_dim=10)

rng = np.random.default_rng()

# Evaluates how well a string is translated
# Note: The string has 964 words.
# @arg string is the translated string from translator
# @return result is the percentage of words from the string that were translated. Range of [0-1]
def evaluate_score(string: str) -> float:
    score = 0
    tokens = string.split()
    for word in tokens:
        if not word.isnumeric():
            score += 1
    result = score/len(tokens)
    return result

# Derives a gradient using the definition of a derivative as a limit formula
# f'(x) = lim h->0 (f(x + h) - f(x))/h
# @arg start_values is a np.array that represents the knob settings first tried
# @arg index tells which knob is being derived with respect to
# @arg nudge changes how much x2 differs from x1 (np.array)
# @return derivative is the calculated gradient
def derive(settings, index: int, nudge: float):
    string1 = translator.translate(settings)
    x1 = evaluate_score(string1)
    diff_settings = settings.copy()
    diff_settings[index] += nudge
    string2 = translator.translate(diff_settings)
    x2 = evaluate_score(string2)
    derivative = (x2 - x1)/nudge # definiton of a derivative as a limit
    return derivative

# Maximizing search function using the gradient descent method
# @arg settings is the initial knob settings
# @arg nudge is how much x2 differs from x1 when calculating the slope in derive()
# @learn_rate is how fast the search function "explores"
# @max_steps is the max amount of steps the search function does before automatically stopping
# @tolerance tells how little change between steps the function will tolerate before it decides it has reached a maxima or plateau
# @return settings is the optimized settings configuration found
def gradient_descent(settings, nudge: float, learn_rate: float, max_steps: int, tolerance: float):
    for i in range(max_steps):
        old_settings = settings.copy()  
        for j in range(len(settings)): # for each knob in settings
            gradient = derive(old_settings, j, nudge)
            # calculate step
            settings[j] =  settings[j] + learn_rate * (gradient)
            # stay within bounds
            settings[j] = max(LOWER_LIMIT, min(settings[j], UPPER_LIMIT))

        diff = 0 # reset for each step
        for j in range(len(settings)):
            diff += abs(old_settings[j] - settings[j])
        if (diff < tolerance): # stop if close to maxima
            print(f'Reached maxima, diff={diff}')
            break
        
        print(f'GD Score: {evaluate_score(translator.translate(settings))}')
    return settings

# Maximizing search function using a stochastic hill climbing method
# @arg settings is the initial knob settings
# @arg step_size is the distance from the current state to a neighboring state
# @arg max_steps is the max amount of steps an iteration of new settings will attempt
# @arg num_knobs_changed is how many random knobs in settings will be tweaked per iteration
# @return settings is the optimized settings configuration found
def hill_climb(settings, step_size: float, max_steps: int, num_knobs_changed: int):
    current_score = evaluate_score(translator.translate(settings)) # initial state

    # Once a "a good area" is found, start exploiting it aggressively
    for i in range(max_steps):
        if current_score > 0.5:
            step_size = 0.01
        elif current_score > 0:
            step_size = 0.1

        new_settings = settings.copy()
        # Randomly choose only some of the knobs to change
        for j in np.random.choice(len(settings), num_knobs_changed, replace=False):
            # size of step taken is a random value between -step_size to step_size
            new_settings[j] += np.random.uniform(-step_size, step_size)
            # stay in bounds
            new_settings[j] = max(LOWER_LIMIT, min(new_settings[j], UPPER_LIMIT))

        new_score = evaluate_score(translator.translate(new_settings))

        if new_score >= current_score:
            settings = new_settings
            current_score = new_score            
        #print(f"HC Score: {current_score}")
    return settings

# Maximizing search function using the simulated anealing method
# @arg settings is the initial knob settings
# @arg temp sets the temperature for the simulated anealing (How exploratory an iteration starts)
# @arg cooling_rate is how fast the temperature cools down. (How fast it switches from exploratory to exploitive)
# @arg step_size is the distance from the current state to a neighboring state
# @arg max_steps is the max amount of steps an iteration of new settings will attempt
# @arg num_knobs_changed is how many random knobs in settings will be tweaked per iteration
# @return settings is the optimized settings configuration found
def simulated_anealing(settings, temp: float, cooling_rate: float, step_size: float, max_steps: int, num_knobs_changed: int):
    current_score = evaluate_score(translator.translate(settings)) # initial state
    accepted = True
    for i in range(max_steps):
        # Adaptive step size. Increase when in a "good area"
        if accepted and current_score != 0: # shrink step_size in exploitable region
            step_size *= 0.95
        else:
            step_size *= 1.05 # grow step_size to explore more region
        # clip step_size changes within bounds
        step_size = np.clip(step_size, 0.01, 0.3)

        # get neighbor
        next = settings.copy()
        # Randomly choose only some of the knobs to change
        for j in np.random.choice(len(settings), num_knobs_changed, replace=False):
            # size of step taken is a random value between -step_size to step_size                
            next[j] += np.random.uniform(-step_size, step_size)
            # stay in bounds
            next[j] = max(LOWER_LIMIT, min(next[j], UPPER_LIMIT))

        new_score = evaluate_score(translator.translate(next))
        delta = new_score - current_score

            # accept if neighbor moves uphill or
            # might be rejected randomly based off temperature
        if (delta >= 0) or rng.random() < math.exp(delta/temp):
            settings = next
            current_score = new_score

            # sets flag to shrink step_size in exploitable region, otherwise set flag to grow step_size
            accepted = True
        else:
            accepted = False
        print(f"Sim_Aneal Score: {current_score}")

        temp *= cooling_rate
        if temp < 1e-8: # stop once temperature has fully cooled down
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
gd_max_steps = 500
tolerance = 1e-5 / len(rand_settings)

# HILL CLIMBING variables
hc_max_steps = 2000
hc_step_size = 0.2
hc_num_knobs_changed = 3

# SIMULATED ANEALING variables
temp = 2 # higher temp -> more exploration
cooling_rate = 0.995 # how fast exploration slows down
sim_step_size = 0.2
sim_max_steps = 1000
sim_num_knobs_changed = 4

best_score = -1
for setting in start_values:
# ONLY USE 1 AT A TIME
# Gradient Descent
    #result = gradient_descent(setting, nudge, learn_rate, gd_max_steps, tolerance)
# Hill Climbing
    result = hill_climb(setting, hc_step_size, hc_max_steps, hc_num_knobs_changed)
# Simulated Anealing
    #result = simulated_anealing(setting, temp, cooling_rate, sim_step_size, sim_max_steps, sim_num_knobs_changed)
    
    # Keeps track of the current best settings found
    final_string = translator.translate(result)
    final_score = evaluate_score(final_string)
    if final_score > best_score:
        best_result = result
        best_score = final_score
        print(f'Current best solution: {best_result}')
        print(f'Current score: {best_score:.3f}')

print(f'# Final solution: {best_result}\n# Decode rate: {best_score:.3f}')
# print total number of settings evaluated
print(f'# settings tried: {translator.n_settings_tried()}')