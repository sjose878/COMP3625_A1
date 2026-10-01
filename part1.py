from translator import UniversalTranslator
import matplotlib.pyplot as plt
import numpy as np

# create the UniversalTranslatsor object, with 2 knobs
translator = UniversalTranslator(n_dim=2)

#implement random -> targeted strategy
#sample_settings = [np.random.rand(2).tolist() for num in range(25)]
#original_settings = [[np.random.uniform(0.05, 0.3), np.random.uniform(0.5, 0.7)] for num in range(25)]

# Evaluates how well a string is translated
# Note: The string has 964 words. Can calculate %
def evaluate_score(string: str) -> int:
    score = 0
    total = 964
    tokens = string.split()

    for word in tokens:
        if not word.isnumeric():
            score += 1

    return score / total

# uses f'(x) = lim h->0 (f(x+h) - f(x))/h formula to determine derivative 
def derivative(translator, setting, i):
    nudge = 0.02 #h value
    setting_nudged = setting.copy()

    setting_nudged[i] += nudge
    setting_nudged[i] = min(max(setting_nudged[i], 0), 1) #bounds

    f_setting = evaluate_score(translator.translate(setting))
    f_settingnudged = evaluate_score(translator.translate(setting_nudged))

    return (f_settingnudged - f_setting) / nudge

# def target_area(start_settings):
#     targeted_settings = []
#     for i in start_settings:
#         if start_settings[i] > targeted_settings[i]
   
#     targeted_settings = [[np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)] for num in range(25)]
    
#     return targeted_settings

def gradient_descent(start_settings, learning_rate, steps):
    current_settings = start_settings.copy()
    best_rate = evaluate_score(translator.translate(current_settings))
    best_x = current_settings.copy()
    tolerance = 0.0005
    
    for step in range(steps): 
        old_settings = current_settings.copy() 
        #deriv_settings = [derivative(translator, current_settings, i) for i in range(len(current_settings))]
        deriv_settings = []
        for i in range(len(current_settings)):
            deriv_settings.append(derivative(translator, current_settings, i))

            current_settings[i] += learning_rate*deriv_settings[i] #update settings
            current_settings[i] = min(max(current_settings[i], 0), 1) #keeps within bounds 0-1

        #tolerance check
        diff = 0.0 # reset for each step
        for j in range(len(current_settings)):
            diff += abs(old_settings[j] - current_settings[j]) # stop if close to maxima
        if (diff < tolerance):
            #print(f'Reached maxima, diff={diff}')
            break

        rate = evaluate_score(translator.translate(current_settings)) #translate updated x
        
        if rate > best_rate:
            best_rate = rate
            best_x = current_settings.copy()
        #print(f"Iteration {step+1}: x = {x}, Rate: {rate}")

    return best_x, best_rate

original_settings = [[np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)] for num in range(15)]

learning_rate = 0.08
decode_rate = []
settings = []
best_rate_overall = 0
best_settings_overall = []
target_tuple = ()
target_list = []

#explore
for i in original_settings:
    final_settings, final_rate = gradient_descent(i, learning_rate, 50)

    target_tuple = (final_settings, final_rate)
    target_list.append(target_tuple)

    decode_rate.append(final_rate)
    settings.append(final_settings)
   
    if final_rate > best_rate_overall:
        best_rate_overall = final_rate
        best_settings_overall = final_settings

target_list = sorted(target_list, key=lambda i: i[1], reverse=True)
target_list = target_list[:3]
print(target_list)

knob0_vals = []
knob1_vals = []

for i in target_list:
    knob0_vals.append(i[0][0])
    knob1_vals.append(i[0][1])

knob0_low = min(knob0_vals)
knob0_high = max(knob0_vals)

knob1_low = min(knob1_vals)
knob1_high = max(knob1_vals)

targeted_settings = [[np.random.uniform(knob0_low, knob0_high), np.random.uniform(knob1_low, knob1_high)] for num in range(30)]

#target
for i in targeted_settings:
    final_settings, final_rate = gradient_descent(i, learning_rate, 100)
    #translated_string = translator.translate(final_settings)
    #print(translated_string)
    #print(f'best settings: ', final_settings)
    #print(f'best rate: ', final_rate)
    decode_rate.append(final_rate)
    settings.append(final_settings)

    if final_rate > best_rate_overall:
        best_rate_overall = final_rate
        best_settings_overall = final_settings


# print total number of settings evaluated
print(f'# settings tried: {translator.n_settings_tried()}')
print(f'Best settings: {best_settings_overall} Best rate: {best_rate_overall}')
translated_string = translator.translate(best_settings_overall)
print(translated_string)

# given settings: a Nx2 array of N setting combinations
# and decode_rate: a length-N array of decode rates corresponding to those settings
# print(settings)
# print(decode_rate)
settings = np.array(settings)

# code from assignment 1 document (cite)
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
