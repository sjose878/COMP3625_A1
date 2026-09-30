from translator import UniversalTranslator
import matplotlib.pyplot as plt
import numpy as np
# create the UniversalTranslatsor object, with 2 knobs
translator = UniversalTranslator(n_dim=2)

#sample_settings = [np.random.rand(2).tolist() for num in range(25)]
sample_settings = [[np.random.uniform(0.05, 0.3), np.random.uniform(0.5, 0.7)] for num in range(25)]

# Evaluates how well a string is translated
# Note: The string has 964 words. Can calculate %
def evaluate_score(string: str) -> int:
    score = 0
    total = 0
    tokens = string.split()
    for word in tokens:
        if not word.isnumeric():
            score += 1
        total += 1
    return score / total

def derivative(translator, x, i):
    e = 0.02 #epsilon
    x_e = x.copy()
    x_e[i] += e
    x_e[i] = min(max(x_e[i], 0), 1) #bounds
    f_x = evaluate_score(translator.translate(x))
    f_xe = evaluate_score(translator.translate(x_e))
    return (f_xe - f_x) / e

a = 0.01

def gradient_descent(start, a, steps):
    x = start.copy()
    best_rate = evaluate_score(translator.translate(x))
    best_x = x.copy()
    for step in range(steps):    
        gradient = [derivative(translator, x, i) for i in range(len(x))]

        for i in range(len(x)):
            x[i] += a*gradient[i]
            x[i] = min(max(x[i], 0), 1) #keeps within bounds 0-1
        
        rate = evaluate_score(translator.translate(x)) #translate updated x
        if rate > best_rate:
            best_rate = rate
            best_x = x.copy()
        #print(f"Iteration {step+1}: x = {x}, Rate: {rate}")

    return best_x, best_rate

#random settings
#for index in sample_settings:
  #  translated_string = translator.translate(index)
  #  print(translated_string)
decode_rate = []
settings = []
best_rate_overall = 0
best_settings_overall = []

for index in sample_settings:
    final_settings, final_rate = gradient_descent(index, a, 100)
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
