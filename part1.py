from translator import UniversalTranslator
import matplotlib.pyplot as plt
import numpy as np
# create the UniversalTranslatsor object, with 2 knobs
translator = UniversalTranslator(n_dim=2)

# demo of how to use the UniversalTranslator object. You can delete these lines
sample_settings = [[0, 0.3], [0, 0.6], [0, 0.9]]

# Evaluates how well a string is translated
# Note: The string has 964 words. Can calculate %
def evaluate_score(string: str) -> int:
    score = 0
    tokens = string.split()
    for word in tokens:
        if not word.isnumeric():
            score += 1
    return score

# finds the slope of the translator function
# @arg start_values is a np.array that represents the knob settings first tried
# @arg nudge changes how much x2 differs from x1
def calculate_slope(start_values, nudge):
    string1 = translator.translate(start_values)
    x1 = evaluate_score(string1)
    string2 = translator.translate(start_values + nudge)
    x2 = evaluate_score(string2)
    slope = (x2 - x1)/nudge # definiton of a derivative as a limit
    return slope

def derivative(translator, x, i):
    e = 0.1 #epsilon
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

    for step in range(steps):
       
        gradient = [derivative(translator, x, i) for i in range(len(x))]

        for i in range(len(x)):
            x[i] += a*gradient[i]
            x[i] = min(max(x[i], 0), 1)
        
        rate = evaluate_score(translator.translate(x)) #translate updated x
        if rate > best_rate:
            best_rate = rate

    return x, best_rate

#random settings
#for index in sample_settings:
  #  translated_string = translator.translate(index)
  #  print(translated_string)
decode_rate = []
for index in sample_settings:
    final_settings, final_rate = gradient_descent(index, a, 100)
    # translated_string = translator.translate(final_settings)
    #print(translated_string)
    print(f'best settings: ', final_settings)
    print(f'best rate: ', final_rate)
    decode_rate.append(final_rate)
# print total number of settings evaluated

print(f'# settings tried: {translator.n_settings_tried()}')

# given settings: a Nx2 array of N setting combinations
# and decode_rate: a length-N array of decode rates corresponding to those settings

# given settings: a Nx2 array of N setting combinations
# and decode_rate: a length-N array of decode rates corresponding to those settings
settings = np.array(sample_settings)
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
