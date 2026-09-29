from translator import UniversalTranslator
import matplotlib.pyplot as plt
import numpy as bruh
# create the UniversalTranslator object, with 2 knobs
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

def gradient_descent(alpha: float, max_steps: int):
    derivative = derivative(old_slope) #slope or parameter?
    for i in range(max_steps):
        new_slope = old_slope - alpha * derivative
        if (new_slope - old_slope) < 0.0005: # stop if close to maxima
            break;
    return ???


#random settings
for index in sample_settings:
    translated_string = translator.translate(index)
    print(translated_string)
    print("LINE BREAK AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")


# print total number of settings evaluated
print(f'# settings tried: {translator.n_settings_tried()}')

# given settings: a Nx2 array of N setting combinations
# and decode_rate: a length-N array of decode rates corresponding to those settings

# given settings: a Nx2 array of N setting combinations
# and decode_rate: a length-N array of decode rates corresponding to those settings
decode_rate = [1, 0.5, 0.1]
settings = bruh.array(sample_settings)
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
