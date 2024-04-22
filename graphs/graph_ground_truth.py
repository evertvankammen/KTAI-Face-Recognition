import matplotlib.pyplot as plt

# Gegevens voor de grafiek
actors = ['Chandler', 'Joey', 'Monica', 'Phoebe', 'Rachel', 'Ross', 'Unknown']
frequency_of_recognized_actors = [331, 292, 639, 131, 289, 456, 689]

frames_from_sample = [331, 292, 639, 131, 289, 456, 689]
percentage_of_sample = [14.43523768, 12.73440907, 27.86742259, 5.713039686, 12.6035761, 19.88661143, 30.04797209]

# Maak de grafiek
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Actors')
ax1.set_ylabel('Frequency of Recognized Actors', color=color)
ax1.bar(actors, frequency_of_recognized_actors, color=color)
ax1.tick_params(axis='y', labelcolor=color)
# Stel de limieten van de y-as expliciet in voor de eerste grafiek
ax1.set_ylim([0, 1000])  # Pas dit aan naar de gewenste bovengrens

# ax2 = ax1.twinx()
# color = 'tab:red'
# ax2.set_ylabel('Percentage of Sample', color=color)
# ax2.plot(actors, percentage_of_sample, color=color, marker='o')
# ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Comparison of Frequency of Recognized Actors with Percentage of Sample')
plt.xticks(rotation=45)
plt.show()
