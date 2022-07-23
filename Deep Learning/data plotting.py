import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# plotdata = pd.DataFrame({
#
#     "Train": [97, 98.5],
#
#     "Test": [92, 92.8]},
#
#     index=["SVM", "RFC"])

# plotdata = pd.DataFrame({
#
#     "Train": [99.8, 99.8],
#
#     "Test": [98.5, 97]},
#
#     index=["CNN", "AlexNet"])
#
# plotdata.plot(kind="bar", figsize=(15, 8), color=('#B3CBD3', '#EED4F2'))

# trainNewMachine = [[99.8, 99.9],
#                    [97.7, 98.5]]
# trainOldDeep = [[99.8, 99.8],
#                 [98.5, 97]]
#
test = [92, 92.8, 86]
classifierMachine = ["SVM", "RFC", "Logistic Regression"]
# # testOldMachine = [91.6, 93.6]
# # testNewMachine = [97.7, 98.5]
#
# classifierDeep = ["CNN", "AlexNet"]
# testOldDeep = [98.5, 97]
# # testNewDeep = [82, 83]
#
# X = np.arange(2)
# fig = plt.figure()
# ax = fig.add_axes([0, 0, 1, 1])
# ax.bar(X + 0.00, trainNewMachine[0], color='#E3E3E3', width=0.25)
# ax.bar(X + 0.25, trainNewMachine[1], color='#B3CBD3', width=0.25)
# # ax.bar(X + 0.50, data[2], color='r', width=0.25)
plt.ylim(65, 95)
plt.bar(classifierMachine, test, width=0.2)

plt.xlabel('Classifiers')
plt.ylabel('Accuracy')

plt.title('Best Models')

plt.show()
