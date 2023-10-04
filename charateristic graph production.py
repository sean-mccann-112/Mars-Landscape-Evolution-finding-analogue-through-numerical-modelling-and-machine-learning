import matplotlib.pyplot as plt

train_cd = [3]*5 + [6]*5 + [9]*5 + [12]*5 + [15]*5
train_ntg = [10, 20, 30, 40, 50]*5
train_aggredation = [
    29.03, 13.71, 8.58, 5.59, 4.41,
    58.09, 43.41, 27.16, 18.97, 14.98,
    171.10, 80.80, 50.54, 35.29, 26.01,
    259.30, 122.45, 76.61, 53.48, 39.42,
    353.02, 166.70, 104.28, 72.82, 53.67,
]


test_cd = [5, 10, 15, 8, 9]
test_ntg = [15, 25, 35, 45, 30]


plt.figure(figsize=(5, 5.5))
plt.scatter(3, 10, marker="$A$", c="black", label="Aggredation Rate(m / 10000 itt)")
plt.scatter(train_cd, train_ntg, s=50, marker="o", c="b", label="Training Datapoints [seed = 1]")
for (i, j, k) in zip(train_cd, train_ntg, train_aggredation):
    plt.text(i-0.5, j-2.2, f'{k}')
plt.scatter(test_cd, test_ntg, s=100, marker="x", c="r", label="Testing Datapoints [seed = 5]")
plt.xlim(1.5, 16.5)
plt.ylim(5, 60)

plt.title("Characteristics Phase Plane")
plt.xlabel("Channel Depth (m)")
plt.ylabel("Net to Gross (%)")

plt.legend()
plt.show()





