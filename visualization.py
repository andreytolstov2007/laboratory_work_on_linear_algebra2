import matplotlib.pyplot as plt

def Plot_decision_boundary(model, X, y, title="Decision Boundary"):
    x_coords = []
    y_coords = []
    for i in range(len(X)):
        x_coords.append(X[i][0])
        y_coords.append(X[i][1])
   
    x_min = min(x_coords) - 1
    x_max = max(x_coords) + 1
    y_min = min(y_coords) - 1
    y_max = max(y_coords) + 1
    step = 0.05
    x_vals = []
    current_x = x_min
    while current_x <= x_max:
        x_vals.append(current_x)
        current_x = current_x + step
   
    y_vals = []
    current_y = y_min
    while current_y <= y_max:
        y_vals.append(current_y)
        current_y = current_y + step
   
    Z = []
    for j in range(len(y_vals)):
        row = []
        for i in range(len(x_vals)):
            pred = model.Predict_single([x_vals[i], y_vals[j]])
            row.append(pred)
        Z.append(row)
   
    plt.figure(figsize=(8, 6))
    plt.contourf(x_vals, y_vals, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    colors = []
    for label in y:
        if label == 0:
            colors.append('blue')
        else:
            colors.append('red')
   
    plt.scatter(x_coords, y_coords, c=colors, edgecolors='black', alpha=0.7)
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.title(title)
    plt.show()

def Plot_loss_curves(losses_dict, title="Loss Curves"):
    plt.figure(figsize=(10, 6))
    for label in losses_dict:
        losses = losses_dict[label]
        plt.plot(losses, label=label)
   
    plt.xlabel("Эпоха")
    plt.ylabel("Бинарная кросс-энтропия")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def Calculate_accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct = correct + 1
    accuracy = correct / len(y_true)
    return accuracy