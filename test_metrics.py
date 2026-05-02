from utils.metrics import accuracy, confusion_matrix

y_true = [0, 1, 2, 1, 0]
y_pred = [0, 1, 2, 0, 0]

print("Accuracy:", accuracy(y_true, y_pred))
cm, labels = confusion_matrix(y_true, y_pred)
print("Labels:", labels)
print(cm)