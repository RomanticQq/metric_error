import torch
from mymetric import MM
from minmax import MinMax
from min import Min
accuracy1 = MM()
accuracy2 = MinMax()
accuracy3 = Min()

batches = 10
for i in range(batches):
    pred = torch.randn(10, 5).softmax(dim=-1)
    target = torch.randint(5, (10,))
    print("aaaa")
    # acc1 = accuracy1(pred, target)
    # accuracy2.update(pred, target)
    # acc2 = accuracy2.compute()
    acc3 = accuracy3(pred)
    # print(f"Accuracy on batch {i} : {acc1}")
    # print(f"Accuracy on batch {i} : {acc2}")
    print(f"Accuracy on batch {i} : {acc3}")

# total_accuracy1 = accuracy1.compute()
# total_accuracy2 = accuracy2.compute()
total_accuracy3 = accuracy3.compute()
# print(f"Accuracy on all data: {total_accuracy1}")
# print(f"Accuracy on all data: {total_accuracy2}")
print(f"Accuracy on all data: {total_accuracy3}")