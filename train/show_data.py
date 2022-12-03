from sklearn.datasets import fetch_openml

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
print("Fetch OpenML Completed")

print(f"len(X) = {len(X)}, len(y) = {len(y)}")
print("X[0]")
print(X[0])

print("y[0]")
print(y[0])

# 一つの画像データの内容を掘り下げてみてみる
X0 = X[0]
# 28 * 28 = 784と表示される
print(f"len(X0) ={len(X0)}")

print("X0")
for i, v in enumerate(X0):
    # 数字を三桁で表示改行しないようにend=""としている
    print("%3d" % v, end="")
    # 28のタイミングで改行を入れる
    if i % 28 == 27:
        print();
