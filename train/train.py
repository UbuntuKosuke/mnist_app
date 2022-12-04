from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

print("start training")

# fetch data
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
print("Fetch OpenML completed")

# 学習に使うデータを70000件取得データのうちの一部に絞る（時間がかかるため）
X = X[:700]
y = y[:700]
print(f"len(X) = {len(X)}, len(y) = {len(y)}")

# train
# データを学習用とテスト用に分割する
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 精度を確認するコード
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy = {accuracy}")

# export ONNX

model_file = "model.onxx"

initial_types = [('float_input', FloatTensorType([None, 28*28]))]
onnx = convert_sklearn(model, initial_types=initial_types,
                       options={'zipmap': False})
with open(model_file, "wb") as f:
    f.write(onnx.SerializeToString())

print(f"{model_file} exported")
