import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Tải dữ liệu từ file CSV
data = pd.read_csv('nvidia_stock_prices .csv') 

# Kiểm tra vài dòng đầu của dữ liệu
print(data.head())

# Chọn các cột dữ liệu
X = data[['Open', 'High', 'Low', 'Volume']]  
y = data['Close'] 

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# In kích thước tập huấn luyện
print("Kích thước tập huấn luyện:", X_train.shape)
print("Kích thước tập kiểm tra:", X_test.shape)

# Điều chỉnh giá trị thiếu
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Khởi tạo và huấn luyện mô hình
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_linear = lin_reg.predict(X_test)

# Đánh giá mô hình Linear Regression
mae_linear = mean_absolute_error(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print(f"Linear Regression - MAE: {mae_linear}, MSE: {mse_linear}, R-squared: {r2_linear}")

# So sánh với mô hình LSTM
# Định dạng lại dữ liệu cho LSTM (dữ liệu 3D)
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Xây dựng mô hình LSTM
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=False, input_shape=(X_train_lstm.shape[1], 1)))
lstm_model.add(Dense(1))

# Biên dịch mô hình
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Huấn luyện mô hình LSTM
lstm_model.fit(X_train_lstm, y_train.values.reshape(-1, 1), epochs=10, batch_size=32, verbose=1)

# Dự đoán trên tập kiểm tra
y_pred_lstm = lstm_model.predict(X_test_lstm)

# Đánh giá mô hình LSTM
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
mse_lstm = mean_squared_error(y_test, y_pred_lstm)
r2_lstm = r2_score(y_test, y_pred_lstm)

print(f"LSTM - MAE: {mae_lstm}, MSE: {mse_lstm}, R-squared: {r2_lstm}")

# Vẽ biểu đồ so sánh
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Prices', color='blue')
plt.plot(y_pred_linear, label='Linear Regression Predicted Prices', color='orange')
plt.plot(y_pred_lstm, label='LSTM Predicted Prices', color='green')
plt.legend()
plt.title('Actual vs Predicted Prices')
plt.show()

# Kết luận: So sánh các chỉ số giữa hai mô hình
print(f"Linear Regression - MAE: {mae_linear}, MSE: {mse_linear}, R-squared: {r2_linear}")
print(f"LSTM - MAE: {mae_lstm}, MSE: {mse_lstm}, R-squared: {r2_lstm}")
