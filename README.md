# ⚡ Energy Consumption Analysis & Forecasting

![Big Data](https://img.shields.io/badge/Big_Data-Spark%20%7C%20Hadoop-orange?style=flat&logo=apachespark&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Spark_MLlib-red?style=flat&logo=scikitlearn&logoColor=white)
![Language](https://img.shields.io/badge/Language-Python-blue?style=flat&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat)

Dự án xây dựng một **Pipeline Machine Learning hoàn chỉnh** trên nền tảng Big Data để phân tích và dự báo mức tiêu thụ năng lượng. Hệ thống có khả năng xử lý bộ dữ liệu lớn (trên **2 triệu bản ghi**) sử dụng hệ sinh thái Hadoop và Spark.

## 🌟 Tính Năng Nổi Bật (Key Features)

*   **🚀 Xử Lý Dữ Liệu Lớn (Big Data Processing):** Triển khai trên kiến trúc **Hadoop HDFS** và **Apache Spark**, đảm bảo hiệu năng cao khi xử lý khối lượng dữ liệu khổng lồ mà các công cụ truyền thống khó đáp ứng.
*   **🔄 Quy Trình ETL Mạnh Mẽ:** Thực hiện quy trình **Extract-Transform-Load** bằng **PySpark**:
    *   Xử lý giá trị bị thiếu (NULL handling).
    *   Chuyển đổi kiểu dữ liệu quy mô lớn.
    *   **Feature Engineering:** Tạo ra các thuộc tính thời gian (Time-series features) để phục vụ mô hình.
*   **📊 Phân Tích Khám Phá (EDA):** Sử dụng **Spark SQL** và **DataFrame API** để truy vấn dữ liệu, kết hợp với **Matplotlib/Seaborn** để trực quan hóa các xu hướng tiêu thụ điện năng và tìm ra insight quan trọng.
*   **🤖 Mô Hình Dự Báo Chính Xác:** Xây dựng và huấn luyện mô hình hồi quy **Gradient-Boosted Trees (GBT)** sử dụng thư viện **Spark MLlib**.
*   **⚙️ Tối Ưu Hóa Đặc Trưng:** Áp dụng kỹ thuật Vector hóa đặc trưng (**VectorAssembler**) để chuẩn hóa dữ liệu đầu vào cho mô hình máy học.
*   **📈 Hiệu Suất Ấn Tượng:** Mô hình đạt độ chính xác cao với chỉ số **RMSE (Root Mean Square Error) là 0.1333 kW** trên tập dữ liệu kiểm thử.

## 🛠️ Công Nghệ Sử Dụng (Tech Stack)

*   **Core:** Python, PySpark
*   **Big Data Platform:** Apache Hadoop (HDFS), Apache Spark
*   **Machine Learning:** Spark MLlib (Gradient-Boosted Trees)
*   **Data Analysis & Visualization:** Spark SQL, Pandas, Matplotlib, Seaborn
