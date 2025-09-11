#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chương trình dự đoán và phân tích tiêu thụ năng lượng sử dụng Spark MLlib
Author: Energy Analytics Team
Date: 2025
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (col, concat_ws, to_timestamp, hour, dayofweek, 
                                  month, weekofyear, year, dayofmonth, lit, regexp_extract, when,
                                  avg, stddev, count, sum)
from pyspark.sql.functions import min as spark_min, max as spark_max
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random

def create_spark_session():
    """Tạo Spark Session"""
    print("Khởi tạo Spark Session...")
    spark = SparkSession.builder \
        .appName("EnergyConsumptionPredictionAnalysis") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    print("Spark Session đã được tạo thành công!")
    return spark

def load_and_preprocess_data(spark, file_path):
    """Tải và tiền xử lý dữ liệu"""
    print("\nĐang tải và tiền xử lý dữ liệu...")
    
    df = spark.read.csv(file_path, header=True, sep=";", inferSchema=False)
    print(f"Dữ liệu gốc có {df.count()} dòng và {len(df.columns)} cột")
    
    print("\nKiểm tra dữ liệu thiếu...")
    for c in df.columns:
        n_missing = df.filter((col(c).isNull()) | (col(c) == "") | (col(c) == "?")).count()
        if n_missing > 0:
            print(f"Cột {c}: {n_missing} giá trị thiếu")
    
    cols_to_clean = [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
    ]
    
    print("\nLàm sạch dữ liệu...")
    for c in cols_to_clean:
        df = df.filter((col(c).isNotNull()) & (col(c) != "?"))
    
    for c in cols_to_clean:
        df = df.withColumn(c, col(c).cast("float"))
    
    print("Tạo cột thời gian...")
    try:
        df = df.withColumn("datetime", to_timestamp(
            concat_ws(' ', df.Date, df.Time), "dd/MM/yyyy HH:mm:ss"))
        
        null_count = df.filter(col("datetime").isNull()).count()
        if null_count > 0:
            print(f"Có {null_count} dòng không parse được datetime, đang thử format khác...")
            df = df.withColumn("datetime", to_timestamp(
                concat_ws(' ', df.Date, df.Time), "d/M/yyyy H:mm:ss"))
            
            null_count_2 = df.filter(col("datetime").isNull()).count()
            if null_count_2 > 0:
                print(f"Vẫn có {null_count_2} dòng lỗi, sẽ loại bỏ...")
                df = df.filter(col("datetime").isNotNull())
        
    except Exception as e:
        print(f"Lỗi parse datetime: {e}")
        print("Đang sử dụng phương pháp thay thế...")
        
        df = df.withColumn("day", regexp_extract(col("Date"), r"(\d+)/\d+/\d+", 1).cast("int")) \
               .withColumn("month", regexp_extract(col("Date"), r"\d+/(\d+)/\d+", 1).cast("int")) \
               .withColumn("year", regexp_extract(col("Date"), r"\d+/\d+/(\d+)", 1).cast("int")) \
               .withColumn("hour_time", regexp_extract(col("Time"), r"(\d+):\d+:\d+", 1).cast("int")) \
               .withColumn("minute", regexp_extract(col("Time"), r"\d+:(\d+):\d+", 1).cast("int")) \
               .withColumn("second", regexp_extract(col("Time"), r"\d+:\d+:(\d+)", 1).cast("int"))
        
        df = df.withColumn("datetime_str", 
                          concat_ws("-", 
                                   col("year"), 
                                   when(col("month") < 10, concat_ws("", lit("0"), col("month"))).otherwise(col("month")),
                                   when(col("day") < 10, concat_ws("", lit("0"), col("day"))).otherwise(col("day"))) + " " +
                          concat_ws(":", 
                                   when(col("hour_time") < 10, concat_ws("", lit("0"), col("hour_time"))).otherwise(col("hour_time")),
                                   when(col("minute") < 10, concat_ws("", lit("0"), col("minute"))).otherwise(col("minute")),
                                   when(col("second") < 10, concat_ws("", lit("0"), col("second"))).otherwise(col("second"))))
        
        df = df.withColumn("datetime", to_timestamp(col("datetime_str"), "yyyy-MM-dd HH:mm:ss"))
        df = df.drop("day", "month", "year", "hour_time", "minute", "second", "datetime_str")
    
    df = df.withColumn("year", year(col("datetime"))) \
           .withColumn("month", month(col("datetime"))) \
           .withColumn("day", dayofmonth(col("datetime"))) \
           .withColumn("hour", hour(col("datetime"))) \
           .withColumn("dayofweek", dayofweek(col("datetime"))) \
           .withColumn("weekofyear", weekofyear(col("datetime")))
    
    print(f"Dữ liệu sau xử lý: {df.count()} dòng")
    return df

def handle_outliers(df, target_col="Global_active_power"):
    """Xử lý outliers"""
    print(f"\nXử lý outliers cho cột {target_col}...")
    
    Q1, Q3 = df.approxQuantile(target_col, [0.25, 0.75], 0.01)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outlier_count = df.filter((df[target_col] < lower) | (df[target_col] > upper)).count()
    print(f"Số lượng outliers: {outlier_count}")
    
    df_clean = df.filter((df[target_col] >= lower) & (df[target_col] <= upper))
    print(f"Dữ liệu sau khi loại outliers: {df_clean.count()} dòng")
    return df_clean

def analyze_and_visualize_data(df_no_outlier):
    """Phân tích và trực quan hóa dữ liệu tiêu thụ năng lượng"""
    print("\n" + "="*60)
    print("PHẦN 3: PHÂN TÍCH VÀ TRỰC QUAN HÓA DỮ LIỆU TIÊU THỤ NĂNG LƯỢNG")
    print("="*60)
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    df_analysis = df_no_outlier
    
    print("\nPhân tích tiêu thụ theo giờ...")
    hourly_consumption = df_analysis.groupBy("hour") \
        .agg(avg("Global_active_power").alias("avg_power"),
             stddev("Global_active_power").alias("std_power"),
             spark_min("Global_active_power").alias("min_power"),
             spark_max("Global_active_power").alias("max_power"),
             count("Global_active_power").alias("count_records")) \
        .orderBy("hour")

    print("=== TIÊU THỤ ĐIỆN THEO GIỜ TRONG NGÀY ===")
    hourly_consumption.show(24)

    hourly_pd = hourly_consumption.toPandas()

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(hourly_pd['hour'], hourly_pd['avg_power'], marker='o', linewidth=2, markersize=6)
    plt.fill_between(hourly_pd['hour'], 
                     hourly_pd['avg_power'] - hourly_pd['std_power'],
                     hourly_pd['avg_power'] + hourly_pd['std_power'], 
                     alpha=0.3)
    plt.title('Mức tiêu thụ điện trung bình theo giờ trong ngày', fontsize=14, fontweight='bold')
    plt.xlabel('Giờ trong ngày')
    plt.ylabel('Công suất (kW)')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24))

    plt.subplot(2, 2, 2)
    x = hourly_pd['hour']
    plt.plot(x, hourly_pd['min_power'], marker='v', label='Minimum', linewidth=2, markersize=6, color='blue')
    plt.plot(x, hourly_pd['max_power'], marker='^', label='Maximum', linewidth=2, markersize=6, color='red')
    plt.fill_between(x, hourly_pd['min_power'], hourly_pd['max_power'], alpha=0.2, color='gray')
    plt.title('Mức tiêu thụ Min-Max theo giờ', fontsize=14, fontweight='bold')
    plt.xlabel('Giờ trong ngày')
    plt.ylabel('Công suất (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24))

    daily_consumption = df_analysis.groupBy("dayofweek") \
        .agg(avg("Global_active_power").alias("avg_power"),
             stddev("Global_active_power").alias("std_power"),
             count("Global_active_power").alias("count_records")) \
        .orderBy("dayofweek")

    daily_pd = daily_consumption.toPandas()
    days_name = ['Chủ nhật', 'Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7']
    daily_pd['day_name'] = days_name

    plt.subplot(2, 2, 3)
    bars = plt.bar(daily_pd['day_name'], daily_pd['avg_power'], 
                   color=['red' if day in ['Chủ nhật', 'Thứ 7'] else 'steelblue' for day in daily_pd['day_name']],
                   alpha=0.8)
    plt.title('Mức tiêu thụ điện trung bình theo ngày trong tuần', fontsize=14, fontweight='bold')
    plt.xlabel('Ngày trong tuần')
    plt.ylabel('Công suất trung bình (kW)')
    plt.xticks(rotation=45)
    for bar, value in zip(bars, daily_pd['avg_power']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    monthly_consumption = df_analysis.groupBy("month") \
        .agg(avg("Global_active_power").alias("avg_power"),
             sum("Global_active_power").alias("total_power"),
             stddev("Global_active_power").alias("std_power")) \
        .orderBy("month")

    monthly_pd = monthly_consumption.toPandas()
    months_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_pd['month_name'] = [months_name[i-1] for i in monthly_pd['month']]

    plt.subplot(2, 2, 4)
    plt.plot(monthly_pd['month_name'], monthly_pd['avg_power'], 
             marker='s', linewidth=3, markersize=8, color='green')
    plt.title('Xu hướng tiêu thụ điện theo tháng', fontsize=14, fontweight='bold')
    plt.xlabel('Tháng')
    plt.ylabel('Công suất trung bình (kW)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('energy_consumption_hourly_analysis.png', dpi=300, bbox_inches='tight')
    try:
        plt.show()
    except Exception as e:
        print(f"Không thể hiển thị biểu đồ: {e}. Biểu đồ đã được lưu thành file.")

    print("\n=== THỐNG KÊ MÔ TẢ CHI TIẾT ===")
    stats_summary = df_analysis.select("Global_active_power") \
        .summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")
    stats_summary.show()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    power_data = df_analysis.select("Global_active_power").rdd.flatMap(lambda x: x).collect()
    sample_size = min(10000, len(power_data))
    power_sample = random.sample(power_data, sample_size)
    plt.hist(power_sample, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Phân phối mức tiêu thụ điện', fontsize=14, fontweight='bold')
    plt.xlabel('Công suất (kW)')
    plt.ylabel('Mật độ')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.boxplot(power_sample, patch_artist=True, 
                boxprops=dict(facecolor='lightgreen', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    plt.title('Box Plot - Phân phối tiêu thụ điện', fontsize=14, fontweight='bold')
    plt.ylabel('Công suất (kW)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    peak_hours_data = df_analysis.filter(col("hour").isin([7, 8, 19, 20, 21])) \
                               .select("hour", "Global_active_power") \
                               .sample(0.1)
    peak_hours_pd = peak_hours_data.toPandas()
    if not peak_hours_pd.empty:
        sns.violinplot(data=peak_hours_pd, x='hour', y='Global_active_power')
        plt.title('Phân phối tiêu thụ trong giờ cao điểm', fontsize=14, fontweight='bold')
        plt.xlabel('Giờ')
        plt.ylabel('Công suất (kW)')

    plt.tight_layout()
    plt.savefig('energy_distribution_analysis.png', dpi=300, bbox_inches='tight')
    try:
        plt.show()
    except Exception as e:
        print(f"Không thể hiển thị biểu đồ: {e}. Biểu đồ đã được lưu thành file.")

    print("\n=== PHÂN TÍCH CÁC THIẾT BỊ TIÊU THỤ PHỤ ===")
    submeter_contrib = df_analysis.agg(
        avg("Sub_metering_1").alias("avg_sub1"),
        avg("Sub_metering_2").alias("avg_sub2"), 
        avg("Sub_metering_3").alias("avg_sub3"),
        avg("Global_active_power").alias("avg_total")
    ).collect()[0]

    print(f"Sub-metering 1 (Bếp): {submeter_contrib['avg_sub1']:.3f} kW ({min(100, submeter_contrib['avg_sub1']/submeter_contrib['avg_total']*100):.1f}%)")
    print(f"Sub-metering 2 (Giặt ủi): {submeter_contrib['avg_sub2']:.3f} kW ({min(100, submeter_contrib['avg_sub2']/submeter_contrib['avg_total']*100):.1f}%)")
    print(f"Sub-metering 3 (Điều hòa/Nước nóng): {submeter_contrib['avg_sub3']:.3f} kW ({min(100, submeter_contrib['avg_sub3']/submeter_contrib['avg_total']*100):.1f}%)")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sub1 = submeter_contrib['avg_sub1']
    sub2 = submeter_contrib['avg_sub2'] 
    sub3 = submeter_contrib['avg_sub3']
    total = submeter_contrib['avg_total']
    
    sub_total = sub1 + sub2 + sub3
    if sub_total > total:
        factor = total / sub_total * 0.9
        sub1 = sub1 * factor
        sub2 = sub2 * factor  
        sub3 = sub3 * factor
        other = total - sub1 - sub2 - sub3
    else:
        other = total - sub_total
    
    other = max(0, other)
    
    labels = ['Bếp', 'Giặt ủi', 'Điều hòa/Nước nóng', 'Khác']
    sizes = [sub1, sub2, sub3, other]
    colors = ['gold', 'lightcoral', 'lightskyblue', 'lightgray']
    explode = (0.05, 0.05, 0.05, 0)

    valid_indices = [i for i, size in enumerate(sizes) if size > 0]
    valid_labels = [labels[i] for i in valid_indices]
    valid_sizes = [sizes[i] for i in valid_indices]
    valid_colors = [colors[i] for i in valid_indices]
    valid_explode = [explode[i] for i in valid_indices]

    plt.pie(valid_sizes, explode=valid_explode, labels=valid_labels, colors=valid_colors, 
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title('Tỷ lệ đóng góp tiêu thụ điện theo thiết bị', fontsize=14, fontweight='bold')

    submeter_hourly = df_analysis.groupBy("hour") \
        .agg(avg("Sub_metering_1").alias("avg_sub1"),
             avg("Sub_metering_2").alias("avg_sub2"),
             avg("Sub_metering_3").alias("avg_sub3")) \
        .orderBy("hour")

    submeter_hourly_pd = submeter_hourly.toPandas()

    plt.subplot(1, 3, 2)
    plt.plot(submeter_hourly_pd['hour'], submeter_hourly_pd['avg_sub1'], 
             marker='o', label='Bếp', linewidth=2)
    plt.plot(submeter_hourly_pd['hour'], submeter_hourly_pd['avg_sub2'], 
             marker='s', label='Giặt ủi', linewidth=2)
    plt.plot(submeter_hourly_pd['hour'], submeter_hourly_pd['avg_sub3'], 
             marker='^', label='Điều hòa/Nước nóng', linewidth=2)
    plt.title('Xu hướng tiêu thụ theo giờ - Từng thiết bị', fontsize=14, fontweight='bold')
    plt.xlabel('Giờ trong ngày')
    plt.ylabel('Công suất (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24))

    plt.subplot(1, 3, 3)
    heatmap_data = df_analysis.groupBy("hour", "dayofweek") \
        .agg(avg("Global_active_power").alias("avg_power")) \
        .toPandas()

    if not heatmap_data.empty:
        heatmap_pivot = heatmap_data.pivot(index='hour', columns='dayofweek', values='avg_power')
        heatmap_pivot.columns = ['CN', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
        sns.heatmap(heatmap_pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
                    cbar_kws={'label': 'Công suất (kW)'})
        plt.title('Heatmap tiêu thụ điện theo giờ và ngày', fontsize=14, fontweight='bold')
        plt.xlabel('Ngày trong tuần')
        plt.ylabel('Giờ trong ngày')

    plt.tight_layout()
    plt.savefig('energy_device_analysis.png', dpi=300, bbox_inches='tight')
    try:
        plt.show()
    except Exception as e:
        print(f"Không thể hiển thị biểu đồ: {e}. Biểu đồ đã được lưu thành file.")

    return {
        'hourly_pd': hourly_pd,
        'daily_pd': daily_pd,
        'monthly_pd': monthly_pd,
        'submeter_contrib': submeter_contrib,
        'hourly_consumption': hourly_consumption,
        'daily_consumption': daily_consumption,
        'monthly_consumption': monthly_consumption
    }

def create_features(df):
    """Tạo feature vector"""
    print("\nTạo feature vector...")
    
    feature_cols = [
        "Global_reactive_power", "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
        "hour", "dayofweek", "month"
    ]
    
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw"
    )
    
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=True
    )
    
    feature_pipeline = Pipeline(stages=[assembler, scaler])
    feature_model = feature_pipeline.fit(df)
    df_features = feature_model.transform(df)
    
    print("Feature vector đã được tạo và chuẩn hóa!")
    return df_features.select("features", "Global_active_power", "datetime", "hour", "dayofweek", "month"), feature_model

def train_models(train_data):
    """Huấn luyện nhiều mô hình ML"""
    print("\nHuấn luyện các mô hình Machine Learning...")
    
    models = {}
    
    print("Huấn luyện Linear Regression...")
    lr = LinearRegression(
        featuresCol="features",
        labelCol="Global_active_power",
        regParam=0.01
    )
    models['Linear Regression'] = lr.fit(train_data)
    
    print("Huấn luyện Random Forest...")
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="Global_active_power",
        numTrees=50,
        maxDepth=10,
        seed=42
    )
    models['Random Forest'] = rf.fit(train_data)
    
    print("Huấn luyện Gradient Boosted Trees...")
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="Global_active_power",
        maxIter=50,
        maxDepth=8,
        seed=42
    )
    models['GBT'] = gbt.fit(train_data)
    
    print("Tất cả mô hình đã được huấn luyện!")
    return models

def evaluate_models(models, test_data):
    """Đánh giá các mô hình"""
    print("\nĐánh giá hiệu suất các mô hình...")
    
    evaluator_rmse = RegressionEvaluator(
        labelCol="Global_active_power",
        predictionCol="prediction",
        metricName="rmse"
    )
    
    evaluator_r2 = RegressionEvaluator(
        labelCol="Global_active_power",
        predictionCol="prediction",
        metricName="r2"
    )
    
    results = {}
    
    for name, model in models.items():
        predictions = model.transform(test_data)
        rmse = evaluator_rmse.evaluate(predictions)
        r2 = evaluator_r2.evaluate(predictions)
        
        results[name] = {
            'RMSE': rmse,
            'R²': r2,
            'predictions': predictions
        }
        
        print(f"   {name}:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²: {r2:.4f}")
    
    best_model_name = min(results.keys(), key=lambda x: results[x]['RMSE'])
    print(f"\nMô hình tốt nhất: {best_model_name}")
    return results, best_model_name

def visualize_predictions(results, best_model_name):
    """Trực quan hóa kết quả dự đoán"""
    print("\nTrực quan hóa kết quả dự đoán...")
    
    best_predictions = results[best_model_name]['predictions']
    pred_sample = best_predictions.select("Global_active_power", "prediction").sample(0.01).toPandas()
    
    if not pred_sample.empty:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.scatter(pred_sample['Global_active_power'], pred_sample['prediction'], alpha=0.5)
        plt.plot([pred_sample['Global_active_power'].min(), pred_sample['Global_active_power'].max()], 
                 [pred_sample['Global_active_power'].min(), pred_sample['Global_active_power'].max()], 
                 'r--', lw=2)
        plt.xlabel('Thực tế (kW)')
        plt.ylabel('Dự đoán (kW)')
        plt.title(f'So sánh Thực tế vs Dự đoán - {best_model_name}')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        errors = pred_sample['prediction'] - pred_sample['Global_active_power']
        plt.hist(errors, bins=30, alpha=0.7, color='orange')
        plt.xlabel('Sai số (kW)')
        plt.ylabel('Tần suất')
        plt.title('Phân phối sai số dự đoán')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        model_names = list(results.keys())
        rmse_values = [results[name]['RMSE'] for name in model_names]
        model_colors = {'Linear Regression': 'blue', 'Random Forest': 'green', 'GBT': 'orange'}
        colors_rmse = [model_colors.get(name, 'gray') for name in model_names]
        bars = plt.bar(model_names, rmse_values, alpha=0.8, color=colors_rmse)
        plt.xlabel('Mô hình')
        plt.ylabel('RMSE')
        plt.title('So sánh RMSE các mô hình')
        plt.xticks(rotation=45)
        max_rmse = max(rmse_values)
        for bar, value in zip(bars, rmse_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_rmse*0.02,
                     f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        plt.ylim(0, max_rmse * 1.15)
        
        plt.subplot(2, 2, 4)
        r2_values = [results[name]['R²'] for name in model_names]
        colors_r2 = [model_colors.get(name, 'gray') for name in model_names]
        bars = plt.bar(model_names, r2_values, color=colors_r2, alpha=0.8)
        plt.xlabel('Mô hình')
        plt.ylabel('R² Score')
        plt.title('So sánh R² Score các mô hình')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.4)
        for bar, value in zip(bars, r2_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        try:
            plt.show()
        except Exception as e:
            print(f"Không thể hiển thị biểu đồ: {e}. Biểu đồ đã được lưu thành file.")

def predict_yearly_consumption(df_features, best_model, spark, df_clean):
    """Dự đoán mức tiêu thụ năng lượng trung bình cho năm tiếp theo"""
    print("\n" + "="*60)
    print("DỰ ĐOÁN TIÊU THỤ NĂNG LƯỢNG CHO NĂM TIẾP THEO")
    print("="*60)
    
    current_year = df_features.select(year(col("datetime")).alias("year")).agg(spark_max("year")).collect()[0][0]
    next_year = current_year + 1
    print(f"📅 Dự đoán cho năm: {next_year}")
    
    avg_stats = df_clean.agg(
        avg("Global_reactive_power").alias("avg_reactive"),
        avg("Voltage").alias("avg_voltage"),
        avg("Global_intensity").alias("avg_intensity"),
        avg("Sub_metering_1").alias("avg_sub1"),
        avg("Sub_metering_2").alias("avg_sub2"),
        avg("Sub_metering_3").alias("avg_sub3")
    ).collect()[0]
    
    prediction_data = []
    for month in range(1, 13):
        days_in_month = 31 if month in [1, 3, 5, 7, 8, 10, 12] else 30 if month in [4, 6, 9, 11] else 28
        for day in range(1, days_in_month + 1):
            dayofweek = ((day + month) % 7) + 1
            for hour in range(24):
                seasonal_factor = 1.15 if month in [12, 1, 2] else 1.1 if month in [6, 7, 8] else 0.95
                hourly_factor = 1.2 if 7 <= hour <= 9 or 18 <= hour <= 22 else 0.7 if 0 <= hour <= 5 else 1.0
                weekly_factor = 1.05 if dayofweek in [1, 7] else 1.0
                
                prediction_data.append({
                    'Global_reactive_power': float(avg_stats['avg_reactive'] * seasonal_factor * hourly_factor),
                    'Voltage': float(avg_stats['avg_voltage']),
                    'Global_intensity': float(avg_stats['avg_intensity'] * seasonal_factor * hourly_factor),
                    'Sub_metering_1': float(avg_stats['avg_sub1'] * seasonal_factor * hourly_factor),
                    'Sub_metering_2': float(avg_stats['avg_sub2'] * weekly_factor),
                    'Sub_metering_3': float(avg_stats['avg_sub3'] * seasonal_factor),
                    'hour': hour,
                    'dayofweek': dayofweek,
                    'month': month,
                    'year': next_year,
                    'day': day
                })
    
    schema = StructType([
        StructField("Global_reactive_power", FloatType(), True),
        StructField("Voltage", FloatType(), True),
        StructField("Global_intensity", FloatType(), True),
        StructField("Sub_metering_1", FloatType(), True),
        StructField("Sub_metering_2", FloatType(), True),
        StructField("Sub_metering_3", FloatType(), True),
        StructField("hour", IntegerType(), True),
        StructField("dayofweek", IntegerType(), True),
        StructField("month", IntegerType(), True),
        StructField("year", IntegerType(), True),
        StructField("day", IntegerType(), True)
    ])
    
    future_df = spark.createDataFrame(prediction_data, schema)
    feature_cols = [
        "Global_reactive_power", "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
        "hour", "dayofweek", "month"
    ]
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
    feature_pipeline = Pipeline(stages=[assembler, scaler])
    sample_data = df_clean.sample(0.1)
    feature_model = feature_pipeline.fit(sample_data)
    future_df_features = feature_model.transform(future_df)
    
    print("🔮 Đang thực hiện dự đoán...")
    yearly_predictions = best_model.transform(future_df_features)
    
    yearly_stats = yearly_predictions.agg(
        avg("prediction").alias("avg_yearly_consumption"),
        spark_min("prediction").alias("min_consumption"),
        spark_max("prediction").alias("max_consumption"),
        stddev("prediction").alias("std_consumption")
    ).collect()[0]
    
    monthly_predictions = yearly_predictions.groupBy("month") \
        .agg(avg("prediction").alias("avg_monthly_consumption"),
             sum("prediction").alias("total_monthly_consumption")) \
        .orderBy("month")
    monthly_pred_pd = monthly_predictions.toPandas()
    
    hourly_predictions = yearly_predictions.groupBy("hour") \
        .agg(avg("prediction").alias("avg_pred"),
             stddev("prediction").alias("std_pred")) \
        .orderBy("hour")
    hourly_pred_pd = hourly_predictions.toPandas()
    
    print(f"\n📊 KẾT QUẢ DỰ ĐOÁN NĂM {next_year}:")
    print(f"   🔹 Mức tiêu thụ trung bình: {yearly_stats['avg_yearly_consumption']:.3f} kW")
    print(f"   🔹 Tổng tiêu thụ ước tính: {yearly_stats['avg_yearly_consumption'] * 8760:.0f} kWh/năm")
    print(f"   🔹 Mức tiêu thụ thấp nhất: {yearly_stats['min_consumption']:.3f} kW")
    print(f"   🔹 Mức tiêu thụ cao nhất: {yearly_stats['max_consumption']:.3f} kW")
    print(f"   🔹 Độ lệch chuẩn: {yearly_stats['std_consumption']:.3f} kW")
    
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 3, 1)
    months_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_pred_pd['month_name'] = [months_name[i-1] for i in monthly_pred_pd['month']]
    bars = plt.bar(monthly_pred_pd['month_name'], monthly_pred_pd['avg_monthly_consumption'], 
                   color='lightblue', alpha=0.8, edgecolor='navy', linewidth=1)
    plt.title(f'Dự đoán tiêu thụ trung bình theo tháng ', fontsize=14, fontweight='bold')
    plt.xlabel('Tháng')
    plt.ylabel('Công suất trung bình (kW)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    for bar, value in zip(bars, monthly_pred_pd['avg_monthly_consumption']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{value:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.subplot(2, 3, 2)
    plt.plot(monthly_pred_pd['month_name'], monthly_pred_pd['total_monthly_consumption']/1000, 
             marker='o', linewidth=3, markersize=8, color='red')
    plt.title(f'Tổng tiêu thụ dự đoán theo tháng ', fontsize=14, fontweight='bold')
    plt.xlabel('Tháng')
    plt.ylabel('Tổng tiêu thụ (MWh)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.plot(hourly_pred_pd['hour'], hourly_pred_pd['avg_pred'], 
             marker='s', linewidth=2, markersize=6, color='green', label='Dự đoán trung bình')
    plt.fill_between(hourly_pred_pd['hour'],
                     hourly_pred_pd['avg_pred'] - 1.96 * hourly_pred_pd['std_pred'],
                     hourly_pred_pd['avg_pred'] + 1.96 * hourly_pred_pd['std_pred'],
                     alpha=0.3, color='green', label='Khoảng tin cậy 95%')
    plt.title(f'Mức tiêu thụ điện trung bình theo giờ trong ngày', fontsize=14, fontweight='bold')
    plt.xlabel('Giờ trong ngày')
    plt.ylabel('Công suất trung bình (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24))
    
    plt.subplot(2, 3, 4)
    pred_sample = yearly_predictions.select("prediction").sample(0.1).rdd.flatMap(lambda x: x).collect()
    sample_size = min(5000, len(pred_sample))
    pred_sample = random.sample(pred_sample, sample_size)
    plt.hist(pred_sample, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(yearly_stats['avg_yearly_consumption'], color='red', linestyle='--', 
                linewidth=2, label=f'Trung bình: {yearly_stats["avg_yearly_consumption"]:.3f} kW')
    plt.title('Phân phối dự đoán tiêu thụ năng lượng', fontsize=14, fontweight='bold')
    plt.xlabel('Công suất (kW)')
    plt.ylabel('Mật độ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    seasons = {'Xuân': [3, 4, 5], 'Hè': [6, 7, 8], 'Thu': [9, 10, 11], 'Đông': [12, 1, 2]}
    seasonal_consumption = []
    season_names = []
    for season, months in seasons.items():
        season_avg = monthly_pred_pd[monthly_pred_pd['month'].isin(months)]['avg_monthly_consumption'].mean()
        seasonal_consumption.append(season_avg)
        season_names.append(season)
    colors = ['lightgreen', 'gold', 'orange', 'lightblue']
    bars = plt.bar(season_names, seasonal_consumption, color=colors, alpha=0.8, edgecolor='black')
    plt.title(f'Dự đoán tiêu thụ theo mùa', fontsize=14, fontweight='bold')
    plt.xlabel('Mùa')
    plt.ylabel('Công suất trung bình (kW)')
    for bar, value in zip(bars, seasonal_consumption):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.subplot(2, 3, 6)
    current_year_avg = df_features.agg(avg("Global_active_power")).collect()[0][0]
    predicted_avg = yearly_stats['avg_yearly_consumption']
    growth_rate = ((predicted_avg - current_year_avg) / current_year_avg) * 100
    years = [current_year, next_year]
    consumptions = [current_year_avg, predicted_avg]
    plt.plot(years, consumptions, marker='o', linewidth=3, markersize=10, color='purple')
    plt.fill_between(years, consumptions, alpha=0.3, color='purple')
    plt.annotate(f'Tăng trưởng: {growth_rate:+.1f}%', 
                xy=(next_year, predicted_avg), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.title('Xu hướng tăng trưởng tiêu thụ năng lượng', fontsize=14, fontweight='bold')
    plt.xlabel('Năm')
    plt.ylabel('Công suất trung bình (kW)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'yearly_energy_prediction_{next_year}.png', dpi=300, bbox_inches='tight')
    try:
        plt.show()
    except Exception as e:
        print(f"Không thể hiển thị biểu đồ: {e}. Biểu đồ đã được lưu thành file.")
    
    print(f"\n📈 PHÂN TÍCH CHI TIẾT DỰ ĐOÁN NĂM {next_year}:")
    print(f"   🔹 Xu hướng tăng trưởng: {growth_rate:+.1f}%")
    if growth_rate > 5:
        print("   ⚠️  Cảnh báo: Mức tăng trưởng cao, cần biện pháp tiết kiệm năng lượng")
    elif growth_rate < -5:
        print("   ✅ Tích cực: Dự đoán giảm tiêu thụ, hiệu quả tiết kiệm năng lượng")
    else:
        print("   📊 Ổn định: Mức tăng trưởng trong phạm vi bình thường")
    
    print(f"\n🏆 THÁNG TIÊU THỤ CAO NHẤT DỰ ĐOÁN:")
    max_month_idx = monthly_pred_pd['avg_monthly_consumption'].idxmax()
    max_month = monthly_pred_pd.loc[max_month_idx]
    print(f"   📅 {max_month['month_name']}: {max_month['avg_monthly_consumption']:.3f} kW")
    
    print(f"\n🏅 THÁNG TIÊU THỤ THẤP NHẤT DỰ ĐOÁN:")
    min_month_idx = monthly_pred_pd['avg_monthly_consumption'].idxmin()
    min_month = monthly_pred_pd.loc[min_month_idx]
    print(f"   📅 {min_month['month_name']}: {min_month['avg_monthly_consumption']:.3f} kW")
    
    estimated_cost_per_kwh = 2500
    total_yearly_kwh = yearly_stats['avg_yearly_consumption'] * 8760
    estimated_yearly_cost = total_yearly_kwh * estimated_cost_per_kwh
    
    print(f"\n💰 ƯỚC TÍNH CHI PHÍ NĂM {next_year}:")
    print(f"   💡 Tổng tiêu thụ: {total_yearly_kwh:,.0f} kWh")
    print(f"   💵 Chi phí ước tính: {estimated_yearly_cost:,.0f} VNĐ")
    print(f"   💸 Chi phí trung bình/tháng: {estimated_yearly_cost/12:,.0f} VNĐ")
    
    return yearly_predictions, yearly_stats, monthly_pred_pd, next_year

def generate_insights(analysis_results, yearly_stats, monthly_pred_pd, next_year):
    """Tạo insights từ phân tích và dự đoán hàng năm"""
    print("\n" + "="*80)
    print("BÁO CÁO TỔNG KẾT PHÂN TÍCH VÀ DỰ ĐOÁN TIÊU THỤ NĂNG LƯỢNG")
    print("="*80)
    
    hourly_pd = analysis_results['hourly_pd']
    daily_pd = analysis_results['daily_pd']
    monthly_pd = analysis_results['monthly_pd']
    submeter_contrib = analysis_results['submeter_contrib']
    
    peak_hour = hourly_pd.loc[hourly_pd['avg_power'].idxmax()]
    low_hour = hourly_pd.loc[hourly_pd['avg_power'].idxmin()]
    print(f"\n🕐 GIỜ CAO ĐIỂM: {int(peak_hour['hour'])}:00 - Tiêu thụ: {peak_hour['avg_power']:.3f} kW")
    print(f"🕐 GIỜ THẤP ĐIỂM: {int(low_hour['hour'])}:00 - Tiêu thụ: {low_hour['avg_power']:.3f} kW")
    print(f"📊 CHÊNH LỆCH: {peak_hour['avg_power'] - low_hour['avg_power']:.3f} kW ({(peak_hour['avg_power'] - low_hour['avg_power'])/low_hour['avg_power']*100:.1f}%)")

    peak_day = daily_pd.loc[daily_pd['avg_power'].idxmax()]
    low_day = daily_pd.loc[daily_pd['avg_power'].idxmin()]
    print(f"\n📅 NGÀY TIÊU THỤ CAO NHẤT: {peak_day['day_name']} - {peak_day['avg_power']:.3f} kW")
    print(f"📅 NGÀY TIÊU THỤ THẤP NHẤT: {low_day['day_name']} - {low_day['avg_power']:.3f} kW")

    peak_month = monthly_pd.loc[monthly_pd['avg_power'].idxmax()]
    low_month = monthly_pd.loc[monthly_pd['avg_power'].idxmin()]
    print(f"\n🗓️ THÁNG TIÊU THỤ CAO NHẤT: {peak_month['month_name']} - {peak_month['avg_power']:.3f} kW")
    print(f"🗓️ THÁNG TIÊU THỤ THẤP NHẤT: {low_month['month_name']} - {low_month['avg_power']:.3f} kW")

    print(f"\n🏠 THIẾT BỊ TIÊU THỤ NHIỀU NHẤT:")
    devices = [
        ("Bếp", submeter_contrib['avg_sub1']),
        ("Điều hòa/Nước nóng", submeter_contrib['avg_sub3']),
        ("Giặt ủi", submeter_contrib['avg_sub2'])
    ]
    devices.sort(key=lambda x: x[1], reverse=True)
    for i, (device, consumption) in enumerate(devices, 1):
        percentage = consumption/submeter_contrib['avg_total']*100
        print(f"   {i}. {device}: {consumption:.3f} kW ({percentage:.1f}%)")

    print(f"\n🔮 DỰ ĐOÁN CHO NĂM {next_year}:")
    print(f"   Mức tiêu thụ trung bình dự đoán: {yearly_stats['avg_yearly_consumption']:.3f} kW")
    print(f"   Tháng cao điểm dự đoán: {monthly_pred_pd.loc[monthly_pred_pd['avg_monthly_consumption'].idxmax(), 'month_name']}")
    print(f"   Tháng thấp điểm dự đoán: {monthly_pred_pd.loc[monthly_pred_pd['avg_monthly_consumption'].idxmin(), 'month_name']}")

    print(f"\n💡 ĐỀ XUẤT TIẾT KIỆM NĂNG LƯỢNG:")
    print("1. 🏠 Sử dụng thiết bị điện vào giờ thấp điểm (giảm 15-20% chi phí)")
    print("2. ❄️ Điều chỉnh nhiệt độ điều hòa +2°C vào giờ cao điểm")
    print("3. 💡 Thay thế đèn truyền thống bằng đèn LED")
    print("4. 🔌 Tắt các thiết bị standby vào ban đêm")
    print("5. ☀️ Cân nhắc lắp đặt hệ thống năng lượng mặt trời")
    print("6. 🏠 Sử dụng thiết bị có nhãn năng lượng cao")
    print("7. 📱 Lắp đặt hệ thống quản lý năng lượng thông minh")

    print(f"\n📈 TIỀM NĂNG TIẾT KIỆM:")
    potential_savings = peak_hour['avg_power'] * 0.15
    print(f"   Tiết kiệm ước tính: {potential_savings:.3f} kW/giờ cao điểm")
    print(f"   Tương đương: {potential_savings * 24 * 30:.0f} kWh/tháng")

def save_analysis_results(analysis_results, spark):
    """Lưu kết quả phân tích vào HDFS"""
    print(f"\n💾 ĐANG LƯU KẾT QUẢ PHÂN TÍCH...")
    try:
        hourly_consumption = analysis_results['hourly_consumption']
        daily_consumption = analysis_results['daily_consumption']
        monthly_consumption = analysis_results['monthly_consumption']
        hourly_consumption.coalesce(1).write.mode("overwrite").csv("hdfs:///energy_analysis/hourly_consumption", header=True)
        daily_consumption.coalesce(1).write.mode("overwrite").csv("hdfs:///energy_analysis/daily_consumption", header=True)
        monthly_consumption.coalesce(1).write.mode("overwrite").csv("hdfs:///energy_analysis/monthly_consumption", header=True)
        print("✅ Đã lưu kết quả phân tích vào HDFS thành công!")
    except Exception as e:
        print(f"⚠️ Lỗi khi lưu vào HDFS: {e}")
        print("💡 Tip: Kiểm tra cấu hình Hadoop và quyền ghi")

def main():
    """Hàm chính"""
    print("🚀 CHƯƠNG TRÌNH DỰ ĐOÁN VÀ PHÂN TÍCH TIÊU THỤ NĂNG LƯỢNG")
    print("="*60)
    
    import matplotlib
    matplotlib.use('Agg')
    
    spark = create_spark_session()
    
    try:
        file_path = "hdfs:///energy_data/household_power_consumption.txt"
        
        print("\n" + "="*50)
        print("PHẦN 1: TIỀN XỬ LÝ DỮ LIỆU")
        print("="*50)
        df = load_and_preprocess_data(spark, file_path)
        
        print("\n" + "="*50)
        print("PHẦN 2: XỬ LÝ OUTLIERS")
        print("="*50)
        df_clean = handle_outliers(df)
        
        analysis_results = analyze_and_visualize_data(df_clean) # PHẦN 3: PHÂN TÍCH VÀ TRỰC QUAN HÓA DỮ LIỆU TIÊU THỤ NĂNG LƯỢNG
        
        print("\n" + "="*50)
        print("PHẦN 4: CHUẨN BỊ DỮ LIỆU CHO MACHINE LEARNING")
        print("="*50)
        df_features, _ = create_features(df_clean)
        
        print("\n📊 Chia dữ liệu train/test (80/20)...")
        train_data, test_data = df_features.randomSplit([0.8, 0.2], seed=42)
        print(f"📈 Train: {train_data.count()} dòng, Test: {test_data.count()} dòng")
        
        print("\n" + "="*50)
        print("PHẦN 5: HUẤN LUYỆN CÁC MÔ HÌNH MACHINE LEARNING")
        print("="*50)
        models = train_models(train_data)
        
        print("\n" + "="*50)
        print("PHẦN 6: ĐÁNH GIÁ HIỆU SUẤT MÔ HÌNH")
        print("="*50)
        results, best_model_name = evaluate_models(models, test_data)
        best_model = models[best_model_name]
        
        visualize_predictions(results, best_model_name)
        
        print("\n" + "="*50)
        print("PHẦN 7: DỰ ĐOÁN TIÊU THỤ NĂNG LƯỢNG NĂM TIẾP THEO")
        print("="*50)
        yearly_predictions, yearly_stats, monthly_pred_pd, next_year = predict_yearly_consumption(df_features, best_model, spark, df_clean)
        
        generate_insights(analysis_results, yearly_stats, monthly_pred_pd, next_year)
        
        save_analysis_results(analysis_results, spark)
        
        print("\n" + "="*60)
        print("🎉 HOÀN THÀNH! Chương trình đã chạy thành công!")
        print("📊 Các file biểu đồ đã được lưu:")
        print("   • energy_consumption_hourly_analysis.png")
        print("   • energy_distribution_analysis.png")
        print("   • energy_device_analysis.png")
        print("   • model_performance_comparison.png")
        print(f"   • yearly_energy_prediction_{next_year}.png")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        print("💡 Kiểm tra lại:")
        print("   • Đường dẫn file dữ liệu")
        print("   • Cấu hình Hadoop/Spark")
        print("   • Kết nối HDFS")
        import traceback
        traceback.print_exc()
    
    finally:
        spark.stop()
        print("🔌 Spark Session đã được đóng.")

if __name__ == "__main__":
    main()