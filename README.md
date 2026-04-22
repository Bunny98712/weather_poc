## ❓ Why This Project

Real-world weather data is fragmented across multiple APIs and often inconsistent.  
This pipeline solves that by:

- Aggregating **weather + flood + earthquake data** into one place
- Normalizing inconsistent API responses
- Storing analytics-ready data in ClickHouse

This enables:
- Disaster monitoring dashboards
- Risk analysis systems
- Location-based alerting# weather_poc


## 🧠 Key Challenges Solved

- Handling **inconsistent API array lengths**
- Aligning multi-source time-series data
- Preventing duplicate ingestion in ClickHouse
- Managing API failures with retry + caching
- Converting semi-structured API responses into structured schema




## 📊 Sample Output

| day       | state | pincode | temp_max | rain_sum | earthquake_mag |
|----------|------|---------|----------|----------|----------------|
| 2026-04-22 | HR   | 122001  | 36.5     | 2.1      | 5.2            |
