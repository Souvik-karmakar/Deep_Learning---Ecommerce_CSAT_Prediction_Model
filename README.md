# E-Commerce-Customer-Satisfaction-Score-Prediction-DL-Model

  ![image](https://github.com/VargheseTito/E-Commerce-Customer-Satisfaction-Score-Prediction-DL-Model/assets/110298267/e71b7160-24b3-4c2c-a59b-12b7bceed09b)


E-Commerce Customer Satisfaction Score Prediction Deep Learning Model


Project Summary

Overview

This project focuses on predicting Customer Satisfaction (CSAT) scores using Deep Learning Artificial Neural Networks (ANN). In the context of e-commerce, understanding customer satisfaction through their interactions and feedback is crucial for enhancing service quality, customer retention, and overall business growth. By leveraging advanced neural network models, we aim to accurately forecast CSAT scores based on a myriad of interaction-related features, providing actionable insights for service improvement.

Project Background

Customer satisfaction in the e-commerce sector is a pivotal metric that influences loyalty, repeat business, and word-of-mouth marketing. Traditionally, companies have relied on direct surveys to gauge customer satisfaction, which can be time-consuming and may not always capture the full spectrum of customer experiences. With the advent of deep learning, it's now possible to predict customer satisfaction scores in real-time, offering a granular view of service performance and identifying areas for immediate improvement.

Dataset Overview

The dataset encompasses customer satisfaction scores over a one-month period on an e-commerce platform named "Shopzilla." It consists of the following features:

Unique id: Unique identifier for each record (integer).
Channel name: Name of the customer service channel (object/string).
Category: Category of the interaction (object/string).
Sub-category: Subcategory of the interaction (object/string).
Customer Remarks: Feedback provided by the customer (object/string).
Order id: Identifier for the order associated with the interaction (integer).
Order date time: Date and time of the order (datetime).
Issue reported at: Timestamp when the issue was reported (datetime).
Issue responded: Timestamp when the issue was responded to (datetime).
Survey response date: Date of the customer survey response (datetime).
Customer city: City of the customer (object/string).
Product category: Category of the product (object/string).
Item price: Price of the item (float).
Connected handling time: Time taken to handle the interaction (float).
Agent name: Name of the customer service agent (object/string).
Supervisor: Name of the supervisor (object/string).
Manager: Name of the manager (object/string).
Tenure Bucket: Bucket categorizing agent tenure (object/string).
Agent Shift: Shift timing of the agent (object/string).
CSAT Score: Customer Satisfaction (CSAT) score (integer).

Project Goal

The primary goal of this project is to develop a deep learning model that can accurately predict the CSAT scores based on customer interactions and feedback. By doing so, we aim to provide e-commerce businesses with a powerful tool to monitor and enhance customer satisfaction in real-time, thereby improving service quality and fostering customer loyalty.

![Streamlit_App_Screenshot](https://github.com/VargheseTito/E-Commerce-Customer-Satisfaction-Score-Prediction-DL-Model/assets/110298267/1335c22e-9615-4d64-b74c-324cc6f34744)


Conclusion

Data Overview: The dataset comprises records from the e-commerce industry, focusing on customer service interactions and CSAT scores. It contains 85907 rows and 20 columns, with missing values in several columns such as Customer_city, Product_category, and item_price.

CSAT Importance: CSAT is a crucial KPI for e-commerce businesses, reflecting customer satisfaction with products, services, and overall experience. Understanding CSAT is vital for driving business success.

Variable Insights: The dataset captures detailed information about customer service interactions, including customer feedback, order details, agent information, and timestamps. Understanding these variables provides valuable insights into customer satisfaction drivers.

Exploratory Data Analysis (EDA): EDA aims to gain insights into customer satisfaction patterns. Factors like response time, product category, channel effectiveness, agent tenure, shift timings, and customer feedback are analyzed to uncover potential reasons for CSAT scores.

Response Time Impact: Longer response times correlate with lower CSAT scores, indicating the need for quicker response mechanisms to improve customer satisfaction.

Agent Experience: Agents with longer tenures tend to receive higher CSAT scores, highlighting the importance of experience in delivering satisfactory customer service.

Shift Timings Influence: CSAT scores vary based on agent shift timings, indicating potential workload or resource issues during specific shifts that need attention.

Customer Satisfaction Analysis: The analysis of CSAT scores reveals that a significant portion of customers (69.4%) rated the service with a score of 5, indicating high satisfaction. However, there is also a notable proportion (15%) of customers who experienced poor service, warranting further investigation into the factors contributing to dissatisfaction.

CSAT Score vs. Item Price: A negative correlation between item price and CSAT score suggests that higher-priced items are associated with lower customer satisfaction. This finding underscores the importance of pricing strategies in maintaining high CSAT scores.

Response Time and CSAT Score: Statistical analysis indicates that a mean response time of less than 2 hours is significantly correlated with higher CSAT scores. This underscores the importance of prompt response times in enhancing customer satisfaction.

Price Impact on CSAT Score: Hypothesis testing suggests that items priced above a certain threshold do not significantly affect CSAT scores to go below 3. This finding provides insights into pricing strategies and their impact on customer satisfaction.

Data Preprocessing Techniques: Various techniques such as handling missing values, outlier detection, and categorical encoding were employed to ensure data quality and prepare it for analysis.

Feature Engineering: Feature manipulation, selection, and transformation techniques were utilized to create informative features and enhance the predictive power of the model.

Handling Imbalance in Target Variable: The Synthetic Minority Over-sampling Technique (SMOTE) was applied to address the imbalanced class distribution, ensuring robust model training.

Deep Learning Model Development: The development of a deep learning model using a neural network architecture, wrapped into a KerasClassifier, demonstrated promising performance in predicting CSAT scores, with an overall accuracy of approximately 85%.


