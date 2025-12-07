# **Data Science Project (2025/1): Traffy Fondue Dataset Analysis**

**Course:** 2110403 Data Science and Data Engineering (DSDE-CEDT)

**Release Date:** Sat 1st Nov 2025

**Package Submission Deadline:** Sun 7th Dec 2025

**Weight:** 15% of total grade

## **🎯 Objective**

Data science is a discipline aimed at data analysis involving various components (AI/ML, data preparation, data engineering, visualization, MLOps, etc.).

The objective of this project is to **build a practical, fully functional, end-to-end pipeline** that demonstrates diverse, actionable data analysis ideas.

**Examples of expected workflows:**

1. **Pipeline A:** Web scraping → Kafka → Workflow control (Airflow) → Visualization (Power BI).  
2. **Pipeline B:** Big Data Analytics with large data ingestion → Processing (Spark) → Visualization (Power BI) → Storytelling insights.

## **📂 Dataset Details**

**Source:** Traffy Fondue Resources (Aug 2021 \- Jan 2025).

**Description:** Complaint reports submitted by citizens. While primarily aggregated from **Bangkok**, visual analysis of the dataset distribution reveals coverage extends to the **Greater Bangkok Metropolitan Area**, including:

* Nonthaburi  
* Pathum Thani  
* Samut Prakan  
* Chachoengsao

Format: CSV  
Download Link: Google Drive Link

### **Data Dictionary**

| Column Name | Description |
| :---- | :---- |
| ticket\_id | Unique identifier for each ticket/case. |
| type | Category/type of issue reported $$multiple labels$$ . |
| organization | Organization or department handling the ticket. |
| comment | User complaints or feedback regarding public services. |
| photo | Link to photos supporting the issue. |
| photo\_after | Link to photos taken after the issue was addressed. |
| coords | Coordinates (latitude and longitude). |
| address | Physical address. |
| subdistrict | Sub-district of the report. |
| district | District of the report. |
| province | Province of the report. |
| timestamp | Date and time the ticket was created. |
| state | Current state or status of the ticket. |
| star | Numerical rating (0-5). |
| count\_reopen | Number of times the ticket was reopened. |
| last\_activity | Date and time of the last activity on the ticket. |

*(Note: The raw dataset contains approximately 16 columns with varying non-null counts.)*

### **📸 Sample Ticket Data (Visual Analysis)**

Based on the project documentation, valid records contain rich, unstructured text and multi-label classifications. Examples include:

* **Complex Environmental Issues:** Reports combining multiple categories like "Road," "Noise," and "Cleanliness."  
  * *Example context:* A complaint regarding a building construction site (e.g., Sukhumvit 42\) mentioning security guard noise, construction dust, exhaust smoke, and improper garbage disposal.  
* **Public Hygiene & Safety:**  
  * *Categories:* "Canal," "Stray Animals," "Cleanliness."  
  * *Example context:* Requests to catch stray dogs and clear garbage dumped in vacant lots (e.g., near Nawamin 24).  
* **Infrastructure Management:**  
  * *Categories:* "Cleanliness."  
  * *Example context:* Simple requests to relocate public trash bins.

## **🛠 Project Requirements**

### **Group Specification**

* **Members:** Up to 6 members per group (Maximum).

### **Core Components**

The project must be a **fully functional, end-to-end pipeline**. It must include **at least** the following 3 components:

1. **AI/ML Component:** At least one module utilizing Artificial Intelligence or Machine Learning.  
2. **Data Engineering (DE) Component:** At least one module focused on data engineering tasks.  
3. **Visualization (Viz) Component:** Must include either **Geospatial Analysis** or **Graph Visualization**.

*Note: Including more than these three components will enhance the project's depth and may earn additional points.*

### **External Data / Web Scraping Requirements**

* **Traffy Fondue Data Usage:** You are not required to use all 700,000+ records, but must utilize **at least 100,000 records**.  
* **External Data:** You must use **web scraping or APIs** to acquire at least **1,000 records** from external sources.  
  * *Examples:* Organization locations, police stations, PM2.5 status, flood data, traffic data, etc.  
  * *Note:* This data collection is a separate task and does **not** count as the required DE module.

### **Large Language Models (LLM) Policy**

* **Allowed?** Yes, LLMs can be used and counted as the AI/ML module.  
* **Constraint:** Simply calling an API is considered too basic and will **not** count.  
* **Requirement:** You must incorporate advanced techniques (e.g., Chain-of-Thought reasoning, Agentic approaches, or loading reasoning models) to make the implementation compelling.

## **🏆 Scoring Criteria (10%)**

Evaluation is based on:

1. **Completeness:** Implementation of all 3 required components \+ Web scraping data integration.  
2. **Project Interestingness:**  
   * Effort (e.g., additional data sources).  
   * Creativity.  
   * Execution.  
   * Technical Quality.  
   * Other relevant factors.

## **📢 Presentation & Submission (5%)**

### **1\. Presentation Video**

* **Platform:** Upload to YouTube (Public).  
* **Length:** 15 minutes.  
* **Content:**  
  1. Explanation of the data used (Traffy Fondue \+ Additional sources).  
  2. Breakdown of the 3 required components with a **diagram** showing their interconnection.  
  3. Demo showcasing interesting results/analysis.  
* **Sharing:** Share the link in the Discord \#project-showroom channel by the deadline.

### **2\. Deliverables**

* Source Code.  
* Presentation Slides (PPT and PDF formats).

### **3\. Submission Method**

* Create a **Google Drive folder**.  
* Share with viewer access for anyone.  
* Submit the link via **MyCourseVille**.

## **⚠️ Important Remarks & Timeline**

* **Group Creation:** Create your group in MyCourseVille (MCV) by **Sun 9 Nov 2025**.  
* **Submission Deadline:** **Sun 7th Dec 2025**.  
* **Access Check:** Ensure YouTube and Google Drive links are accessible. **If links cannot be opened, the score will be 0\.**  
* **Discord:** Don't forget to post your video link and a short description in \#project-showroom.