# Project_in_ML

## Main CS: Hybrid RAG for Academic Assistance
An intelligent, unified interface designed for University of Haifa Computer Science students to navigate course curricula, lecturer performance, and academic statistics.

---

## Overview
This system processes dual-intent queries by combining **Unstructured Retrieval** (student reviews) and **Structured Retrieval** (grades and course metrics).It uses a Hybrid Architecture that routes questions to either a Vector Database or a Text-to-SQL engine.

## Key Features
* **Hybrid Routing**: Automatically classifies queries into "SQL" for statistics or "Semantic" for opinions.
* **Domain-Specific Normalization**: Translates student slang (for example "חומרה") into formal course names (e.g., "מבוא לחמרה") using a custom `course_nicknames.json` mapping.
* **Advanced Reranking**: Prioritizes the most relevant student reviews.
* **Text-to-SQL**: Generates valid MySQL queries from natural language.

## System Architecture


1.  **Preprocessing**: Cleans Hebrew text, removes Nikud, and normalizes course aliases.
2.  **Classification**: Routes the query based on vector similarity against a known dataset.
3.  **SQL Path**: Table routing and SQL generation for factual data.
4.  **Semantic Path**: ChromaDB retrieval with hard metadata filtering followed by neural reranking.
5.  **Generation**: Final response synthesis.


## Setup & Installation
1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/YanalSerhan/Project_in_ML.git](https://github.com/YanalSerhan/Project_in_ML.git)
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure Environment**:
    Create a `.env` file with your API keys:
    ```env
    NVIDIA_API_KEY=your_key_here
    OPENAI_API_KEY=your_key_here
    DB_PASSWORD=your_db_password
    MYSQL_DB="Project_ML_DB"
    ```
4.  **Database Setup (One-time Only)**:
    Before running the system for the first time, you must initialize the local MySQL database with the project data. 
    * **Ensure MySQL is running** on your local machine.
    * **Create the Database**: Create a database with the name specified in your `.env`.
    * **Run the Initialization Script**: Execute the script to create tables and import data from JSON sources.
    ```bash
    code here (from json2sql)
    ```

## Execution (How to Run)
To run the application, you need to open two separate terminals:

0. **Step 0: Create a Virtual Environment (Recommended)**

   Creating a virtual environment ensures that the project dependencies do not interfere with other Python projects on your system.
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
1. **Step 1: Start the Backend Server**
   ```bash
   uvicorn main:app --reload
2. **Step 2: Start the UI (Streamlit)**
   ```bash
   streamlit run ui.py

---
