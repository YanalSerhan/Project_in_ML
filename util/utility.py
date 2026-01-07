from config.DB_Connection import get_connection
from collections import defaultdict

def fetch_grades(course_name: str, lecturer_name: str):
    """
    Fetch a list of grades (ints/floats) for a given course name and lecturer name.
    """

    sql = """
        SELECT `avg`, `year`, semester, moed
        FROM q
        WHERE course = %s
          AND lecture = %s
        ORDER BY year DESC LIMIT 6;
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(sql, (course_name, lecturer_name))
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return rows

def fetch_kdams(course_name: str):
    """
    Fetch a list of grades (ints/floats) for a given course name and lecturer name.
    """

    sql = """
        SELECT kdams
        FROM Tkdams
        WHERE name = %s;
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(sql, [course_name])
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return rows


def docs2str(docs, grades_flag=False) -> str:
    """
    Convert retrieved documents into a readable context string
    including selected metadata: course_name, lecturer, date.
    """
    out_str = ""

    for doc in docs:
        meta = getattr(doc, "metadata", {})

        course_name = meta.get("course_name", "Unknown Course")
        lecturer    = meta.get("lecturer", "Unknown Lecturer")
        date        = meta.get("date", "Unknown Date")
        grades      = "N/A"
        #kdams     = "N/A"
        if grades_flag:
            grades = fetch_grades(course_name, lecturer)
            #kdams = fetch_kdams(course_name)
        
        # header with metadata
        out_str += (
            f"course_name: {course_name}\n"
            f"lecturer: {lecturer}\n"
            f"date: {date}\n"
            f"grades: {grades}\n"
            #f"required courses: {kdams}\n"
        )

        # the actual text
        content = getattr(doc, "page_content", str(doc))
        out_str += f"passage: {content}\n\n"

    return out_str


def reviews_from_sql(sql_result, docs_split):
    ids = set()
    reviews = []
    
    # convert IDs and collect them
    for row in sql_result:
        row['id'] = int(row['id'])
        ids.add(str(row['id']))
    
    # track how many reviews we added per id
    count_per_id = defaultdict(int)

    # iterate through docs_split and collect max 2 reviews per id
    for doc in docs_split:
        cid = doc.metadata['course_id']

        if cid in ids and count_per_id[cid] < 2:
            reviews.append(doc)
            count_per_id[cid] += 1

    return reviews

## Optional; Reorders longer documents to center of output text
#long_reorder = RunnableLambda(LongContextReorder().transform_documents)