import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────────
sql_queries = [
    "מה הציון הממוצע בקורס מבני נתונים?",
    "מה הציונים בשנים קודמות של מבני נתונים?",
    "איך השתנה הציון הממוצע בקורס לאורך השנים?",
    "איזה קורסים עם הציון הממוצע הכי נמוך?",
    "דירוג קורסים לפי ממוצע ציונים",
    "מה הציון הממוצע של דני קרן בחדוא 1 בשנה שעברה?",
    "האם יש שיפור בציונים בקורס אלגוריתמים עם משה לוי?",
    "עשרת הממוצעים הכי גבוהים בקורס למידה עמוקה עם ריטה ב 3 שנים אחרונות",
    "מה הקורסים שצריך לפני מבני נתונים?",
    "מהם הקדמים של אלגוריתמים?",
    "איזה קורסים חובה לפני קורס מסוים?",
    "מה צריך ללמוד לפני קורס מערכות הפעלה?",
    "מה הקדמים של מבוא למדעי המחשב?",
    "האם לינארית 1 היא קדם לאנליזה 2?",
    "What is the average grade in Databases?",
    "SELECT the top 5 courses with the highest average grade",
    "How many students passed the AI course?",
    "List all students who failed Operating Systems",
    "מה הקורס עם מספר המתקדמים הגבוה ביותר?",
    "List courses that require Calculus as a prerequisite",
    "How many courses require Algebra 2 as prerequisite?",
    "מהם הקורסים המצריכים לאלגוריתמים?",
    "What is the maximum average in Computer Networks?",
    "Show students with grade above 90 in Databases",
    "מה הם הממוצעים המינימליים בקורס מערכות הפעלה?",
    "List all courses that have no prerequisites",
    "What are the prerequisites for reinforcement learning",
    "מה הציון הממוצע בקורס אנליזה 1?",
    "מה הממוצע בחדוא 2 השנה?",
    "מה הציון הממוצע בקורס תכנות מונחה עצמים?",
    "הצג את הממוצע של כל הקורסים בשנה האחרונה",
    "מה הציון המקסימלי שהושג בקורס בסיסי נתונים?",
    "מה הציון המינימלי בקורס רשתות תקשורת?",
    "כמה סטודנטים נכשלו בקורס מבוא לתכנות?",
    "כמה סטודנטים עברו בקורס מבני נתונים בשנת 2023?",
    "מה שיעור ההצלחה בקורס אלגוריתמים?",
    "מה שיעור הנכשלים בקורס חשבון דיפרנציאלי?",
    "הצג את כל הציונים של המרצה יוסי כהן בקורס מבוא לתכנות",
    "מה הציון הממוצע של מרים לוי בקורס בינה מלאכותית?",
    "מה הציון הממוצע בקורסים של פרופסור אברהם?",
    "השווה ציונים ממוצעים בין שנת 2022 לשנת 2023 בקורס מבני נתונים",
    "האם הציון הממוצע בקורס למידת מכונה עלה בין 2021 ל-2023?",
    "מה הממוצע המשוקלל של כל קורסי חובה?",
    "מה הממוצע בחלוקה לפי סמסטר א וסמסטר ב?",
    "מה אחוז הסטודנטים עם ציון מעל 85 בקורס מבוא לבינה מלאכותית?",
    "כמה סטודנטים קיבלו ציון בין 70 ל-80 בקורס אלגוריתמים?",
    "הצג סטודנטים עם ציון מתחת ל-60 בקורס רשתות",
    "What is the minimum grade recorded in Linear Algebra?",
    "What is the pass rate for Operating Systems?",
    "How many students scored above 95 in Machine Learning?",
    "Show the average grade per year for Data Structures",
    "What is the average grade for all mandatory courses?",
    "Which year had the highest average in Computer Networks?",
    "How many students failed Calculus 2?",
    "List courses where the average grade dropped between 2022 and 2023",
    "What is the average grade of students taught by Prof. Cohen?",
    "Show me all grades above 80 in the AI course",
    "What percentage of students passed Deep Learning last year?",
    "How many students enrolled in Databases in 2022?",
    "What is the median grade in Algorithms?",
    "Show the grade distribution for Operating Systems",
    "Which semester had a higher average in Linear Algebra?",
    "What is the failure rate in Discrete Mathematics?",
    "List all courses with an average grade below 70",
    "How did the average grade in Networks change over 3 years?",
    "What is the highest average grade among all courses?",
    "Count students who got exactly 100 in any course",
    "אילו קורסים מצריכים את קורס מבוא לתכנות?",
    "האם חשבון 1 הוא קדם הכרחי לחשבון 2?",
    "מה הקדמים הישירים של קורס בינה מלאכותית?",
    "כמה קדמים יש לקורס למידת מכונה?",
    "אילו קורסים ניתן לקחת ללא שום קדמים?",
    "מה הרשימה המלאה של קדמי קורס מערכות מבוזרות?",
    "מה שרשרת הקדמים עד קורס למידה עמוקה?",
    "כמה קורסים דורשים את לינארית 2 כקדם?",
    "האם קורס רשתות מצריך קדמים?",
    "What courses have no prerequisites at all?",
    "Is Linear Algebra 1 a prerequisite for Machine Learning?",
    "How many prerequisites does Deep Learning have?",
    "List all courses that require Data Structures as a prerequisite",
    "What is the full prerequisite chain for Reinforcement Learning?",
    "Which courses directly require Calculus 1?",
    "Does Operating Systems have more prerequisites than Networks?",
    "Is Probability a prerequisite for Machine Learning?",
    "How many courses require Algorithms as a prerequisite?",
    "List prerequisites for all fourth-year courses",
    "דרג את כל הקורסים לפי ציון ממוצע מהגבוה לנמוך",
    "מה 5 הקורסים עם שיעור ההצלחה הגבוה ביותר?",
    "מה 3 הקורסים הקשים ביותר לפי אחוז נכשלים?",
    "Rank all lecturers by average student grade",
    "Which course has the most students enrolled?",
    "What is the top-scoring course overall?",
    "List the bottom 10 courses by pass rate",
    "Which lecturer has the highest average grade across all courses?",
    "מה הקורס עם הכי הרבה נכשלים?",
    "מי המרצה עם הממוצע הגבוה ביותר?",
    "מה ה-average grade בקורס Databases?",
    "כמה students עברו את קורס AI?",
    "מה ה-pass rate של קורס Machine Learning?",
    "List הקורסים עם prerequisites ב-CS department",
    "מה הציון ה-maximum בקורס Deep Learning?",
    # Hebrew
    "כמה סטודנטים קיבלו מעל 90 במדעי הנתונים?",
    "מה ממוצע הציונים בסמסטר אביב אשתקד?",
    "אילו קורסים הם חובה לשנה ג?",
    "הצג את התפלגות הציונים בקורס פיזיקה 1",
    "מה ציון המעבר בקורס מתמטיקה בדידה?",
    "כמה נקודות זכות מקבלים על אלגוריתמים?",
    "האם יש דרישת קדם לקורס גרפיקה ממוחשבת?",
    "מה הציון החציוני בבחינה האחרונה של מערכות מבוזרות?",
    
    # English
    "What is the average score in Computer Architecture?",
    "Show me the prerequisite tree for Software Engineering.",
    "Count the number of students who failed Physics 1.",
    "List all students with a GPA above 85.",
    "What is the standard deviation of grades in Calculus 1?",
    "Which courses require Introduction to Programming?",
    
    # Mixed
    "מה ה-pass rate בקורס Linear Algebra?",
    "כמה credits מקבלים על קורס Databases?",
    "האם צריך לעשות את ה-prerequisites לפני סמסטר ב?",
    "Show me the ממוצע for Data Structures."
]

semantic_queries = [
    "חוות דעת על קורס מבני נתונים",
    "מה הסטודנטים אומרים על קורס למידה עמוקה?",
    "Why do students struggle with Computer Networks?",
    "What do students think about Artificial Intelligence?",
    "מרצים מומלצים",
    "חוות דעת על דני קרן",
    "האם קורס אלגוריתמים קשה?",
    "איך אני יכול לשפר את הציונים שלי בקורסים?",
    "What do students think about Databases course?",
    "Which professor is recommended for AI course?",
    "How difficult is Operating Systems?",
    "מה דעתכם על הקורס מבוא למבני נתונים?",
    "Which courses in the prerequisites table are most interesting?",
    "מה חוות הדעת של הסטודנטים על הקורס למידת מכונה?",
    "Is Computer Networks course more practical than Database Systems?",
    "Which courses require more preparation according to students?",
    "מהם הקורסים הכי מומלצים ללימודי הסמינר?",
    "Explain why the AI course is challenging",
    "How do students rate the Algebra 1 course?",
    "מהן ההמלצות לגבי פרויקט גמר בקורס למידה עמוקה?",
    "Which courses are easier to pass",
    "מה כדאי לדעת לפני שמתחילים את קורס תורת המשחקים?",
    "קורסים עם תרגילי בית קשים",
    "קורסים עם תרגילי בית שצריכים השקעה ומאמץ",
    "חוות דעת על קורס מבוא לחומרה",
    "מרצים טובים ומומלצים",
    "קורסים שמסתכלים עליהם בעבודה",
    "איזה קורס נחשב קשה יותר לפי ביקורות?",
    "מה יותר מומלץ – קורס A או קורס B?",
    "איזה קורס מקבל ביקורות חיוביות יותר?",
    "Compare student satisfaction between two courses",
    "Is Course A more recommended than Course B?",
    "מי נחשב ברור יותר בהסברים?",
    "השוואת סגנון הוראה בין שני מרצים",
    "מי נחשב קשוח יותר בבדיקות?",
    "איזה מרצה נחשב מעביר חומר בצורה מעניינת יותר?",
    "איזה מרצה מועדף על ידי הסטודנטים?",
    "איך הקורס אלגוריתמים שונה אצל מרצה A לעומת מרצה B?",
    "איזה מרצה מלמד את הקורס בצורה קלה יותר להבנה?",
    "Which instructor improves the course experience?",
    "עם מי עדיף לקחת את הקורס X, עם מרצה A או מרצה B?",
    "תשווה לי בין מרצה א ומרצה ב בקורס X",
    "תשווה לי בין מרצה א לבין מרצה ב בקורס X",
    "האם עדיף לעשות את הקורס X עם מרצה Y?",
    "עד כמה קורס מבני נתונים קשה?",
    "האם כדאי לקחת בינה מלאכותית בשנה ב?",
    "מה הקורסים שכדאי לקחת בשנה הראשונה?",
    "האם קורס רשתות תקשורת מעניין?",
    "מה הקורסים שהכי שווים ללמוד?",
    "האם קורס תכנות מונחה עצמים מומלץ?",
    "מה הקורסים שהכי שימושיים בתעשייה?",
    "האם כדאי לקחת קורס אבטחת מידע?",
    "מה החוויה של סטודנטים בקורס מערכות הפעלה?",
    "האם קורס למידה עמוקה מתאים למתחילים?",
    "האם הקורס מחייב הרבה עבודה עצמאית?",
    "מה הקורסים שדורשים הכי הרבה זמן?",
    "האם קורס גרפים קל יחסית?",
    "מה הקורסים עם הכי פחות עבודת בית?",
    "האם יש קורסים שניתן ללמוד עליהם לבד?",
    "מה הסטודנטים אומרים על עומס הקורס?",
    "האם קורס בינה מלאכותית מעניין לדעת הסטודנטים?",
    "מה הקורסים שהכי כיף ללמוד?",
    "קורסים שנחשבים לפתוחים ולא קשיחים",
    "מה הקורס הכי מאתגר בתואר?",
    "Is Machine Learning suitable for second-year students?",
    "What courses do students find most enjoyable?",
    "Is the Algorithms course considered boring or engaging?",
    "Which courses have the heaviest workload according to students?",
    "Are there any courses considered easy by most students?",
    "Is Deep Learning too advanced for undergraduates?",
    "What do students say about the pace of the AI course?",
    "Which course is most relevant for industry jobs?",
    "Is Operating Systems worth taking as an elective?",
    "What do students think about the difficulty of Calculus 2?",
    "How engaging is the Computer Vision course?",
    "Is Probability considered a tough course by students?",
    "What courses do students recommend for practical skills?",
    "How is the workload in Networks compared to Databases?",
    "Do students feel supported in the Machine Learning course?",
    "Is Discrete Mathematics considered dry or interesting?",
    "Which courses prepare you best for a software engineering job?",
    "What do students say about course X being too theoretical?",
    "Are lab sessions in Computer Architecture helpful according to reviews?",
    "What is the general impression of the Software Engineering course?",
    "איך הסטודנטים מתארים את שיטת ההוראה של פרופסור כהן?",
    "האם ד\"ר לוי מסביר בצורה ברורה?",
    "מה חוות הדעת על המרצה של קורס רשתות?",
    "האם המרצה של בינה מלאכותית מגיב לשאלות?",
    "מה הסטודנטים אומרים על הדרך שבה מרצה X מלמד?",
    "האם מרצה Y ידוע כקשוח בבחינות?",
    "איזה מרצה מסביר הכי טוב חומר קשה?",
    "מה הסטודנטים חושבים על הזמינות של המרצה?",
    "האם פרופסור X מאורגן בהרצאות?",
    "מי הכי מומלץ ללמד קורס למידת מכונה?",
    "Is Prof. Cohen known for being strict in exams?",
    "Which lecturer explains complex topics most clearly?",
    "What do students say about the teaching style of Dr. Levi?",
    "Is the AI lecturer responsive to student questions?",
    "Which professor is most organized in their lectures?",
    "Who is the most popular lecturer in the CS department?",
    "Do students prefer lecturer A or lecturer B for Algorithms?",
    "What is the general opinion of the Databases lecturer?",
    "Is the Networks professor known for difficult exams?",
    "Which lecturer provides the best feedback on assignments?",
    "איך כדאי להתכונן לבחינה בקורס מבני נתונים?",
    "מה העצות של סטודנטים לקורס אלגוריתמים?",
    "האם כדאי ללמוד עם חברים בקורס מערכות הפעלה?",
    "מה האסטרטגיה הכי טובה ללמוד למבחן בחשבון?",
    "האם יש ספרים מומלצים לקורס בינה מלאכותית?",
    "מה הטיפים הכי שימושיים לעבור קורס רשתות?",
    "האם כדאי ללכת לכל ההרצאות בקורס מבני נתונים?",
    "מה הסטודנטים ממליצים לגבי תרגילי הבית?",
    "איך מתמודדים עם הקורסים הקשים בשנה ב?",
    "מה הדברים שהסטודנטים מאחלים שהיו יודעים לפני הקורס?",
    "What study tips do students recommend for Algorithms?",
    "How should I prepare for the Machine Learning exam?",
    "Is it worth attending all lectures in Databases?",
    "What resources do students suggest for Deep Learning?",
    "How do students manage the workload in Operating Systems?",
    "Any advice for a first-year student taking Data Structures?",
    "What do students wish they knew before taking AI?",
    "Is self-study enough for passing Computer Networks?",
    "Should I take Algorithms before or after Data Structures?",
    "What books do students recommend for the AI course?",
    "מה ההבדל בין קורס X לקורס Y לפי סטודנטים?",
    "איזה קורס יותר מעשי, בסיסי נתונים או רשתות?",
    "מה יותר שימושי בתעשייה, למידת מכונה או בינה מלאכותית?",
    "האם למידה עמוקה מעניינת יותר מלמידת מכונה לפי ביקורות?",
    "Compare reviews of Data Structures vs Algorithms",
    "Which is more theoretical, AI or Machine Learning according to students?",
    "Do students prefer practical or theoretical courses?",
    "Is Computer Vision more interesting than NLP according to reviews?",
    "מה יותר מאתגר, מבני נתונים או אלגוריתמים?",
    "Compare student experience in Operating Systems vs Networks",
    "מה הסטודנטים אומרים על ה-workload של קורס Deep Learning?",
    "האם ה-lecturer של קורס AI מסביר בצורה ברורה?",
    "Is קורס מבני נתונים considered hard?",
    "What do students say about קורס אלגוריתמים?",
    "מה ה-reviews על קורס Databases?",
    "האם קורס Machine Learning מומלץ?",
    "Which קורסים are most interesting according to students?",
    "מה הטיפים ל-exam בקורס Networks?",
    # Hebrew
    "האם הקורס במדעי הנתונים עמוס מדי?",
    "מי מרצה יותר טוב לחדווא, יוסי או דני?",
    "מה אומרים על פרויקט הגמר בהנדסת תוכנה?",
    "האם שווה להשקיע בנוכחות בהרצאות של קורס רשתות?",
    "איזה קורס בחירה נחשב הכי קליל?",
    "מה רמת הקושי של המטלות בתכנות מונחה עצמים?",
    "האם המבחן של פרופסור כהן הוגן?",
    "שתפו חוויות מקורס אבטחת מידע",
    
    # English
    "Are the assignments in Computer Architecture too long?",
    "Which professor gives better grades in Physics?",
    "Is the Software Engineering project worth the effort?",
    "How harsh is the grading in the AI course?",
    "What is the general vibe of the Advanced Algorithms class?",
    "Do students recommend taking NLP and Vision in the same semester?",
    
    # Mixed
    "האם ה-assignments בקורס AI קשים?",
    "מה ה-feedback על המרצה של Deep Learning?",
    "איך ה-workload בקורס Operating Systems לעומת Networks?",
    "Is the קורס considered a GPA booster?"
]

# ─────────────────────────────────────────────────────────
#  BUILD DATASET
# ─────────────────────────────────────────────────────────
texts  = sql_queries + semantic_queries
labels = [1] * len(sql_queries) + [0] * len(semantic_queries)   # 1=sql, 0=semantic

print(f"Dataset: {len(sql_queries)} SQL  +  {len(semantic_queries)} semantic  =  {len(texts)} total")
print(f"Class balance: {labels.count(1)/len(labels)*100:.1f}% SQL / {labels.count(0)/len(labels)*100:.1f}% semantic\n")

# ─────────────────────────────────────────────────────────
#  PIPELINE
#  char-level n-grams work well for mixed-script text
#  (Hebrew chars + Latin chars + digits)
# ─────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char_wb",        # character n-grams with word boundaries
        ngram_range=(2, 4),        # bi- to 4-grams
        min_df=1,
        sublinear_tf=True,
        strip_accents=None,        # keep Hebrew niqqud if present
    )),
    ("clf", LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
    )),
])

# ─────────────────────────────────────────────────────────
#  CROSS-VALIDATION  (5-fold stratified)
# ─────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    pipeline, texts, labels,
    cv=cv,
    scoring=["accuracy", "f1", "precision", "recall"],
    return_train_score=True,
)

print("── 5-Fold Cross-Validation Results ──────────────────")
for metric in ["accuracy", "f1", "precision", "recall"]:
    test_scores  = cv_results[f"test_{metric}"]
    train_scores = cv_results[f"train_{metric}"]
    print(f"  {metric:<12}  val: {test_scores.mean():.3f} ± {test_scores.std():.3f}   "
          f"train: {train_scores.mean():.3f} ± {train_scores.std():.3f}")

# ─────────────────────────────────────────────────────────
#  TRAIN ON FULL DATASET
# ─────────────────────────────────────────────────────────
pipeline.fit(texts, labels)
print("\n── Full-dataset training complete ───────────────────")

train_preds = pipeline.predict(texts)
print("\nClassification report (train set):")
print(classification_report(labels, train_preds, target_names=["semantic", "sql"]))

cm = confusion_matrix(labels, train_preds)
print("Confusion matrix (rows=actual, cols=predicted):")
print(f"           pred:semantic  pred:sql")
print(f"act:semantic     {cm[0,0]:>5}       {cm[0,1]:>5}")
print(f"act:sql          {cm[1,0]:>5}       {cm[1,1]:>5}")

# ─────────────────────────────────────────────────────────
#  SAVE MODEL
# ─────────────────────────────────────────────────────────
joblib.dump(pipeline, "query_classifier.joblib")
print("\n✓ Model saved to query_classifier.joblib")

# ─────────────────────────────────────────────────────────
#  QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────
test_cases = [
    ("מה הציון הממוצע בקורס מבני נתונים?",          "sql"),
    ("חוות דעת על קורס מבני נתונים",               "semantic"),
    ("How many students failed Operating Systems?",  "sql"),
    ("Which professor is best for AI?",              "semantic"),
    ("מה הקדמים של אלגוריתמים?",                   "sql"),
    ("האם קורס רשתות מעניין?",                      "semantic"),
    ("What is the pass rate for Databases?",         "sql"),
    ("Is Deep Learning suitable for beginners?",     "semantic"),
]

label_map = {1: "sql", 0: "semantic"}
print("\n── Sanity check ─────────────────────────────────────")
all_correct = True
for query, expected in test_cases:
    pred_label = label_map[pipeline.predict([query])[0]]
    prob       = pipeline.predict_proba([query])[0]
    confidence = max(prob) * 100
    status     = "✓" if pred_label == expected else "✗"
    if pred_label != expected:
        all_correct = False
    print(f"  {status} [{confidence:5.1f}%]  {pred_label:<10}  {query[:55]}")

print(f"\n{'All sanity checks passed!' if all_correct else 'Some checks failed — consider adding more training data.'}")
