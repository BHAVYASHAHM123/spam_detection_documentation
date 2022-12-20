# Approach 

!!! Approach
    Machine learning methods of recent are being used to successfully detect
    and filter spam emails. Some Methods are Content Based Filtering Technique, Case
    Base Spam Filtering Method, Heuristic or Rule Based Spam Filtering Technique or
    Previous Likeness Based Spam Filtering Technique.

??? note "Diagram"

    ``` mermaid
    graph LR

    A[Spam] ---> C{Feature Extraction};
    B[Ham] ---> C{Feature Extraction};
    C --->|Naive Bayes| D[Score Based Spam Detection]
    D ---> E{It's Spam Message};
    D ---> E{It's Ham Message};
    ```
