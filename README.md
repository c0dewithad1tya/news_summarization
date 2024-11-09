# news_summarization
Objective:
To scrape news articles from multiple sources, categorize them, summarize related articles together to reduce bias, and assess the political leaning of political news articles, storing the data in Elasticsearch.

1. Scrape News Articles
Use: Scrapy framework to collect news from at least 15 different websites.

Scope: Scrape all categories of news (politics, sports, technology, local, global, etc.).

Data Collected:

URL

Title

Article content

Timestamp

2. Pre-process and Store Articles
Caching: Temporarily store a batch of articles (e.g., 50 articles) for grouping and processing.

Batch Processing: Once a certain number of articles are cached, start clustering and processing.

3. Calculate Cosine Similarity
Convert Articles to Vectors:

Use a sentence embedding model (e.g., SentenceTransformer) to convert article content into vector embeddings.

Compute Cosine Similarity:

Measure the similarity between each pair of articles using cosine similarity to identify which articles discuss the same or closely related topics.

4. Cluster Similar Articles
Clustering Method: Agglomerative Clustering.

Group articles that are semantically similar based on cosine similarity scores.

Automatically determine the number of clusters by specifying a distance threshold for merging.

Result: Each cluster contains articles discussing the same or similar events.

5. Summarize and Categorize Clusters
Combine Content: For each cluster, combine the text of all articles.

Use GPT to Summarize:

Summarize the combined content into 100 words.

Categorize the Summary:

GPT categorizes the summary into predefined categories (e.g., Politics, Sports, Technology, Economy, Culture, General).

Result: Each cluster receives a summary and a category.

6. Political Bias Scoring (Only for Political Articles)
For Political News Only:

Use GPT to assign a political bias score for individual articles in clusters that are categorized as Politics.

Score range: 0 (Far Left) to 10 (Far Right).

No Bias Scoring for non-political news (e.g., Sports, Technology, etc.).

7. Store Data in Elasticsearch
Index the Following Data:

URLs and titles of all articles in the cluster.

Combined summary of the articles.

Category (Politics, Sports, etc.).

List of political bias scores for individual articles (if applicable).

Timestamp.

Data Storage: Elasticsearch will store all this data for retrieval and display in the news application.
