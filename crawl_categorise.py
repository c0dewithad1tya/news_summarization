import openai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import numpy as np
from elasticsearch import Elasticsearch
from datetime import datetime

class NewsPipeline:

    def __init__(self):
        # Elasticsearch setup
        self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
        openai.api_key = 'YOUR_OPENAI_API_KEY'
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Pretrained sentence embedding model
        self.article_cache = []  # Cache to hold articles for clustering

    def process_item(self, item, spider):
        # Store the article in cache for clustering
        self.article_cache.append({
            'url': item['url'],
            'title': item['title'],
            'content': item['content']
        })

        # Check if cache has enough articles to start clustering
        if len(self.article_cache) >= 50:
            self.cluster_and_process_articles()
            self.article_cache = []  # Reset the cache for the next batch

        return item

    def cluster_and_process_articles(self):
        # Step 1: Generate embeddings for articles
        article_texts = [article['content'] for article in self.article_cache]
        embeddings = self.model.encode(article_texts)

        # Step 2: Compute cosine similarity and cluster the articles
        similarity_matrix = cosine_similarity(embeddings)
        clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=0.5)
        labels = clustering.fit_predict(1 - similarity_matrix)  # 1 - cosine similarity = distance

        # Step 3: Group articles by clusters
        grouped_articles = {}
        for idx, label in enumerate(labels):
            if label not in grouped_articles:
                grouped_articles[label] = []
            grouped_articles[label].append(self.article_cache[idx])

        # Step 4: Process each group of articles (summarize, categorize, score political bias if needed)
        for group, articles in grouped_articles.items():
            combined_summary, category = self.summarize_and_categorize(articles)

            # Index combined summary in Elasticsearch
            self.index_combined_summary(articles, combined_summary, category)

    def summarize_and_categorize(self, articles):
        # Combine the content of all articles in the group
        combined_text = " ".join([article['content'] for article in articles])

        # Use GPT to summarize and categorize the combined articles
        prompt = f"""
        Summarize the following combined news articles in 100 words:
        {combined_text}

        Classify this combined article into one of the following categories:
        - Politics
        - Sports
        - Technology
        - Economy
        - Culture
        - General

        Provide the output in this format:
        1. Summary: [Summary]
        2. Category: [Category]
        """
        
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=200
        )

        result = response.choices[0].text.strip().split('\n')
        summary = result[0].split('Summary: ')[-1]
        category = result[1].split('Category: ')[-1]
        
        # Process each article individually for political scoring if it's a political article
        if category == "Politics":
            for article in articles:
                political_score = self.score_political_bias(article['content'])
                article['political_score'] = political_score  # Add score to each article for Elasticsearch
        else:
            # If not political, no political score is added
            for article in articles:
                article['political_score'] = None
        
        return summary, category

    def score_political_bias(self, article_text):
        # Use GPT to assess political bias only for political articles
        prompt = f"""
        Based on the following article, assign a political leaning score between 0 (far-left) and 10 (far-right):
        {article_text}

        Just output the score as a single number.
        """

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=3
        )

        score = float(response.choices[0].text.strip())
        return score

    def index_combined_summary(self, articles, summary, category):
        urls = [article['url'] for article in articles]
        titles = [article['title'] for article in articles]
        political_scores = [article['political_score'] for article in articles]

        doc = {
            'urls': urls,
            'titles': titles,
            'summary': summary,
            'category': category,
            'political_scores': political_scores,  # List of individual scores, may contain None for non-political
            'timestamp': datetime.now()
        }

        # Index in Elasticsearch
        res = self.es.index(index="news_article_clusters", body=doc)
        print(f"Indexed clustered article summary: {titles[0]}... (total {len(articles)} articles), Result: {res['result']}")
