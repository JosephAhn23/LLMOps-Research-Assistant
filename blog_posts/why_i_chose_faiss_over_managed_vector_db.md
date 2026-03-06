# Why I Chose Raw FAISS Over Pinecone (And Where It Hurt)

I originally assumed I would use a managed vector database. It is the default advice: less ops overhead, easy APIs, and quick prototypes.  
I still ended up choosing raw FAISS for this project, and the reason was not ideology. It was latency control and cost transparency.

In my local pipeline, FAISS retrieval stayed under 5 ms on benchmark runs (`n=50` queries), and reranking sat around 40-47 ms. End-to-end latency was still in seconds because generation dominated, not retrieval. That mattered: if retrieval is already cheap and fast, moving it to a managed service adds network hops and another bill without solving the actual bottleneck.

The upside was clear:

- I could run vector search in-process and reason about each millisecond.
- I had no vendor lock-in while iterating index settings.
- I could shard retrieval logic inside this codebase and keep the same query interface.

The downside was also real:

- I had to manage index persistence and metadata consistency myself.
- Concurrency bugs were my problem (I had to add lock protection around read-modify-write index updates in `ingestion/pipeline.py`).
- Operational ergonomics were weaker than managed systems: fewer built-in dashboards, fewer safety rails.

The lesson is not "managed DBs are bad." The lesson is to optimize for your bottleneck and team constraints.  
If I had high-velocity real-time updates, multi-tenant isolation needs, or strict uptime SLOs from day one, I would likely choose a managed vector store and pay the tax.  
For this project's workload, raw FAISS gave me faster iteration and clearer performance ownership.

A second lesson: benchmark claims without methodology are noise. I now attach sample sizes and measurement context to every number because "3 ms" without workload details is just marketing.

