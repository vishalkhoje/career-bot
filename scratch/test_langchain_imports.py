try:
    from langchain.retrievers.ensemble import EnsembleRetriever
    print("Imported EnsembleRetriever successfully")
except Exception as e:
    print(f"Failed to import EnsembleRetriever: {e}")

try:
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    print("Imported ContextualCompressionRetriever successfully")
except Exception as e:
    print(f"Failed to import ContextualCompressionRetriever: {e}")
