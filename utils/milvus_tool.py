from pymilvus import (
                        connections,
                        utility,
                        FieldSchema,
                        CollectionSchema,
                        DataType,
                        Collection,
                    )
import pickle
import numpy as np

def init_Milvus(new_data):
    if new_data:
        # Connect to Milvus
        connections.connect("default", host="localhost", port="19530")

        # Define the collection name
        collection_name = "example_collection2"

        # Load data from pickle file
        with open('../sample.pkl', 'rb') as f:
            data = pickle.load(f)

        # Extract names and embeddings from the data
        names = [item['name'] for item in data]
        embeddings = [item['emb'] for item in data]

        # Verify the dimension and type of the embeddings
        dim = 512  # Expected dimension
        assert all(emb.shape == (dim,) for emb in embeddings), f"All embeddings must have shape ({dim},)"
        assert all(emb.dtype == np.float32 for emb in embeddings), "All embeddings must be float32"

        # Define the collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields, "Example collection")

        # Create the collection
        collection = Collection(collection_name, schema)

        # Create an IVF_FLAT index for the collection using IP metric
        index_params = {
            "metric_type": "IP",
            "index_type": "GPU_IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index("embedding", index_params)

        # Load the collection for search
        collection.load()

        # Insert the data into the collection
        collection.insert([names, embeddings])
        collection.flush()
        
        print("****Create new data successful****")
        
    else:
        # Connect to Milvus
        connections.connect("default", host="localhost", port="19530")
        collection = Collection(name = 'example_collection2')
        collection.load()
        
    return collection

if __name__ == "__main__":
    init_Milvus(True)
