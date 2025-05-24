import argparse
# from langchain_openai import OpenAIEmbeddings
# from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # 🧠 Use HuggingFace for Embeddings
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 🔍 Load Chroma DB with HF Embeddings
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # 🔎 Perform similarity search
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    # 🧱 Prepare context for prompting
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("\n=== Prompt ===\n")
    print(prompt)

    # 🔁 Use Hugging Face model for answering
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto", torch_dtype="auto")

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    response = generator(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
    response_text = response[0]['generated_text']

    # 📚 Output sources
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"\nResponse:\n{response_text}\n\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
