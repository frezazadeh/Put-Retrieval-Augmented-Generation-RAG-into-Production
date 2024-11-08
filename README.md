# Project Setup Instruction

Follow these steps to set up your environment and install all necessary dependencies:

1. **Install `pipenv`**, the Python virtual environment management tool:
   ```bash
   pip3 install pipenv
   ```

2. **Activate the `pipenv` shell** to create a new virtual environment:
   ```bash
   pipenv shell
   ```

3. **Install primary dependencies (`llama-index` and `python.env`)**:
   ```bash
   pipenv install llama-index python.env
   ```

4. **Install additional packages for HTTP requests and HTML parsing**:
   ```bash
   pipenv install requests beautifulsoup4
   ```

5. **Run the document downloader script**:
   ```bash
   python3 dl-docs.py
   ```

6. **Reinstall or update `llama-index` if necessary**:
   ```bash
   pipenv install llama-index
   ```

7. **Install the OpenAI API client**:
   ```bash
   pipenv install openai
   ```

8. **Install the Pinecone client for vector storage**:
   ```bash
   pipenv install pinecone-client
   ```

9. **Install `python-dotenv` to manage environment variables**:
   ```bash
   pipenv install python-dotenv
   ```

10. **Install the Pinecone plugin for `llama-index`**:
    ```bash
    pipenv install llama-index-vector-stores-pinecone
    ```

11. **Install `unstructured` for handling complex data inputs**:
    ```bash
    pipenv install unstructured
    ```

12. **Install the code formatter `black`**:
    ```bash
    pipenv install black
    ```

13. **Install `tqdm` for progress bars and `black` (if not installed)**:
    ```bash
    pipenv install tqdm black
    ```

14. **Run the ingestion script**:
    ```bash
    python3 ingestion.py
    ```

15. **Install `streamlit` for building web applications**:
    ```bash
    pipenv install streamlit
    ```

16. **Install `streamlit` for building web applications**:
    ```bash
    streamlit run main.py
    ```
Now you are all set up! 🎉

