#Run code step-by-step:

mkdir llamaindex-hello
cd llamaindex-hello
python3 --version
pip3 install pipenv (is python virtualenv managment tool)
ls
pipenv shell
pipenv install llama-index python.env
#touch main.py .env
#check Pipfile and Pipfile.lock
pipenv install requests beautifulsoup4
python3 dl-docs.py
pipenv install llama-index
pipenv install openai
pipenv install pinecone-client
pipenv install python-dotenv
pipenv install llama-index-vector-stores-pinecone
pipenv install unstructured
pipenv install black
pipenv install tqdm black
#check pipfile to see all packeges are installed
python3 ingestion.py
pipenv install streamlit
