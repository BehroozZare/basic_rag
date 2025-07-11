# basic_rag

a side hobby to test rag and these stuff

---

## Downloading Models for This Project

To set up and download the required models for this project, follow these steps:

### 0. Install Required Packages

First, make sure you have the necessary Python packages installed. You will need:

- `transformers`
- `sentence-transformers`
- `huggingface_hub`
- `docling` (for model downloading utilities)
- `torch` (required by most models)

You can install them using pip:

```bash
pip install transformers sentence-transformers huggingface_hub docling torch
```

### 1. Add Your Hugging Face Token

To download models from Hugging Face, you need an access token.

- Create a folder named `config` in the root of your project (if it doesn't already exist).
- Inside the `config` folder, create a file named `token.txt`.
- Paste your Hugging Face token into `token.txt`.  
  You can get your token from: https://huggingface.co/settings/tokens

**Example:**
```
project_root/
├── config/
│   └── token.txt
```

### 2. Configure the Base Address

- Go to the `status` folder and open `config.json`.
- Change the value of `"base_address"` to the absolute path of your project’s root folder.

**Example:**
```json
{
  "base_address": "/absolute/path/to/your/project",
  ...
}
```

### 3. Download the Models

- Run the script at `src/model_downloader/download_models.py`:

```bash
python src/model_downloader/download_models.py
```

This script will:
- Log in to Hugging Face using your token.
- Download all required models (DocLing, embedding, generation, and image captioning models) as specified in your `status/config.json`.
- Save the models to the directories specified in the config.

**Note:**  
You can control which models are downloaded by setting the corresponding flags (`download_docling_models`, `download_embedding_model`, etc.) in `status/config.json`.

If the downloads are successful, an `offline_models` folder will be created in your project directory containing all the downloaded models.
