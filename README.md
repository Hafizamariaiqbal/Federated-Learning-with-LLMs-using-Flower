# Federated Learning with LLMs using Flower & Hugging Face 🤖🌍

This project demonstrates **federated fine-tuning of a pre-trained LLM** (`DistilBERT`) for **text classification (sentiment analysis)** using the [Flower](https://flower.dev) federated learning framework.

It uses the **IMDB dataset** and simulates multiple clients training locally without sharing their data.

## 🧠 Features
- Federated learning using Flower (FedAvg strategy)
- Transformer model: `distilbert-base-uncased`
- Text classification task (IMDB sentiment)
- Evaluation metrics (accuracy per round)

## 🛠️ Requirements

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

Open multiple terminals for simulation.

### Terminal 1 – Run FL Server:
```bash
python server.py
```

### Terminal 2+ – Run FL Clients (you can run multiple):
```bash
python client.py
```

## 📊 Example Results

| Round | Accuracy |
|-------|----------|
| 1     | 0.78     |
| 2     | 0.82     |
| 3     | 0.85     |

## 📂 Files Overview

- `client.py`: Local model training and evaluation logic
- `server.py`: Federated server with aggregation strategy
- `utils.py`: Dataset loading, preprocessing, tokenizer/model setup
- `requirements.txt`: List of dependencies
- `LICENSE`: MIT License

## 🔐 Future Work

- Add **Differential Privacy** (e.g., using Opacus)
- Use real distributed clients (on different machines)
- Add **Secure Aggregation**
- Try larger LLMs (BERT, RoBERTa, LLaMA, etc.)

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 👩‍💻 Author

**Your Name**  
Researcher | Federated Learning | NLP | Privacy  
[GitHub Profile](https://github.com/your-username)