import flwr as fl
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from utils import get_model, get_tokenizer, load_and_prepare_data

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.dataset.items() if key in ['input_ids', 'attention_mask']}
        item["labels"] = torch.tensor(self.dataset["label"][idx])
        return item

class FLClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = get_model()
        self.tokenizer = get_tokenizer()
        train_data, test_data = load_and_prepare_data()
        self.train_loader = DataLoader(IMDbDataset(train_data), batch_size=8, shuffle=True)
        self.test_loader = DataLoader(IMDbDataset(test_data), batch_size=8)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        for batch in self.train_loader:
            optimizer.zero_grad()
            outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"}, labels=batch["labels"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in self.test_loader:
                outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"})
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
        accuracy = correct / total
        return float(accuracy), len(self.test_loader.dataset), {"accuracy": accuracy}

fl.client.start_numpy_client(server_address="localhost:8080", client=FLClient())