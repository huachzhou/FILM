# Self-Monitoring Large Language Models for Click-Through Rate Prediction

The sample data from Amazon Movies and example fine-tuning code for FILM and Base LLMs. Complete code and data will be released upon paper acceptance.

## Performance

On this sample data with Llama-7b-hf:
- **FILM performance**: 
  - AUC: 0.8670 
  - Logloss: 0.2796 
  - (3 pre-training and 4 fine-tuning epochs)
- **Base LLMs**: 
  - AUC: 0.8485 
  - Logloss: 0.2945 
  - (4 fine-tuning epochs)
  
Please run them on A100/A800 80/40G GPU. 

## Setup

The python packages are in the `requirement.txt` and you could install necessary ones.

## Running the Code

To run the code:

```bash
./shell/instruct_film.sh GPU_id
```
Model Type could be FILM or Base.

## 📮 Contact
For any questions or feedback, please leave a issue or contact:
- Huachi Zhou - huachi.zhou@connect.polyu.hk
